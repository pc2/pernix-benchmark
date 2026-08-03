"""Generate reproducible in-core instruction-throughput models."""

from __future__ import annotations

import csv
import json
import logging
import math
import os
import re
import shutil
import statistics
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)

MODEL_COLUMNS = [
    "operation",
    "isa",
    "bit_width",
    "blocks_per_second",
    "cpu_frequency_hz",
]
DIAGNOSTIC_COLUMNS = [
    "operation",
    "isa",
    "bit_width",
    "model_kind",
    "llvm_cpu",
    "block_rthroughput_cycles",
    "model_instructions_per_block",
    "model_uops_per_block",
    "model_ipc",
    "retired_instructions_per_block",
    "measured_cycles_per_block",
    "cpu_frequency_hz",
    "blocks_per_second",
    "probe_iterations",
    "control_probe_iterations",
    "frequency_source",
    "compiler",
    "compiler_flags",
    "llvm_mca_version",
]

_REGION_RE = re.compile(
    r"^model_(?P<operation>compression|decompression)_"
    r"(?P<isa>avx2|avx512vbmi)_(?P<width>\d+)$"
)
_ISA = {
    "avx2": ("AVX2", ["-march=x86-64-v3"], "PERNIX_MODEL_ISA_AVX2"),
    "avx512vbmi": (
        "AVX-512-VBMI",
        ["-march=x86-64-v4", "-mavx512vbmi"],
        "PERNIX_MODEL_ISA_AVX512VBMI",
    ),
}
_MODEL_OVERLAY_HEADERS = (
    "pernix/x86/avx2/avx2_compression.h",
    "pernix/x86/avx2/avx2_decompression.h",
    "pernix/x86/avx512vbmi/avx512vbmi_compression.h",
    "pernix/x86/avx512vbmi/avx512vbmi_decompression.h",
)


class ModelGenerationError(RuntimeError):
    """Raised when the instruction model cannot be generated safely."""


@dataclass(frozen=True)
class ModelOptions:
    results_dir: Path
    build_dir: Path
    compiler: str = "g++"
    llvm_mca: str = "llvm-mca"
    likwid_perfctr: str | None = None
    cpu_frequency_hz: float | None = None
    llvm_cpu: str | None = None
    mca_iterations: int = 200
    probe_min_seconds: float = 1.0


def _executable(command: str, label: str) -> str:
    resolved = shutil.which(command)
    if resolved is None:
        raise ModelGenerationError(f"{label} executable not found: {command}")
    return resolved


def _run(
    command: list[str], *, cwd: Path, capture_stderr: bool = False
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
            env={**os.environ, "LC_ALL": "C"},
        )
    except (OSError, subprocess.CalledProcessError) as error:
        detail = ""
        if isinstance(error, subprocess.CalledProcessError):
            detail = (error.stderr if capture_stderr else error.stdout) or error.stderr
        raise ModelGenerationError(
            f"Command failed: {' '.join(command)}"
            + (f"\n{detail.strip()}" if detail else "")
        ) from error


def _prepare_model_include_overlay(include: Path, artifact_dir: Path) -> Path:
    """Create model-only header copies without changing the Pernix checkout."""

    overlay_include = artifact_dir / "include-overlay"
    for relative in _MODEL_OVERLAY_HEADERS:
        source = include / relative
        destination = overlay_include / relative
        try:
            contents = source.read_text(encoding="utf-8")
        except OSError as error:
            raise ModelGenerationError(
                f"Could not read Pernix header: {source}"
            ) from error

        contents, pragma_count = re.subn(
            r"#pragma\s+GCC\s+unroll\s+\d+",
            "#pragma GCC unroll 64",
            contents,
        )
        if pragma_count == 0:
            raise ModelGenerationError(
                f"Model overlay found no GCC unroll pragmas in {source}"
            )
        if relative.endswith("avx2_compression.h"):
            signature = "__m256i mm256_pack_epi32_avx2(__m256i input) {"
            replacement = (
                "__attribute__((always_inline)) inline __m256i "
                "mm256_pack_epi32_avx2(__m256i input) {"
            )
            if contents.count(signature) != 1:
                raise ModelGenerationError(
                    "Could not locate the AVX2 pack helper in the model overlay"
                )
            contents = contents.replace(signature, replacement)

        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(contents, encoding="utf-8")
    return overlay_include


def _compile_model_artifacts(
    repository: Path,
    options: ModelOptions,
    isa_key: str,
    compiler: str,
) -> tuple[Path, Path, list[str]]:
    _, isa_flags, definition = _ISA[isa_key]
    source = repository / "src" / "model" / "incore_model.cpp"
    include = repository / "external" / "pernix" / "include"
    artifact_dir = options.build_dir / "instruction-model"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    overlay_include = _prepare_model_include_overlay(include, artifact_dir)
    assembly = artifact_dir / f"incore_model_{isa_key}.s"
    probe = artifact_dir / f"incore_probe_{isa_key}"
    common = [
        compiler,
        "-std=c++20",
        "-O3",
        "-DNDEBUG",
        *isa_flags,
        f"-D{definition}=1",
        "-w",
        str(source),
    ]
    compiler_version = _run([compiler, "--version"], cwd=repository).stdout.lower()
    model_unroll_flags = ["-funroll-loops"]
    if "gcc" in compiler_version or "g++" in compiler_version:
        # GCC 14 may retain small constant-trip loops despite an unroll pragma
        # when its default code-growth limits are reached. These limits apply
        # only to the model assembly and are deliberately generous enough for
        # the largest one-block specialization (32 iterations).
        model_unroll_flags.extend(
            [
                "--param=max-completely-peeled-insns=100000",
                "--param=max-completely-peel-times=64",
                "--param=max-unrolled-insns=100000",
                "--param=max-average-unrolled-insns=100000",
                "--param=max-unroll-times=64",
            ]
        )
    _run(
        [
            *common,
            "-DPERNIX_MODEL_MCA=1",
            f"-I{overlay_include}",
            f"-I{include}",
            *model_unroll_flags,
            "-S",
            "-o",
            str(assembly),
        ],
        cwd=repository,
    )
    _run([*common, f"-I{include}", "-o", str(probe)], cwd=repository)
    return (
        assembly,
        probe,
        [
            "-O3",
            *isa_flags,
            *model_unroll_flags,
            f"-I{overlay_include}",
        ],
    )


def _llvm_version(llvm_mca: str, repository: Path) -> str:
    output = _run([llvm_mca, "--version"], cwd=repository).stdout.splitlines()
    return output[1].strip() if len(output) > 1 else output[0].strip()


def _analyze_assembly(
    assembly: Path,
    *,
    llvm_mca: str,
    llvm_cpu: str | None,
    iterations: int,
    repository: Path,
) -> tuple[dict[tuple[str, str, int], dict[str, float]], dict[str, Any]]:
    command = [
        llvm_mca,
        "-mtriple=x86_64",
        f"-iterations={iterations}",
        "-json",
    ]
    if llvm_cpu:
        command.append(f"-mcpu={llvm_cpu}")
    command.append(str(assembly))
    completed = _run(command, cwd=repository, capture_stderr=True)
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise ModelGenerationError("LLVM-MCA returned invalid JSON") from error

    rows: dict[tuple[str, str, int], dict[str, float]] = {}
    for region in payload.get("CodeRegions", []):
        match = _REGION_RE.fullmatch(str(region.get("Name", "")))
        if match is None:
            raise ModelGenerationError(
                f"Unexpected LLVM-MCA region: {region.get('Name')}"
            )
        instructions = [str(value).lstrip() for value in region.get("Instructions", [])]
        control_flow = [
            instruction
            for instruction in instructions
            if instruction.startswith(("call", "j", "loop"))
        ]
        if control_flow:
            raise ModelGenerationError(
                f"Region {region['Name']} contains unmodeled control flow: "
                + ", ".join(control_flow[:3])
            )
        summary = region["SummaryView"]
        block_cycles = float(summary["BlockRThroughput"])
        if not math.isfinite(block_cycles) or block_cycles <= 0:
            raise ModelGenerationError(f"Invalid throughput for {region['Name']}")
        key = (match.group("operation"), match.group("isa"), int(match.group("width")))
        rows[key] = {
            "block_rthroughput_cycles": block_cycles,
            "model_instructions_per_block": float(summary["Instructions"]) / iterations,
            "model_uops_per_block": float(summary["TotaluOps"]) / iterations,
            "model_ipc": float(summary["IPC"]),
        }
    return rows, payload.get("TargetInfo", {})


def _probe_command(
    probe: Path, operation: str, width: int, iterations: int, control: bool
) -> list[str]:
    command = [str(probe), operation, str(width), str(iterations)]
    if control:
        command.append("--control")
    if hasattr(os, "sched_getaffinity"):
        affinity = sorted(os.sched_getaffinity(0))
        taskset = shutil.which("taskset")
        if affinity and taskset:
            command = [taskset, "-c", str(affinity[0]), *command]
    return command


def _selected_cpu() -> int:
    if hasattr(os, "sched_getaffinity"):
        affinity = sorted(os.sched_getaffinity(0))
        if affinity:
            return affinity[0]
    return 0


def _calibrate_iterations(
    probe: Path,
    operation: str,
    width: int,
    minimum_seconds: float,
    repository: Path,
    *,
    control: bool,
) -> int:
    iterations = 1_000
    for _ in range(8):
        start = time.monotonic()
        _run(
            _probe_command(probe, operation, width, iterations, control),
            cwd=repository,
        )
        elapsed = time.monotonic() - start
        if elapsed >= min(0.1, minimum_seconds):
            return max(
                iterations, math.ceil(iterations * minimum_seconds / elapsed * 1.1)
            )
        iterations *= 10
    raise ModelGenerationError(f"Could not calibrate probe for {operation} N={width}")


def _parse_likwid(output: str) -> dict[str, float]:
    """Extract the CLOCK group's counts and frequency from LIKWID CSV output."""

    values: dict[str, float] = {}
    aliases = {
        "RETIRED_INSTRUCTIONS": "instructions",
        "INSTR_RETIRED_ANY": "instructions",
        "CPU_CLOCKS_UNHALTED": "cycles",
        "CPU_CLK_UNHALTED_CORE": "cycles",
        "CLOCK [MHZ]": "frequency_mhz",
    }
    for fields in csv.reader(output.splitlines()):
        if len(fields) < 2:
            continue
        label = fields[0].strip().upper()
        key = aliases.get(label)
        if key is None:
            continue
        for field in reversed(fields[1:]):
            try:
                value = float(field.strip())
            except ValueError:
                continue
            if math.isfinite(value):
                values[key] = value
                break
    missing = {"instructions", "cycles", "frequency_mhz"} - values.keys()
    if missing:
        raise ModelGenerationError(
            "LIKWID CLOCK did not report required value(s): "
            + ", ".join(sorted(missing))
        )
    return {
        "instructions": values["instructions"],
        "cycles": values["cycles"],
        "frequency_hz": values["frequency_mhz"] * 1e6,
    }


def _measure_likwid_probe(
    probe: Path,
    operation: str,
    width: int,
    iterations: int,
    *,
    likwid_perfctr: str,
    repository: Path,
    control: bool,
) -> dict[str, float]:
    probe_command = [str(probe), operation, str(width), str(iterations)]
    if control:
        probe_command.append("--control")
    command = [
        likwid_perfctr,
        "-C",
        str(_selected_cpu()),
        "-g",
        "CLOCK",
        "-O",
        *probe_command,
    ]
    completed = _run(command, cwd=repository, capture_stderr=True)
    return _parse_likwid(completed.stdout)


def _read_cpu_frequency_hz(
    cpu: int,
    *,
    sysfs_root: Path = Path("/sys/devices/system/cpu"),
    cpuinfo_path: Path = Path("/proc/cpuinfo"),
) -> tuple[float, str] | None:
    """Read the current frequency for one CPU from Linux's live interfaces."""

    cpufreq = sysfs_root / f"cpu{cpu}" / "cpufreq"
    for filename in ("cpuinfo_cur_freq", "scaling_cur_freq"):
        path = cpufreq / filename
        try:
            frequency_hz = float(path.read_text(encoding="utf-8").strip()) * 1e3
        except (OSError, ValueError):
            continue
        if math.isfinite(frequency_hz) and frequency_hz > 0:
            return frequency_hz, f"linux_sysfs:{path}"

    try:
        blocks = cpuinfo_path.read_text(encoding="utf-8").split("\n\n")
    except OSError:
        return None
    for block in blocks:
        fields = {
            key.strip().lower(): value.strip()
            for line in block.splitlines()
            for key, separator, value in [line.partition(":")]
            if separator
        }
        if fields.get("processor") != str(cpu) or "cpu mhz" not in fields:
            continue
        try:
            frequency_hz = float(fields["cpu mhz"]) * 1e6
        except ValueError:
            return None
        if math.isfinite(frequency_hz) and frequency_hz > 0:
            return frequency_hz, f"proc_cpuinfo:cpu{cpu}"
    return None


def _measure_probe_frequency(
    probe: Path,
    operation: str,
    width: int,
    iterations: int,
    *,
    repository: Path,
    fixed_frequency_hz: float | None,
) -> tuple[float, str]:
    """Run a probe and sample its pinned CPU frequency without LIKWID counters."""

    command = _probe_command(probe, operation, width, iterations, False)
    cpu = _selected_cpu()
    if fixed_frequency_hz is not None:
        _run(command, cwd=repository)
        return fixed_frequency_hz, "explicit"

    samples: list[float] = []
    sources: list[str] = []

    def sample() -> None:
        current = _read_cpu_frequency_hz(cpu)
        if current is not None:
            frequency_hz, source = current
            samples.append(frequency_hz)
            sources.append(source)

    sample()
    try:
        process = subprocess.Popen(
            command,
            cwd=repository,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env={**os.environ, "LC_ALL": "C"},
        )
        while process.poll() is None:
            sample()
            time.sleep(0.01)
        stdout, stderr = process.communicate()
    except OSError as error:
        raise ModelGenerationError(f"Command failed: {' '.join(command)}") from error
    sample()
    if process.returncode != 0:
        raise ModelGenerationError(
            f"Command failed: {' '.join(command)}\n{(stderr or stdout).strip()}"
        )
    if not samples:
        raise ModelGenerationError(
            "LIKWID is unavailable and the live CPU frequency could not be read from "
            f"Linux sysfs or /proc/cpuinfo for CPU {cpu}; pass --cpu-frequency-hz "
            "with an independently measured frequency"
        )
    return statistics.median(samples), sources[0]


def _atomic_csv(path: Path, rows: list[dict[str, object]], columns: list[str]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def generate_incore_models(repository: Path, options: ModelOptions) -> Path:
    """Generate model, diagnostics, and metadata files in ``results_dir``."""

    repository = repository.resolve()
    results_dir = options.results_dir.resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    compiler = _executable(options.compiler, "C++ compiler")
    llvm_mca = _executable(options.llvm_mca, "LLVM-MCA")
    likwid_perfctr = shutil.which(options.likwid_perfctr or "likwid-perfctr")
    detected_likwid_perfctr = likwid_perfctr
    likwid_error: str | None = None
    llvm_version = _llvm_version(llvm_mca, repository)
    likwid_version = (
        _run([likwid_perfctr, "--version"], cwd=repository).stdout.splitlines()[0]
        if likwid_perfctr is not None
        else None
    )

    mca_rows: dict[tuple[str, str, int], dict[str, float]] = {}
    target_info: dict[str, Any] = {}
    probes: dict[str, Path] = {}
    flags: dict[str, list[str]] = {}
    for isa_key in _ISA:
        assembly, probe, flags[isa_key] = _compile_model_artifacts(
            repository, options, isa_key, compiler
        )
        analyzed, current_target = _analyze_assembly(
            assembly,
            llvm_mca=llvm_mca,
            llvm_cpu=options.llvm_cpu,
            iterations=options.mca_iterations,
            repository=repository,
        )
        mca_rows.update(analyzed)
        target_info = current_target
        probes[isa_key] = probe

    expected = {
        (operation, isa_key, width)
        for operation in ("compression", "decompression")
        for isa_key in _ISA
        for width in range(2, 25)
    }
    if set(mca_rows) != expected:
        missing = sorted(expected - set(mca_rows))
        raise ModelGenerationError(f"Incomplete LLVM-MCA coverage: {missing[:5]}")

    models: list[dict[str, object]] = []
    diagnostics: list[dict[str, object]] = []
    for operation, isa_key, width in sorted(expected):
        iterations = _calibrate_iterations(
            probes[isa_key],
            operation,
            width,
            options.probe_min_seconds,
            repository,
            control=False,
        )
        control_iterations: int | None = None
        instruction_delta: float | None = None
        cycle_delta: float | None = None
        frequency_source: str
        if likwid_perfctr is not None:
            try:
                control_iterations = _calibrate_iterations(
                    probes[isa_key],
                    operation,
                    width,
                    options.probe_min_seconds,
                    repository,
                    control=True,
                )
                measured = _measure_likwid_probe(
                    probes[isa_key],
                    operation,
                    width,
                    iterations,
                    likwid_perfctr=likwid_perfctr,
                    repository=repository,
                    control=False,
                )
                control = _measure_likwid_probe(
                    probes[isa_key],
                    operation,
                    width,
                    control_iterations,
                    likwid_perfctr=likwid_perfctr,
                    repository=repository,
                    control=True,
                )
                instruction_delta = (
                    measured["instructions"] / iterations
                    - control["instructions"] / control_iterations
                )
                cycle_delta = (
                    measured["cycles"] / iterations
                    - control["cycles"] / control_iterations
                )
                if instruction_delta <= 0 or measured["frequency_hz"] <= 0:
                    raise ModelGenerationError(
                        f"Invalid LIKWID calibration for {operation} {isa_key} N={width}"
                    )
                frequency_hz = measured["frequency_hz"]
                frequency_source = "likwid:CLOCK"
            except ModelGenerationError as error:
                likwid_error = str(error)
                logger.warning(
                    "LIKWID hardware counters are unavailable; using live Linux "
                    "CPU-frequency sampling without retired-instruction validation: %s",
                    error,
                )
                likwid_perfctr = None

        if likwid_perfctr is None:
            frequency_hz, frequency_source = _measure_probe_frequency(
                probes[isa_key],
                operation,
                width,
                iterations,
                repository=repository,
                fixed_frequency_hz=options.cpu_frequency_hz,
            )
        mca = mca_rows[(operation, isa_key, width)]
        blocks_per_second = frequency_hz / mca["block_rthroughput_cycles"]
        isa_name = _ISA[isa_key][0]
        models.append(
            {
                "operation": operation,
                "isa": isa_name,
                "bit_width": width,
                "blocks_per_second": blocks_per_second,
                "cpu_frequency_hz": frequency_hz,
            }
        )
        diagnostics.append(
            {
                "operation": operation,
                "isa": isa_name,
                "bit_width": width,
                "model_kind": "ideal_fully_unrolled",
                "llvm_cpu": target_info.get("CPUName", options.llvm_cpu or "native"),
                **mca,
                "retired_instructions_per_block": instruction_delta,
                # A control-subtracted cycle count is intentionally left empty when
                # the kernel overlaps the loop overhead well enough to make the
                # incremental cycle cost non-positive. Retired instructions remain
                # additive and are the validation metric for the static count.
                "measured_cycles_per_block": (
                    cycle_delta if cycle_delta is not None and cycle_delta > 0 else None
                ),
                "cpu_frequency_hz": frequency_hz,
                "blocks_per_second": blocks_per_second,
                "probe_iterations": iterations,
                "control_probe_iterations": control_iterations,
                "frequency_source": frequency_source,
                "compiler": compiler,
                "compiler_flags": " ".join(flags[isa_key]),
                "llvm_mca_version": llvm_version,
            }
        )

    model_path = results_dir / "incore_models.csv"
    _atomic_csv(model_path, models, MODEL_COLUMNS)
    _atomic_csv(
        results_dir / "incore_model_diagnostics.csv",
        diagnostics,
        DIAGNOSTIC_COLUMNS,
    )
    metadata = {
        "model_kind": "ideal_fully_unrolled",
        "payload_bytes": 64,
        "value_type": "f32",
        "signed_values": True,
        "widths": [2, 24],
        "mca_iterations": options.mca_iterations,
        "probe_min_seconds": options.probe_min_seconds,
        "target_info": target_info,
        "compiler": compiler,
        "llvm_mca": llvm_mca,
        "llvm_mca_version": llvm_version,
        "likwid_perfctr": detected_likwid_perfctr,
        "likwid_version": likwid_version,
        "likwid_error": likwid_error,
        "frequency_fallback": (
            "Median live frequency sampled from the probe's pinned logical CPU "
            "using cpuinfo_cur_freq, scaling_cur_freq, then /proc/cpuinfo."
        ),
        "semantics": (
            "Direct one-block kernel compiled through a build-local copy of the "
            "original Pernix headers with model-only complete unrolling and AVX2 "
            "helper inlining; the Pernix checkout is not modified. Loads and stores "
            "are included, while harness and function entry/exit are excluded."
        ),
        "model_header_overlay": str(
            options.build_dir.resolve() / "instruction-model" / "include-overlay"
        ),
        "hardware_diagnostics": (
            "When LIKWID's CLOCK group is usable, kernel and no-op control are calibrated "
            "independently and normalized per iteration. Without LIKWID, retired "
            "instruction and measured-cycle fields are empty; LLVM-MCA still "
            "provides the static instruction count and throughput model."
        ),
    }
    (results_dir / "incore_model_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return model_path


__all__ = [
    "DIAGNOSTIC_COLUMNS",
    "MODEL_COLUMNS",
    "ModelGenerationError",
    "ModelOptions",
    "generate_incore_models",
]
