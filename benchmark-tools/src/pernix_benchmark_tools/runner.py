from __future__ import annotations

import json
import logging
import os
import platform
import re
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .cmake import CMakeProject
from .model import ModelOptions, generate_incore_models

logger = logging.getLogger(__name__)

PERNIX_VARIANTS = ("fallback", "avx2", "bmi2", "avx512vbmi", "neon", "sve2")
X86_VARIANTS = ("fallback", "avx2", "bmi2", "avx512vbmi")
ARM64_VARIANTS = ("fallback", "neon", "sve2")

REQUIRED_FEATURES: dict[str, frozenset[str]] = {
    "fallback": frozenset(),
    "avx2": frozenset({"avx2"}),
    "bmi2": frozenset({"bmi2"}),
    "avx512vbmi": frozenset({"avx512f", "avx512vbmi"}),
    "neon": frozenset({"asimd"}),
    "sve2": frozenset({"sve2"}),
}


@dataclass(frozen=True)
class HostCapabilities:
    architecture: str
    features: frozenset[str]


@dataclass(frozen=True)
class RunOptions:
    compiler: str = "g++"
    build_type: str = "Release"
    jobs: int | None = None
    build_dir: str | None = None
    output_dir: str | None = None
    clean: bool = False
    benchmark_min_time: float = 0.25
    benchmark_repetitions: int = 5
    benchmark_min_warmup_time: float = 0.05
    enable_cuda: bool = False


def normalize_architecture(machine: str) -> str:
    normalized = machine.lower()
    if normalized in {"x86_64", "amd64", "i386", "i486", "i586", "i686"}:
        return "x86"
    if normalized in {"aarch64", "arm64"}:
        return "arm64"
    return normalized


def parse_cpuinfo_features(contents: str) -> frozenset[str]:
    """Return CPU features present on every processor described by /proc/cpuinfo."""
    feature_sets: list[set[str]] = []
    for line in contents.splitlines():
        key, separator, value = line.partition(":")
        if separator and key.strip().lower() in {"flags", "features"}:
            feature_sets.append(set(value.lower().split()))
    if not feature_sets:
        return frozenset()
    return frozenset(set.intersection(*feature_sets))


def detect_host(cpuinfo_path: Path = Path("/proc/cpuinfo")) -> HostCapabilities:
    features: frozenset[str] = frozenset()
    try:
        features = parse_cpuinfo_features(cpuinfo_path.read_text(encoding="utf-8"))
    except OSError:
        logger.warning("Could not read CPU features from %s", cpuinfo_path)
    return HostCapabilities(normalize_architecture(platform.machine()), features)


def supported_pernix_variants(host: HostCapabilities) -> tuple[str, ...]:
    candidates = (
        X86_VARIANTS
        if host.architecture == "x86"
        else ARM64_VARIANTS
        if host.architecture == "arm64"
        else ("fallback",)
    )
    return tuple(
        variant
        for variant in candidates
        if REQUIRED_FEATURES[variant].issubset(host.features)
    )


def select_pernix_variants(
    requested: str | None,
    host: HostCapabilities,
) -> tuple[str, ...]:
    supported = supported_pernix_variants(host)
    if requested is None:
        if not supported:
            raise RuntimeError(
                f"No supported Pernix benchmark variants detected for {host.architecture!r}"
            )
        return supported

    architecture_variants = (
        X86_VARIANTS
        if host.architecture == "x86"
        else ARM64_VARIANTS
        if host.architecture == "arm64"
        else ("fallback",)
    )
    if requested not in architecture_variants:
        raise RuntimeError(
            f"Pernix variant {requested!r} is incompatible with host architecture "
            f"{host.architecture!r}"
        )
    missing = REQUIRED_FEATURES[requested] - host.features
    if missing:
        raise RuntimeError(
            f"Pernix variant {requested!r} requires missing CPU feature(s): "
            f"{', '.join(sorted(missing))}"
        )
    return (requested,)


def find_repository_root() -> Path:
    starts = (Path.cwd(), Path(__file__).resolve())
    for start in starts:
        for candidate in (start, *start.parents):
            if (candidate / "CMakeLists.txt").is_file() and (
                candidate / "src" / "CMakeLists.txt"
            ).is_file():
                return candidate
    raise FileNotFoundError(
        "Could not locate the pernix-benchmark repository root; run the command from the checkout"
    )


def _safe_build_component(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def resolve_build_dir(repository: Path, options: RunOptions) -> Path:
    if options.build_dir:
        return Path(options.build_dir).expanduser().resolve()
    compiler = _safe_build_component(Path(options.compiler).name)
    build_type = _safe_build_component(options.build_type.lower())
    suffix = "_cuda" if options.enable_cuda else ""
    return repository / "build" / f"benchmarks_{compiler}_{build_type}{suffix}"


def resolve_output_dir(
    repository: Path,
    options: RunOptions,
    *,
    now: datetime | None = None,
) -> Path:
    if options.output_dir:
        return Path(options.output_dir).expanduser().resolve()
    timestamp = (now or datetime.now()).strftime("%Y%m%d_%H%M%S")
    return repository / "benchmark-results" / timestamp


def _create_project(repository: Path, options: RunOptions) -> CMakeProject:
    return CMakeProject(
        source_dir=repository,
        build_dir=resolve_build_dir(repository, options),
        build_type=options.build_type,
        jobs=options.jobs or os.cpu_count() or 1,
        definitions={
            "CMAKE_CXX_COMPILER": options.compiler,
            "PERNIX_BENCHMARK_ENABLE_CUDA": "ON" if options.enable_cuda else "OFF",
        },
    )


def _collect_machine_state(repository: Path, output_dir: Path) -> None:
    output_name = os.environ.get("PERNIX_MACHINE_STATE_FILENAME", "machinestate.json")
    if Path(output_name).name != output_name or not output_name.endswith(".json"):
        raise RuntimeError(
            "PERNIX_MACHINE_STATE_FILENAME must be a JSON filename without directories"
        )
    output_file = output_dir / output_name
    command = ["machinestate", "-e", "-o", str(output_file)]
    logger.info("Collecting machine state information")
    try:
        subprocess.run(command, cwd=repository, check=True)
    except (OSError, subprocess.CalledProcessError) as error:
        logger.warning(
            "MachineState collection failed; continuing without it: %s", error
        )
        try:
            output_file.unlink(missing_ok=True)
        except OSError as cleanup_error:
            logger.warning(
                "Could not remove incomplete MachineState output %s: %s",
                output_file,
                cleanup_error,
            )


def _collect_pcie_state(output_dir: Path, compiler: str) -> None:
    """Record the accelerator and negotiated PCIe link used by a CUDA run."""

    query = [
        "nvidia-smi",
        "--query-gpu=name,pcie.link.gen.current,pcie.link.width.current",
        "--format=csv,noheader,nounits",
    ]
    try:
        gpu_lines = subprocess.run(
            query, check=True, capture_output=True, text=True
        ).stdout.splitlines()
        compiler_line = subprocess.run(
            [compiler, "--version"], check=True, capture_output=True, text=True
        ).stdout.splitlines()[0]
    except (OSError, subprocess.CalledProcessError, IndexError) as error:
        logger.warning("PCIe metadata collection failed: %s", error)
        return

    gpus: list[dict[str, object]] = []
    for index, line in enumerate(gpu_lines):
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 3:
            logger.warning("Unexpected nvidia-smi PCIe metadata row: %s", line)
            continue
        name, generation, width = fields
        gpus.append(
            {
                "index": index,
                "model": name,
                "pcie_generation": int(generation)
                if generation.isdigit()
                else generation,
                "negotiated_link_width": int(width) if width.isdigit() else width,
            }
        )
    (output_dir / "pcie-metadata.json").write_text(
        json.dumps({"gpus": gpus, "compiler": compiler_line}, indent=2),
        encoding="utf-8",
    )


def _target_display_name(target: str) -> str:
    for prefix in ("pernix_", "pcie_"):
        if target.startswith(prefix):
            target = target.removeprefix(prefix)
            break
    return target.upper()


def _format_duration(seconds: float) -> str:
    total_seconds = max(0, int(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _report_slurm_progress(
    completed: int,
    total: int,
    *,
    current: str | None = None,
    elapsed_seconds: float = 0,
    remaining_seconds: float | None = None,
) -> None:
    job_id = os.environ.get("SLURM_JOB_ID")
    if not job_id:
        return

    status = (
        f"{completed}/{total} (Current: {_target_display_name(current)})"
        if current is not None
        else f"{completed}/{total} (Complete)"
    )
    elapsed = _format_duration(elapsed_seconds)
    if current is None:
        comment = f"{status} [{elapsed}]"
    else:
        remaining = (
            _format_duration(remaining_seconds)
            if remaining_seconds is not None
            else "?"
        )
        comment = f"{status} [{elapsed}<{remaining}]"
    try:
        subprocess.run(
            [
                "scontrol",
                "update",
                f"JobId={job_id}",
                f"Comment={comment}",
            ],
            check=False,
        )
    except OSError as error:
        logger.warning("Could not update SLURM job progress: %s", error)


def _run_targets(targets: tuple[str, ...], options: RunOptions) -> Path:
    repository = find_repository_root()
    output_dir = resolve_output_dir(repository, options)
    output_dir.mkdir(parents=True, exist_ok=True)

    project = _create_project(repository, options)
    project.configure(fresh=options.clean)

    built_targets: list[tuple[str, Path]] = []
    for target in targets:
        cmake_target = f"bench_{target}"
        project.build(target=cmake_target, configure=False)
        executable = project.build_dir / "src" / cmake_target
        if not executable.is_file():
            raise FileNotFoundError(f"Benchmark executable not found: {executable}")
        built_targets.append((target, executable))

    _collect_machine_state(repository, output_dir)
    if options.enable_cuda:
        _collect_pcie_state(output_dir, options.compiler)

    total_targets = len(built_targets)
    progress_started = time.monotonic()
    for index, (target, executable) in enumerate(built_targets):
        elapsed_seconds = time.monotonic() - progress_started
        remaining_seconds = (
            elapsed_seconds / index * (total_targets - index) if index > 0 else None
        )
        _report_slurm_progress(
            index,
            total_targets,
            current=target,
            elapsed_seconds=elapsed_seconds,
            remaining_seconds=remaining_seconds,
        )
        output_file = output_dir / f"benchmark_{target}_results.json"
        command = [
            str(executable),
            f"--benchmark_out={output_file}",
            "--benchmark_out_format=json",
            f"--benchmark_min_time={options.benchmark_min_time:g}s",
            f"--benchmark_repetitions={options.benchmark_repetitions}",
            "--benchmark_report_aggregates_only=false",
            f"--benchmark_min_warmup_time={options.benchmark_min_warmup_time:g}s",
            "--benchmark_context="
            f"benchmark_min_time_seconds={options.benchmark_min_time:g}",
            "--benchmark_context="
            f"benchmark_repetitions={options.benchmark_repetitions}",
            "--benchmark_context="
            f"benchmark_min_warmup_time_seconds={options.benchmark_min_warmup_time:g}",
            "--benchmark_context=compiler_optimization=-O3",
            f"--benchmark_context=compiler={options.compiler}",
            f"--benchmark_context=build_type={options.build_type}",
            f"--benchmark_context=benchmark_target={target}",
        ]
        logger.info("Running benchmark target: %s", target)
        subprocess.run(command, cwd=project.build_dir, check=True)
        logger.info("Results saved to %s", output_file)

    _report_slurm_progress(
        total_targets,
        total_targets,
        elapsed_seconds=time.monotonic() - progress_started,
    )
    logger.info("Benchmark results directory: %s", output_dir)
    return output_dir


def run_pernix(
    variant: str | None,
    options: RunOptions,
    *,
    host: HostCapabilities | None = None,
) -> int:
    selected = select_pernix_variants(variant, host or detect_host())
    targets = tuple(f"pernix_{name}" for name in selected)
    _run_targets(targets, options)
    return 0


def run_cp2k(options: RunOptions) -> int:
    _run_targets(("cp2k",), options)
    return 0


def run_pcie(
    variant: str | None,
    options: RunOptions,
    *,
    host: HostCapabilities | None = None,
) -> int:
    detected = host or detect_host()
    if detected.architecture != "x86":
        raise RuntimeError(
            "CUDA PCIe benchmarks currently support x86 Pernix implementations only"
        )
    selected = select_pernix_variants(variant, detected)
    cuda_options = RunOptions(
        compiler=options.compiler,
        build_type=options.build_type,
        jobs=options.jobs,
        build_dir=options.build_dir,
        output_dir=options.output_dir,
        clean=options.clean,
        benchmark_min_time=options.benchmark_min_time,
        benchmark_repetitions=options.benchmark_repetitions,
        benchmark_min_warmup_time=options.benchmark_min_warmup_time,
        enable_cuda=True,
    )
    _run_targets(tuple(f"pcie_{name}" for name in selected), cuda_options)
    return 0


def run_all(
    options: RunOptions,
    *,
    host: HostCapabilities | None = None,
) -> int:
    selected = select_pernix_variants(None, host or detect_host())
    targets = tuple(f"pernix_{name}" for name in selected) + ("cp2k",)
    output_dir = _run_targets(targets, options)
    if {"avx2", "avx512vbmi"}.issubset(selected):
        repository = find_repository_root()
        logger.info("Generating LLVM-MCA in-core instruction models")
        generate_incore_models(
            repository,
            ModelOptions(
                results_dir=output_dir,
                build_dir=resolve_build_dir(repository, options),
                compiler=options.compiler,
                llvm_mca=os.environ.get("LLVM_MCA", "llvm-mca"),
                likwid_perfctr=os.environ.get("LIKWID_PERFCTR"),
                llvm_cpu=os.environ.get("PERNIX_LLVM_CPU"),
            ),
        )
    return 0
