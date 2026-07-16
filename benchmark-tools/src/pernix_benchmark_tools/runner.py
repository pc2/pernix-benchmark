from __future__ import annotations

import logging
import os
import platform
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .cmake import CMakeProject

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
    return repository / "build" / f"benchmarks_{compiler}_{build_type}"


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
        definitions={"CMAKE_CXX_COMPILER": options.compiler},
    )


def _collect_machine_state(repository: Path, output_dir: Path) -> None:
    output_file = output_dir / "machinestate.json"
    command = ["machinestate", "-e", "-o", str(output_file)]
    logger.info("Collecting machine state information")
    try:
        subprocess.run(command, cwd=repository, check=True)
    except (OSError, subprocess.CalledProcessError) as error:
        logger.warning("MachineState collection failed; continuing without it: %s", error)
        try:
            output_file.unlink(missing_ok=True)
        except OSError as cleanup_error:
            logger.warning(
                "Could not remove incomplete MachineState output %s: %s",
                output_file,
                cleanup_error,
            )


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

    for target, executable in built_targets:
        output_file = output_dir / f"benchmark_{target}_results.json"
        command = [
            str(executable),
            f"--benchmark_out={output_file}",
            "--benchmark_out_format=json",
        ]
        logger.info("Running benchmark target: %s", target)
        subprocess.run(command, cwd=project.build_dir, check=True)
        logger.info("Results saved to %s", output_file)

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


def run_all(
    options: RunOptions,
    *,
    host: HostCapabilities | None = None,
) -> int:
    selected = select_pernix_variants(None, host or detect_host())
    targets = tuple(f"pernix_{name}" for name in selected) + ("cp2k",)
    _run_targets(targets, options)
    return 0
