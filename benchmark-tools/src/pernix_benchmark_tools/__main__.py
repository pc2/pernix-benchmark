from __future__ import annotations

import argparse
import logging
import math
import subprocess
from collections.abc import Sequence

from rich.logging import RichHandler

from .runner import PERNIX_VARIANTS, RunOptions, run_all, run_cp2k, run_pcie, run_pernix

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        RichHandler(
            show_time=True,
            show_level=True,
            show_path=False,
            rich_tracebacks=True,
        )
    ],
)

logger = logging.getLogger(__name__)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _positive_finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be a finite number greater than zero")
    return parsed


def _add_run_options(
    parser: argparse.ArgumentParser,
    *,
    suppress_defaults: bool = False,
) -> None:
    default = argparse.SUPPRESS if suppress_defaults else None
    parser.add_argument(
        "--compiler",
        default=argparse.SUPPRESS if suppress_defaults else "g++",
        help="C++ compiler passed to CMake (default: g++).",
    )
    parser.add_argument(
        "--build-type",
        default=argparse.SUPPRESS if suppress_defaults else "Release",
        help="CMake build type (default: Release).",
    )
    parser.add_argument(
        "--jobs",
        type=_positive_int,
        default=default,
        help="Number of parallel build jobs (default: detected CPU count).",
    )
    parser.add_argument(
        "--build-dir",
        type=str,
        default=default,
        help="CMake build directory (default: a compiler/build-type-specific directory).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=default,
        help="Result directory (default: benchmark-results/<timestamp>).",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        default=argparse.SUPPRESS if suppress_defaults else False,
        help="Delete the selected build directory before configuring.",
    )
    parser.add_argument(
        "--benchmark-min-time",
        type=_positive_finite_float,
        default=argparse.SUPPRESS if suppress_defaults else 0.25,
        metavar="SECONDS",
        help="Minimum measurement time per benchmark case (default: 0.25 seconds).",
    )


def _options(args: argparse.Namespace) -> RunOptions:
    return RunOptions(
        compiler=args.compiler,
        build_type=args.build_type,
        jobs=args.jobs,
        build_dir=args.build_dir,
        output_dir=args.output_dir,
        clean=args.clean,
        benchmark_min_time=args.benchmark_min_time,
    )


def _handle_pernix(args: argparse.Namespace) -> int:
    return run_pernix(args.variant, _options(args))


def _handle_cp2k(args: argparse.Namespace) -> int:
    return run_cp2k(_options(args))


def _handle_pcie(args: argparse.Namespace) -> int:
    return run_pcie(args.variant, _options(args))


def _handle_all(args: argparse.Namespace) -> int:
    return run_all(_options(args))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pernix-bench",
        description="Build and run Pernix compression benchmarks.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    run_parser = commands.add_parser(
        "run",
        help="Run all benchmarks, or select a benchmark family.",
    )
    _add_run_options(run_parser)
    run_parser.set_defaults(handler=_handle_all)
    run_commands = run_parser.add_subparsers(dest="benchmark")

    pernix_parser = run_commands.add_parser(
        "pernix",
        help="Run Pernix benchmarks supported by this host.",
    )
    pernix_parser.add_argument(
        "variant",
        nargs="?",
        choices=PERNIX_VARIANTS,
        help="Run only this Pernix implementation.",
    )
    _add_run_options(pernix_parser, suppress_defaults=True)
    pernix_parser.set_defaults(handler=_handle_pernix)

    cp2k_parser = run_commands.add_parser("cp2k", help="Run CP2K benchmarks.")
    _add_run_options(cp2k_parser, suppress_defaults=True)
    cp2k_parser.set_defaults(handler=_handle_cp2k)

    pcie_parser = run_commands.add_parser(
        "pcie", help="Run CUDA PCIe end-to-end Pernix benchmarks on x86.")
    pcie_parser.add_argument(
        "variant", nargs="?", choices=PERNIX_VARIANTS,
        help="Run only this x86 Pernix implementation.")
    _add_run_options(pcie_parser, suppress_defaults=True)
    pcie_parser.set_defaults(handler=_handle_pcie)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.handler(args))
    except (FileNotFoundError, PermissionError, RuntimeError) as error:
        logger.error("%s", error)
        return 1
    except subprocess.CalledProcessError as error:
        logger.error(
            "Command failed with exit status %s: %s", error.returncode, error.cmd
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
