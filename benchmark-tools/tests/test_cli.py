from __future__ import annotations

from unittest.mock import patch

import pytest

from pernix_benchmark_tools.__main__ import build_parser, main


def test_run_pernix_command_is_registered() -> None:
    args = build_parser().parse_args(["run", "pernix", "avx2", "--clean"])

    assert args.command == "run"
    assert args.benchmark == "pernix"
    assert args.variant == "avx2"
    assert args.clean is True
    assert callable(args.handler)


def test_run_cp2k_command_is_registered_with_shared_options() -> None:
    args = build_parser().parse_args(
        ["run", "cp2k", "--compiler", "clang++", "--jobs", "4"]
    )

    assert args.benchmark == "cp2k"
    assert args.compiler == "clang++"
    assert args.jobs == 4
    assert callable(args.handler)


def test_bare_run_dispatches_all_benchmarks() -> None:
    with patch("pernix_benchmark_tools.__main__.run_all", return_value=0) as run:
        result = main(["run", "--jobs", "3"])

    assert result == 0
    options = run.call_args.args[0]
    assert options.jobs == 3
    assert options.benchmark_min_time == 0.25


@pytest.mark.parametrize("command", ["memory", "transform"])
def test_deferred_commands_are_not_registered(command: str) -> None:
    with pytest.raises(SystemExit) as error:
        build_parser().parse_args([command])

    assert error.value.code == 2


def test_unknown_pernix_variant_is_rejected() -> None:
    with pytest.raises(SystemExit) as error:
        build_parser().parse_args(["run", "pernix", "unknown"])

    assert error.value.code == 2


def test_fallback_pernix_variant_is_registered() -> None:
    args = build_parser().parse_args(["run", "pernix", "fallback"])

    assert args.variant == "fallback"


def test_run_options_work_before_selected_family() -> None:
    args = build_parser().parse_args(
        ["run", "--jobs", "6", "--clean", "pernix", "fallback"]
    )

    assert args.jobs == 6
    assert args.clean is True


def test_jobs_must_be_positive() -> None:
    with pytest.raises(SystemExit) as error:
        build_parser().parse_args(["run", "cp2k", "--jobs", "0"])

    assert error.value.code == 2


@pytest.mark.parametrize(
    "arguments",
    [
        ["run", "--benchmark-min-time", "0.5", "cp2k"],
        ["run", "cp2k", "--benchmark-min-time", "0.5"],
    ],
)
def test_benchmark_min_time_works_before_or_after_family(
    arguments: list[str],
) -> None:
    args = build_parser().parse_args(arguments)

    assert args.benchmark_min_time == 0.5


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "-inf"])
def test_benchmark_min_time_must_be_positive_and_finite(value: str) -> None:
    with pytest.raises(SystemExit) as error:
        build_parser().parse_args(["run", "--benchmark-min-time", value])

    assert error.value.code == 2


def test_main_dispatches_pernix_options() -> None:
    with patch("pernix_benchmark_tools.__main__.run_pernix", return_value=0) as run:
        result = main(
            [
                "run",
                "pernix",
                "bmi2",
                "--build-type",
                "Debug",
                "--output-dir",
                "results",
            ]
        )

    assert result == 0
    variant, options = run.call_args.args
    assert variant == "bmi2"
    assert options.build_type == "Debug"
    assert options.output_dir == "results"


def test_main_returns_one_for_runtime_error() -> None:
    with patch(
        "pernix_benchmark_tools.__main__.run_cp2k",
        side_effect=RuntimeError("failed"),
    ):
        assert main(["run", "cp2k"]) == 1
