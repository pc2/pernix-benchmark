from __future__ import annotations

import subprocess
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pernix_benchmark_tools.runner import (
    HostCapabilities,
    RunOptions,
    normalize_architecture,
    parse_cpuinfo_features,
    resolve_build_dir,
    resolve_output_dir,
    run_all,
    run_pcie,
    run_pernix,
    select_pernix_variants,
)


def test_parse_cpuinfo_returns_features_shared_by_all_processors() -> None:
    contents = """
processor : 0
flags : avx2 bmi2 avx512f avx512vbmi
processor : 1
flags : avx2 bmi2
"""

    assert parse_cpuinfo_features(contents) == frozenset({"avx2", "bmi2"})


@pytest.mark.parametrize(
    ("machine", "expected"),
    [("x86_64", "x86"), ("AMD64", "x86"), ("aarch64", "arm64")],
)
def test_normalize_architecture(machine: str, expected: str) -> None:
    assert normalize_architecture(machine) == expected


def test_bare_pernix_selects_supported_x86_variants_in_order() -> None:
    host = HostCapabilities(
        "x86",
        frozenset({"avx2", "bmi2", "avx512f", "avx512vbmi"}),
    )

    assert select_pernix_variants(None, host) == (
        "fallback",
        "avx2",
        "bmi2",
        "avx512vbmi",
    )


def test_bare_pernix_selects_supported_arm_variants() -> None:
    host = HostCapabilities("arm64", frozenset({"asimd", "sve2"}))

    assert select_pernix_variants(None, host) == ("fallback", "neon", "sve2")


def test_fallback_is_supported_on_unknown_architecture() -> None:
    host = HostCapabilities("riscv64", frozenset())

    assert select_pernix_variants(None, host) == ("fallback",)
    assert select_pernix_variants("fallback", host) == ("fallback",)


def test_unsupported_explicit_feature_fails_before_building() -> None:
    host = HostCapabilities("x86", frozenset({"avx2", "bmi2"}))

    with pytest.raises(RuntimeError, match="avx512vbmi"):
        select_pernix_variants("avx512vbmi", host)


def test_incompatible_explicit_architecture_fails() -> None:
    host = HostCapabilities("arm64", frozenset({"asimd"}))

    with pytest.raises(RuntimeError, match="incompatible"):
        select_pernix_variants("avx2", host)


def test_default_paths_are_deterministic(tmp_path: Path) -> None:
    options = RunOptions(compiler="/usr/bin/clang++", build_type="RelWithDebInfo")

    assert resolve_build_dir(tmp_path, options) == (
        tmp_path / "build" / "benchmarks_clang__relwithdebinfo"
    )
    assert resolve_output_dir(
        tmp_path,
        options,
        now=datetime(2026, 7, 16, 12, 34, 56),
    ) == (tmp_path / "benchmark-results" / "20260716_123456")


def test_run_pernix_maps_variants_to_cmake_targets() -> None:
    host = HostCapabilities("x86", frozenset({"avx2", "bmi2"}))
    options = RunOptions()

    with patch("pernix_benchmark_tools.runner._run_targets") as run_targets:
        result = run_pernix(None, options, host=host)

    assert result == 0
    run_targets.assert_called_once_with(
        ("pernix_fallback", "pernix_avx2", "pernix_bmi2"),
        options,
    )


def test_run_all_maps_supported_pernix_targets_and_cp2k() -> None:
    host = HostCapabilities("x86", frozenset({"avx2"}))
    options = RunOptions()

    with patch("pernix_benchmark_tools.runner._run_targets") as run_targets:
        result = run_all(options, host=host)

    assert result == 0
    run_targets.assert_called_once_with(
        ("pernix_fallback", "pernix_avx2", "cp2k"),
        options,
    )


def test_run_pcie_maps_x86_variants_to_cuda_targets() -> None:
    host = HostCapabilities("x86", frozenset({"avx2", "bmi2"}))
    options = RunOptions()

    with patch("pernix_benchmark_tools.runner._run_targets") as run_targets:
        result = run_pcie(None, options, host=host)

    assert result == 0
    targets, cuda_options = run_targets.call_args.args
    assert targets == ("pcie_fallback", "pcie_avx2", "pcie_bmi2")
    assert cuda_options.enable_cuda


def test_run_pcie_rejects_non_x86_hosts() -> None:
    with pytest.raises(RuntimeError, match="x86"):
        run_pcie(None, RunOptions(), host=HostCapabilities("arm64", frozenset({"asimd"})))


def test_run_targets_configures_once_and_runs_each_executable(tmp_path: Path) -> None:
    from pernix_benchmark_tools import runner

    repository = tmp_path / "repository"
    build_dir = repository / "build-dir"
    output_dir = repository / "output"
    executable_dir = build_dir / "src"
    executable_dir.mkdir(parents=True)
    for target in ("bench_pernix_avx2", "bench_pernix_bmi2"):
        (executable_dir / target).touch()

    project = MagicMock()
    project.build_dir = build_dir
    options = RunOptions(output_dir=str(output_dir), clean=True)
    events: list[str] = []
    project.build.side_effect = lambda **_: events.append("build")

    def record_command(command: list[str], **_: object) -> None:
        events.append(command[0])

    with (
        patch.object(runner, "find_repository_root", return_value=repository),
        patch.object(runner, "_create_project", return_value=project),
        patch.object(
            runner.subprocess,
            "run",
            side_effect=record_command,
        ) as subprocess_run,
    ):
        result = runner._run_targets(
            ("pernix_avx2", "pernix_bmi2"),
            options,
        )

    assert result == output_dir
    project.configure.assert_called_once_with(fresh=True)
    assert project.build.call_count == 2
    assert subprocess_run.call_count == 3
    machine_state_command = subprocess_run.call_args_list[0].args[0]
    assert machine_state_command == [
        "machinestate",
        "-e",
        "-o",
        str(output_dir / "machinestate.json"),
    ]
    first_benchmark_command = subprocess_run.call_args_list[1].args[0]
    assert first_benchmark_command[0].endswith("bench_pernix_avx2")
    assert first_benchmark_command[1].endswith(
        "benchmark_pernix_avx2_results.json"
    )
    assert "--benchmark_min_time=0.25s" in first_benchmark_command
    assert (
        "--benchmark_context=benchmark_min_time_seconds=0.25"
        in first_benchmark_command
    )
    assert events == [
        "build",
        "build",
        "machinestate",
        str(executable_dir / "bench_pernix_avx2"),
        str(executable_dir / "bench_pernix_bmi2"),
    ]


def test_machine_state_failure_warns_cleans_partial_file_and_continues(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    from pernix_benchmark_tools import runner

    repository = tmp_path / "repository"
    build_dir = repository / "build-dir"
    output_dir = repository / "output"
    executable = build_dir / "src" / "bench_cp2k"
    executable.parent.mkdir(parents=True)
    executable.touch()

    project = MagicMock(build_dir=build_dir)
    options = RunOptions(output_dir=str(output_dir))

    def fail_machine_state(command: list[str], **_: object) -> None:
        if command[0] == "machinestate":
            (output_dir / "machinestate.json").write_text("partial")
            raise subprocess.CalledProcessError(1, command)

    with (
        patch.object(runner, "find_repository_root", return_value=repository),
        patch.object(runner, "_create_project", return_value=project),
        patch.object(runner.subprocess, "run", side_effect=fail_machine_state) as run,
        caplog.at_level("WARNING"),
    ):
        assert runner._run_targets(("cp2k",), options) == output_dir

    assert run.call_count == 2
    assert not (output_dir / "machinestate.json").exists()
    assert "MachineState collection failed" in caplog.text


def test_run_targets_forwards_custom_benchmark_min_time(tmp_path: Path) -> None:
    from pernix_benchmark_tools import runner

    repository = tmp_path / "repository"
    build_dir = repository / "build-dir"
    executable = build_dir / "src" / "bench_cp2k"
    executable.parent.mkdir(parents=True)
    executable.touch()

    project = MagicMock(build_dir=build_dir)
    options = RunOptions(
        output_dir=str(repository / "output"),
        benchmark_min_time=1.5,
    )

    with (
        patch.object(runner, "find_repository_root", return_value=repository),
        patch.object(runner, "_create_project", return_value=project),
        patch.object(runner, "_collect_machine_state"),
        patch.object(runner.subprocess, "run") as run,
    ):
        runner._run_targets(("cp2k",), options)

    command = run.call_args.args[0]
    assert "--benchmark_min_time=1.5s" in command
    assert "--benchmark_context=benchmark_min_time_seconds=1.5" in command
