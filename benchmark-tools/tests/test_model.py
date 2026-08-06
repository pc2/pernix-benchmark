from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from pernix_benchmark_tools import model


def test_model_overlay_does_not_modify_pernix_headers(tmp_path: Path) -> None:
    include = tmp_path / "pernix-include"
    originals: dict[str, str] = {}
    for relative in model._MODEL_OVERLAY_HEADERS:
        path = include / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        contents = "#pragma GCC unroll 4\n"
        if relative.endswith("avx2_compression.h"):
            contents += "__m256i mm256_pack_epi32_avx2(__m256i input) {\n"
        path.write_text(contents)
        originals[relative] = contents

    overlay = model._prepare_model_include_overlay(include, tmp_path / "artifacts")

    for relative, original in originals.items():
        assert (include / relative).read_text() == original
        generated = (overlay / relative).read_text()
        assert "#pragma GCC unroll 64" in generated
    assert (
        "__attribute__((always_inline)) inline"
        in (overlay / "pernix/x86/avx2/avx2_compression.h").read_text()
    )


def test_parse_likwid_extracts_clock_group_values() -> None:
    values = model._parse_likwid(
        "Event,Counter,HWThread 0\n"
        "ACTUAL_CPU_CLOCK,FIXC1,2400000\n"
        "MAX_CPU_CLOCK,FIXC2,2000000\n"
        "RETIRED_INSTRUCTIONS,PMC0,1000000\n"
        "CPU_CLOCKS_UNHALTED,PMC1,2500000\n"
        "Metric,HWThread 0\n"
        "Runtime (RDTSC) [s],0.001\n"
        "Clock [MHz],5125.5\n"
    )

    assert values == {
        "instructions": 1_000_000,
        "cycles": 2_500_000,
        "frequency_hz": 5_125_500_000,
    }


def test_parse_likwid_rejects_missing_values() -> None:
    with pytest.raises(model.ModelGenerationError, match="frequency_mhz"):
        model._parse_likwid(
            "RETIRED_INSTRUCTIONS,PMC0,100\nCPU_CLOCKS_UNHALTED,PMC1,200\n"
        )


def test_read_cpu_frequency_prefers_live_sysfs(tmp_path: Path) -> None:
    cpufreq = tmp_path / "sys" / "cpu3" / "cpufreq"
    cpufreq.mkdir(parents=True)
    (cpufreq / "cpuinfo_cur_freq").write_text("4875000\n")

    value = model._read_cpu_frequency_hz(
        3,
        sysfs_root=tmp_path / "sys",
        cpuinfo_path=tmp_path / "missing",
    )

    assert value is not None
    assert value[0] == 4_875_000_000
    assert value[1].startswith("linux_sysfs:")


def test_analyze_assembly_parses_named_regions(tmp_path: Path) -> None:
    assembly = tmp_path / "model.s"
    assembly.touch()
    payload = {
        "TargetInfo": {"CPUName": "znver5"},
        "CodeRegions": [
            {
                "Name": "model_compression_avx2_7",
                "Instructions": ["vpaddd %ymm0, %ymm1, %ymm2"],
                "SummaryView": {
                    "BlockRThroughput": 2.5,
                    "Instructions": 400,
                    "TotaluOps": 600,
                    "IPC": 3.2,
                },
            }
        ],
    }
    completed = model.subprocess.CompletedProcess([], 0, json.dumps(payload), "")
    with patch.object(model, "_run", return_value=completed):
        rows, target = model._analyze_assembly(
            assembly,
            llvm_mca="llvm-mca",
            llvm_cpu="znver5",
            iterations=200,
            repository=tmp_path,
        )

    assert target["CPUName"] == "znver5"
    assert rows[("compression", "avx2", 7)] == {
        "block_rthroughput_cycles": 2.5,
        "model_instructions_per_block": 2.0,
        "model_uops_per_block": 3.0,
        "model_ipc": 3.2,
    }


@pytest.mark.parametrize("instruction", ["callq function", "jne .L2"])
def test_analyze_assembly_rejects_control_flow(
    tmp_path: Path, instruction: str
) -> None:
    payload = {
        "CodeRegions": [
            {
                "Name": "model_compression_avx2_2",
                "Instructions": [instruction],
                "SummaryView": {
                    "BlockRThroughput": 1,
                    "Instructions": 1,
                    "TotaluOps": 1,
                    "IPC": 1,
                },
            }
        ]
    }
    completed = model.subprocess.CompletedProcess([], 0, json.dumps(payload), "")
    with (
        patch.object(model, "_run", return_value=completed),
        pytest.raises(model.ModelGenerationError, match="control flow"),
    ):
        model._analyze_assembly(
            tmp_path / "model.s",
            llvm_mca="llvm-mca",
            llvm_cpu=None,
            iterations=1,
            repository=tmp_path,
        )


def test_atomic_csv_uses_requested_schema(tmp_path: Path) -> None:
    path = tmp_path / "models.csv"
    model._atomic_csv(
        path,
        [{column: index for index, column in enumerate(model.MODEL_COLUMNS)}],
        model.MODEL_COLUMNS,
    )

    assert path.read_text().splitlines()[0] == ",".join(model.MODEL_COLUMNS)
    assert not path.with_suffix(".csv.tmp").exists()
