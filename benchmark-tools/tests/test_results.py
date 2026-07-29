from __future__ import annotations

import json
from pathlib import Path

import pandas as pd  # type: ignore[import-untyped]
import pytest

from pernix_benchmark_tools.results import (
    load_benchmark_contexts,
    load_benchmark_results,
)


def _write_result(path: Path, names: list[str], *, host: str = "otus") -> None:
    benchmarks = [
        {
            "name": name,
            "bytes_per_second": float((index + 1) * 2**30),
            "items_per_second": float(index + 1),
        }
        for index, name in enumerate(names)
    ]
    path.write_text(
        json.dumps(
            {
                "context": {
                    "date": "2026-07-17T00:00:00+02:00",
                    "host_name": host,
                    "mhz_per_cpu": 2600,
                    "load_avg": [1.0, 2.0, 3.0],
                },
                "benchmarks": benchmarks,
            }
        ),
        encoding="utf-8",
    )


def test_loads_and_normalizes_pernix_and_cp2k_results(tmp_path: Path) -> None:
    _write_result(
        tmp_path / "benchmark_pernix_avx2_results.json",
        [
            "BM_compress_avx2f32_true_8/16",
            "BM_decompress_avx2f64_false_16/32",
        ],
    )
    _write_result(
        tmp_path / "benchmark_cp2k_results.json",
        ["BM_cp2k_decompression/width16/32/manual_time"],
    )

    results = load_benchmark_results(tmp_path)

    assert len(results) == 3
    core = results.loc[results["name"].str.contains("compress_avx2f32")].iloc[0]
    assert core["direction"] == "compression"
    assert core["implementation"] == "avx2"
    assert core["value_type"] == "f32"
    assert bool(core["core_throughput"])
    assert core["memory_mode"] == "core"
    assert core["bit_width"] == 8
    assert core["blocks"] == 16
    assert core["gib_per_second"] == pytest.approx(1.0)
    assert core["benchmark_schema"] == "pernix_v1"

    cp2k = results.loc[results["implementation"] == "cp2k"].iloc[0]
    assert cp2k["direction"] == "decompression"
    assert cp2k["value_type"] == "f64"
    assert not bool(cp2k["core_throughput"])
    assert cp2k["memory_mode"] == "full"
    assert cp2k["benchmark_schema"] == "cp2k_legacy"

    assert pd.api.types.is_integer_dtype(results["bit_width"])
    assert pd.api.types.is_integer_dtype(results["blocks"])


def test_loads_corrected_cp2k_names_with_comparable_schema(tmp_path: Path) -> None:
    _write_result(
        tmp_path / "benchmark_cp2k_results.json",
        [
            "BM_compress_cp2kf32_true_1/1",
            "BM_decompress_cp2kf64_false_24/4194304",
        ],
    )

    results = load_benchmark_results(tmp_path)

    assert set(results["implementation"]) == {"cp2k"}
    assert set(results["direction"]) == {"compression", "decompression"}
    assert set(results["value_type"]) == {"f32", "f64"}
    assert set(results["memory_mode"]) == {"core", "full"}
    assert set(results["benchmark_schema"]) == {"pernix_v1"}
    assert set(results["bit_width"]) == {1, 24}


def test_loads_and_normalizes_pcie_results(tmp_path: Path) -> None:
    _write_result(
        tmp_path / "benchmark_pcie_avx2_results.json",
        [
            "BM_pcie_h2d_avx2f32_1/4096",
            "BM_pcie_d2h_avx2f64_24/67108864",
        ],
    )

    results = load_benchmark_results(tmp_path)

    h2d = results.loc[results["transfer_direction"] == "h2d"].iloc[0]
    assert h2d["implementation"] == "avx2"
    assert h2d["direction"] == "compression"
    assert h2d["memory_mode"] == "pcie"
    assert h2d["payload_bytes"] == 4096
    assert h2d["blocks"] == 64
    assert h2d["benchmark_schema"] == "pcie_v1"

    d2h = results.loc[results["transfer_direction"] == "d2h"].iloc[0]
    assert d2h["direction"] == "decompression"
    assert d2h["payload_bytes"] == 67108864


def test_filters_result_files_by_benchmark_family(tmp_path: Path) -> None:
    _write_result(
        tmp_path / "benchmark_pernix_avx2_results.json",
        ["BM_compress_avx2f32_true_8/16"],
    )
    _write_result(
        tmp_path / "benchmark_pcie_avx2_results.json",
        ["BM_pcie_h2d_avx2f32_8/4096"],
    )

    kernel = load_benchmark_results(tmp_path, family="kernel")
    pcie = load_benchmark_results(tmp_path, family="pcie")
    kernel_contexts = load_benchmark_contexts(tmp_path, family="kernel")
    pcie_contexts = load_benchmark_contexts(tmp_path, family="pcie")

    assert set(kernel["benchmark_schema"]) == {"pernix_v1"}
    assert set(pcie["benchmark_schema"]) == {"pcie_v1"}
    assert kernel_contexts["source_file"].tolist() == [
        "benchmark_pernix_avx2_results.json"
    ]
    assert pcie_contexts["source_file"].tolist() == [
        "benchmark_pcie_avx2_results.json"
    ]


def test_loads_one_flattened_context_per_result_file(tmp_path: Path) -> None:
    _write_result(
        tmp_path / "benchmark_pernix_bmi2_results.json",
        ["BM_compress_bmi2f64_false_12/4"],
        host="cn0761",
    )

    contexts = load_benchmark_contexts(tmp_path)

    assert contexts.loc[0, "implementation"] == "bmi2"
    assert contexts.loc[0, "host_name"] == "cn0761"
    assert contexts.loc[0, "load_avg"] == [1.0, 2.0, 3.0]


def test_rejects_unrecognized_benchmark_names(tmp_path: Path) -> None:
    _write_result(
        tmp_path / "benchmark_pernix_avx2_results.json",
        ["BM_new_shape_avx2/8"],
    )

    with pytest.raises(ValueError, match="Unrecognized benchmark name"):
        load_benchmark_results(tmp_path)


def test_rejects_filename_and_identifier_implementation_mismatch(
    tmp_path: Path,
) -> None:
    _write_result(
        tmp_path / "benchmark_pernix_avx2_results.json",
        ["BM_compress_bmi2f32_true_8/1"],
    )

    with pytest.raises(ValueError, match="benchmark names contain"):
        load_benchmark_results(tmp_path)


def test_requires_a_result_directory_with_result_files(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="No benchmark result JSON"):
        load_benchmark_results(tmp_path)

    with pytest.raises(FileNotFoundError, match="does not exist"):
        load_benchmark_results(tmp_path / "missing")
