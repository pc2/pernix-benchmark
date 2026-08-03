from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd  # type: ignore[import-untyped]

from pernix_benchmark_tools.poster import (
    HOST_MEMORY_COLUMNS,
    INCORE_COLUMNS,
    PCIE_COLUMNS,
    cache_metadata,
    plot_host_memory_throughput,
    plot_incore_throughput,
    plot_pcie_end_to_end,
    prepare_host_memory_data,
    prepare_incore_data,
    prepare_pcie_data,
)


def _kernel_results() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for direction in ("compression", "decompression"):
        for implementation in ("avx2", "bmi2", "avx512vbmi", "cp2k"):
            for width in range(2, 25):
                for memory_mode, blocks in (("core", 1), ("full", 32)):
                    for repetition in range(2):
                        rows.append(
                            {
                                "name": f"case-{direction}-{implementation}-{width}-{memory_mode}",
                                "run_type": "iteration",
                                "direction": direction,
                                "implementation": implementation,
                                "value_type": "f32",
                                "memory_mode": memory_mode,
                                "bit_width": width,
                                "blocks": blocks,
                                "items_per_second": 1000.0 + width + repetition,
                                "real_time": 10.0 + repetition,
                                "time_unit": "ns",
                                "mhz_per_cpu": 2600,
                                "repetition_index": repetition,
                            }
                        )
    return pd.DataFrame(rows)


def _models() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "operation": operation,
                "isa": isa,
                "bit_width": width,
                "blocks_per_second": 1200.0 + width,
                "cpu_frequency_hz": 2.6e9,
            }
            for operation in ("compression", "decompression")
            for isa in ("AVX2", "AVX-512-VBMI")
            for width in range(2, 25)
        ]
    )


def test_prepares_and_plots_incore_data() -> None:
    data = prepare_incore_data(_kernel_results(), _models())

    assert list(data.columns) == INCORE_COLUMNS
    row = data[
        data["operation"].eq("compression")
        & data["isa"].eq("AVX2")
        & data["kind"].eq("measured")
        & data["bit_width"].eq(7)
    ].iloc[0]
    assert row["repetitions"] == 2
    assert row["cpu_frequency_hz"] == 2.6e9
    assert row["values_per_second"] == row["blocks_per_second"] * (512 // 7)

    figure = plot_incore_throughput(data)
    assert tuple(figure.get_size_inches()) == (12.6, 4.8)
    plt.close(figure)


def test_prepares_host_memory_data_and_cache_metadata() -> None:
    data = prepare_host_memory_data(_kernel_results())
    assert list(data.columns) == HOST_MEMORY_COLUMNS
    assert set(data["bit_width"]) == {7, 13, 21}
    assert set(data["scale_bytes"]) == {0}
    sample = data.iloc[0]
    assert sample["working_set_bytes"] == sample["blocks"] * (
        4 * sample["values_per_block"] + 64
    )

    contexts = pd.DataFrame(
        {
            "caches": [
                [
                    {"type": "Data", "level": 1, "size": 48 * 2**10, "num_sharing": 1},
                    {"type": "Unified", "level": 2, "size": 2**20, "num_sharing": 1},
                    {"type": "Unified", "level": 3, "size": 32 * 2**20, "num_sharing": 8},
                ]
            ]
        }
    )
    caches = cache_metadata(contexts)
    assert caches["l3"]["scope"] == "shared by 8 logical CPUs"
    figure = plot_host_memory_throughput(data, caches)
    assert tuple(figure.get_size_inches()) == (12.6, 5.2)
    plt.close(figure)


def test_prepares_and_plots_pcie_data() -> None:
    rows: list[dict[str, object]] = []
    for direction in ("h2d", "d2h"):
        for width in range(2, 25):
            values = 512 // width
            blocks = 2**30 // (4 * values)
            for repetition in range(2):
                rows.append(
                    {
                        "run_type": "iteration",
                        "transfer_direction": direction,
                        "implementation": "avx512vbmi",
                        "value_type": "f32",
                        "bit_width": width,
                        "blocks": blocks,
                        "payload_bytes": blocks * 64,
                        "real_time": 20.0 + width,
                        "time_unit": "ms",
                        "iterations": 2,
                        "dma_seconds": 0.01,
                        "uncompressed_seconds": 0.03,
                        "pinned_memory": 1,
                        "async_copy": 1,
                        "streams": 1,
                        "overlap": 0,
                        "repetition_index": repetition,
                    }
                )
    data = prepare_pcie_data(pd.DataFrame(rows))

    assert list(data.columns) == PCIE_COLUMNS
    assert set(data["bit_width"]) == set(range(2, 25))
    assert (data["original_bytes"] <= 2**30).all()
    figure = plot_pcie_end_to_end(data)
    assert tuple(figure.get_size_inches()) == (12.6, 4.8)
    plt.close(figure)
