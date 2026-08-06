from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd  # type: ignore[import-untyped]

from pernix_benchmark_tools.poster import (
    HOST_MEMORY_COLUMNS,
    INCORE_COLUMNS,
    PCIE_COLUMNS,
    cache_metadata,
    copy_bandwidth_metadata,
    plot_host_memory_throughput,
    plot_incore_throughput,
    plot_pcie_end_to_end,
    prepare_copy_bandwidth_data,
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
    caches["l3"]["plot_scope"] = "per CCD"
    working_sets = [4 * 2**10, 128 * 2**10, 2 * 2**20, 64 * 2**20]
    data["working_set_bytes"] = [
        working_sets[index % len(working_sets)] for index in range(len(data))
    ]
    memory_bandwidth = prepare_copy_bandwidth_data(
        pd.DataFrame(
            {
                "data_size": [4 * 2**10, 2**20, 32 * 2**20, 64 * 2**20],
                "copy_avx512": [200e9, 150e9, 100e9, 60e9],
            }
        )
    )
    figure = plot_host_memory_throughput(
        data, caches, memory_bandwidth=memory_bandwidth
    )
    assert tuple(figure.get_size_inches()) == (12.6, 5.2)
    region_labels = {text.get_text() for text in figure.axes[0].texts}
    assert {"L1\n48 KiB", "L2\n1 MiB", "L3 (per CCD)\n32 MiB", "DRAM"} <= region_labels
    vertical_boundaries = {
        float(line.get_xdata()[0])
        for line in figure.axes[0].lines
        if len(line.get_xdata()) == 2 and line.get_xdata()[0] == line.get_xdata()[1]
    }
    assert vertical_boundaries == {48 * 2**10, 2**20, 32 * 2**20}
    boundary_lines = [
        line
        for line in figure.axes[0].lines
        if len(line.get_xdata()) == 2 and line.get_xdata()[0] == line.get_xdata()[1]
    ]
    assert all(line.get_linestyle() == "--" for line in boundary_lines)
    assert all(line.get_linewidth() == 1.5 for line in boundary_lines)
    plotted_maximum = (
        data.loc[data["operation"].eq("compression"), "logical_bytes_per_second"].max()
        / 1e9
    )
    assert plotted_maximum / figure.axes[0].get_ylim()[1] <= 0.84 + 1e-12
    legend_labels = figure.axes[1].get_legend_handles_labels()[1]
    assert "LIKWID copy bound (N=12)" in legend_labels
    assert len(figure.legends) == 1
    assert not figure.axes[1].get_legend()
    plt.close(figure)


def test_derives_physical_memory_bandwidth_from_dram_copy_results() -> None:
    measurements = pd.DataFrame(
        {
            "data_size": [32 * 2**20, 512 * 2**20, 768 * 2**20, 2**30],
            "copy": [80e9, 40e9, 39e9, 38e9],
            "copy_avx512": [90e9, 52e9, 50e9, 48e9],
        }
    )

    metadata = copy_bandwidth_metadata(measurements)
    series = prepare_copy_bandwidth_data(measurements)

    assert metadata["kernel"] == "copy_avx512"
    assert metadata["samples"] == 3
    assert metadata["useful_bytes_per_second"] == 50e9
    assert metadata["physical_bytes_per_second"] == 100e9
    assert list(series.columns) == [
        "working_set_bytes",
        "useful_bytes_per_second",
        "physical_bytes_per_second",
        "kernel",
    ]
    assert series["physical_bytes_per_second"].iloc[0] == 180e9


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
