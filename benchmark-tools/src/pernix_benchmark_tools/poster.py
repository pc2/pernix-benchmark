"""Prepare and render the three poster benchmark figures."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd  # type: ignore[import-untyped]
from matplotlib.ticker import FuncFormatter


COMPRESSION = "#0A75C4"
DECOMPRESSION = "#0025AA"
BASELINE = "#555B66"
MODEL = "#9AA0A6"
CACHE_FILL = "#E8EBF0"
GRID = "#D9DEE7"
TEXT = "#1F2937"

INCORE_COLUMNS = [
    "operation",
    "isa",
    "bit_width",
    "kind",
    "blocks_per_second",
    "values_per_second",
    "median_seconds",
    "repetitions",
    "cpu_frequency_hz",
]
HOST_MEMORY_COLUMNS = [
    "operation",
    "isa",
    "bit_width",
    "blocks",
    "values_per_block",
    "scale_bytes",
    "working_set_bytes",
    "original_bytes",
    "elapsed_seconds",
    "logical_bytes_per_second",
    "repetition",
]
PCIE_COLUMNS = [
    "direction",
    "bit_width",
    "isa",
    "original_bytes",
    "compressed_bytes",
    "blocks",
    "codec_seconds",
    "transfer_seconds",
    "total_seconds",
    "uncompressed_seconds",
    "pinned_memory",
    "async_copy",
    "streams",
    "overlap",
    "repetition",
]

_TIME_SCALES = {"ns": 1e-9, "us": 1e-6, "ms": 1e-3, "s": 1.0}
_ISA_NAMES = {
    "avx2": "AVX2",
    "bmi2": "AVX2+BMI2",
    "avx512vbmi": "AVX-512-VBMI",
    "cp2k": "CP2K",
}


class PosterDataError(ValueError):
    """Raised when source measurements cannot support a requested poster plot."""


def configure_poster_style() -> None:
    """Apply the UPB poster style and PDF font embedding settings."""

    plt.rcParams.update(
        {
            "font.family": ["Karla", "DejaVu Sans"],
            "font.size": 10.5,
            "axes.labelcolor": TEXT,
            "axes.edgecolor": TEXT,
            "axes.titlecolor": TEXT,
            "xtick.color": TEXT,
            "ytick.color": TEXT,
            "text.color": TEXT,
            "figure.facecolor": "#FFFFFF",
            "axes.facecolor": "#FFFFFF",
            "axes.grid": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "legend.frameon": False,
        }
    )


def _require_columns(frame: pd.DataFrame, columns: list[str], source: str) -> None:
    missing = [column for column in columns if column not in frame]
    if missing:
        raise PosterDataError(f"{source} is missing required columns: {', '.join(missing)}")


def _elapsed_seconds(frame: pd.DataFrame) -> pd.Series:
    scales = frame["time_unit"].map(_TIME_SCALES)
    if scales.isna().any():
        units = sorted(frame.loc[scales.isna(), "time_unit"].astype(str).unique())
        raise PosterDataError(f"Unsupported benchmark time unit(s): {', '.join(units)}")
    return pd.to_numeric(frame["real_time"], errors="raise") * scales


def _column_or_default(
    frame: pd.DataFrame, column: str, default: int | float | bool
) -> pd.Series:
    if column in frame:
        return frame[column]
    return pd.Series(default, index=frame.index)


def _iteration_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if "run_type" not in frame:
        return frame
    return frame[frame["run_type"].eq("iteration")]


def _binary_bytes_label(value: float, _position: float) -> str:
    if value >= 2**30:
        return f"{value / 2**30:g} GiB"
    if value >= 2**20:
        return f"{value / 2**20:g} MiB"
    return f"{value / 2**10:g} KiB"


def _normalize_operation(value: str) -> str:
    normalized = value.lower()
    aliases = {"compress": "compression", "decompress": "decompression"}
    return aliases.get(normalized, normalized)


def prepare_incore_data(
    kernel_results: pd.DataFrame,
    model_results: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Create the required tidy in-core table, using medians across repetitions."""

    _require_columns(
        kernel_results,
        [
            "direction",
            "implementation",
            "value_type",
            "memory_mode",
            "bit_width",
            "blocks",
            "items_per_second",
            "name",
            "real_time",
            "time_unit",
            "mhz_per_cpu",
        ],
        "kernel results",
    )
    kernel_results = _iteration_rows(kernel_results)
    measured = kernel_results[
        kernel_results["direction"].isin(["compression", "decompression"])
        & kernel_results["implementation"].isin(["avx2", "bmi2", "avx512vbmi"])
        & kernel_results["value_type"].eq("f32")
        & kernel_results["memory_mode"].eq("core")
        & kernel_results["bit_width"].between(2, 24)
        & kernel_results["blocks"].eq(1)
    ].copy()
    if measured.empty:
        raise PosterDataError("No one-block FP32 in-core measurements were found")

    measured["elapsed_seconds"] = _elapsed_seconds(measured)
    measured["cpu_frequency_hz"] = (
        pd.to_numeric(measured["mhz_per_cpu"], errors="coerce") * 1e6
    )
    if measured["cpu_frequency_hz"].isna().any():
        raise PosterDataError("In-core measurements do not record CPU frequency")
    keys = ["direction", "implementation", "bit_width"]
    tidy = (
        measured.groupby(keys, as_index=False, observed=True)
        .agg(
            blocks_per_second=("items_per_second", "median"),
            median_seconds=("elapsed_seconds", "median"),
            repetitions=("name", "size"),
            cpu_frequency_hz=("cpu_frequency_hz", "median"),
        )
        .rename(columns={"direction": "operation"})
    )
    tidy["isa"] = tidy.pop("implementation").map(_ISA_NAMES)
    tidy["kind"] = "measured"
    tidy["values_per_second"] = tidy["blocks_per_second"] * (
        512 // tidy["bit_width"]
    )

    if model_results is not None and not model_results.empty:
        required_model = [
            "operation",
            "isa",
            "bit_width",
            "blocks_per_second",
            "cpu_frequency_hz",
        ]
        _require_columns(model_results, required_model, "in-core model CSV")
        models = model_results.copy()
        models["operation"] = models["operation"].map(_normalize_operation)
        models["kind"] = "model"
        models["values_per_second"] = models["blocks_per_second"] * (
            512 // models["bit_width"]
        )
        models["median_seconds"] = pd.NA
        models["repetitions"] = 0
        tidy = pd.concat([tidy, models[INCORE_COLUMNS]], ignore_index=True)

    return tidy[INCORE_COLUMNS].sort_values(
        ["operation", "kind", "isa", "bit_width"], ignore_index=True
    )


def validate_incore_coverage(data: pd.DataFrame) -> None:
    """Require complete measured widths and both requested instruction models."""

    expected_widths = set(range(2, 25))
    measured_isas = ("AVX2", "AVX-512-VBMI")
    model_isas = ("AVX2", "AVX-512-VBMI")
    failures: list[str] = []
    for operation in ("compression", "decompression"):
        for isa in measured_isas:
            widths = set(
                data.loc[
                    data["operation"].eq(operation)
                    & data["isa"].eq(isa)
                    & data["kind"].eq("measured"),
                    "bit_width",
                ]
            )
            if widths != expected_widths:
                failures.append(f"{operation} {isa} measured")
        for isa in model_isas:
            widths = set(
                data.loc[
                    data["operation"].eq(operation)
                    & data["isa"].eq(isa)
                    & data["kind"].eq("model"),
                    "bit_width",
                ]
            )
            if widths != expected_widths:
                failures.append(f"{operation} {isa} model")
    if failures:
        raise PosterDataError(
            "Incomplete width coverage (expected N=2..24): " + ", ".join(failures)
        )


def plot_incore_throughput(data: pd.DataFrame) -> plt.Figure:
    """Render the two-panel in-core throughput poster figure."""

    validate_incore_coverage(data)
    configure_poster_style()
    figure, axes = plt.subplots(
        1, 2, figsize=(12.6, 4.8), sharey=True, constrained_layout=True
    )
    measured_styles = {
        "AVX2": ("o", 2.4),
        "AVX2+BMI2": ("^", 2.4),
        "AVX-512-VBMI": ("s", 2.4),
    }
    model_styles = {
        "AVX2": ("o", ":"),
        "AVX-512-VBMI": ("s", "--"),
    }
    operation_colors = {"compression": COMPRESSION, "decompression": DECOMPRESSION}

    for axis, operation in zip(axes, ("compression", "decompression"), strict=True):
        subset = data[data["operation"].eq(operation)]
        for width in (8, 16, 24):
            axis.axvspan(width - 0.16, width + 0.16, color=CACHE_FILL, zorder=0)
        for isa, (marker, linewidth) in measured_styles.items():
            series = subset[
                subset["kind"].eq("measured") & subset["isa"].eq(isa)
            ]
            if series.empty:
                continue
            axis.plot(
                series["bit_width"],
                series["values_per_second"] / 1e9,
                color=operation_colors[operation],
                marker=marker,
                markersize=5.5,
                linewidth=linewidth,
                label=isa,
                zorder=3,
            )
        for isa, (marker, linestyle) in model_styles.items():
            series = subset[subset["kind"].eq("model") & subset["isa"].eq(isa)]
            axis.plot(
                series["bit_width"],
                series["values_per_second"] / 1e9,
                color=MODEL,
                marker=marker,
                markerfacecolor="white",
                markersize=4.5,
                linewidth=1.6,
                linestyle=linestyle,
                label=f"{isa} model",
                zorder=2,
            )
        axis.set_title(operation.title())
        axis.set_xlabel("Bit width N")
        axis.set_xticks(range(2, 25, 2))
        axis.grid(axis="y", color=GRID, linewidth=0.8)
        axis.spines[["top", "right"]].set_visible(False)
        axis.text(
            16,
            0.99,
            "byte-aligned: N = 8, 16, 24",
            transform=axis.get_xaxis_transform(),
            ha="center",
            va="top",
            color=BASELINE,
            fontsize=8.5,
        )
    axes[0].set_ylabel("Throughput [Gvalues/s]")
    axes[1].legend(loc="upper right", fontsize=8.5, ncols=1)
    return figure


def prepare_host_memory_data(kernel_results: pd.DataFrame) -> pd.DataFrame:
    """Create the required tidy FP32 host-memory table."""

    _require_columns(
        kernel_results,
        [
            "direction",
            "implementation",
            "value_type",
            "memory_mode",
            "bit_width",
            "blocks",
            "real_time",
            "time_unit",
        ],
        "kernel results",
    )
    kernel_results = _iteration_rows(kernel_results)
    data = kernel_results[
        kernel_results["direction"].isin(["compression", "decompression"])
        & kernel_results["implementation"].isin(["avx2", "avx512vbmi", "cp2k"])
        & kernel_results["value_type"].eq("f32")
        & kernel_results["memory_mode"].eq("full")
        & kernel_results["bit_width"].isin([7, 13, 21])
    ].copy()
    if data.empty:
        raise PosterDataError("No FP32 full-memory measurements for N=7,13,21 were found")
    data["values_per_block"] = 512 // data["bit_width"]
    # The implementation passes one scalar by value for the entire call. It is
    # outside the 512-bit packed block, so there is no per-block scale storage.
    data["scale_bytes"] = 0
    data["working_set_bytes"] = data["blocks"] * (
        4 * data["values_per_block"] + 64 + data["scale_bytes"]
    )
    data["original_bytes"] = 4 * data["blocks"] * data["values_per_block"]
    data["elapsed_seconds"] = _elapsed_seconds(data)
    data["logical_bytes_per_second"] = (
        data["original_bytes"] / data["elapsed_seconds"]
    )
    data["operation"] = data["direction"]
    data["isa"] = data["implementation"].map(_ISA_NAMES)
    data["repetition"] = pd.to_numeric(
        _column_or_default(data, "repetition_index", 0), errors="coerce"
    ).fillna(0).astype(int)
    return data[HOST_MEMORY_COLUMNS].sort_values(
        ["operation", "isa", "bit_width", "working_set_bytes", "repetition"],
        ignore_index=True,
    )


def cache_metadata(contexts: pd.DataFrame) -> dict[str, Any]:
    """Extract measured data-cache capacities and sharing scope."""

    if "caches" not in contexts:
        raise PosterDataError("Benchmark context contains no cache metadata")
    cache_lists = [value for value in contexts["caches"] if isinstance(value, list)]
    if not cache_lists:
        raise PosterDataError("Benchmark context contains no usable cache metadata")
    caches = cache_lists[0]
    result: dict[str, Any] = {}
    for cache in caches:
        level = cache.get("level")
        cache_type = cache.get("type")
        if level == 1 and cache_type != "Data":
            continue
        if level not in (1, 2, 3):
            continue
        key = f"l{level}"
        result[key] = {
            "bytes": int(cache["size"]),
            "num_sharing": int(cache.get("num_sharing", 1)),
            "scope": (
                "per core"
                if int(cache.get("num_sharing", 1)) == 1
                else f"shared by {int(cache['num_sharing'])} logical CPUs"
            ),
        }
    missing = [key for key in ("l1", "l2", "l3") if key not in result]
    if missing:
        raise PosterDataError("Missing cache capacities: " + ", ".join(missing))
    return result


def plot_host_memory_throughput(
    data: pd.DataFrame,
    caches: Mapping[str, Mapping[str, Any]],
    *,
    physical_memory_bytes_per_second: float | None = None,
) -> plt.Figure:
    """Render the two-panel host-memory throughput poster figure."""

    configure_poster_style()
    principal = data[
        data["isa"].eq("AVX-512-VBMI")
        | (data["isa"].eq("CP2K") & data["bit_width"].eq(13))
    ].copy()
    if principal.empty:
        raise PosterDataError("Host-memory poster data has no AVX-512-VBMI rows")
    summary = (
        principal.groupby(
            ["operation", "isa", "bit_width", "working_set_bytes"],
            as_index=False,
            observed=True,
        )["logical_bytes_per_second"]
        .median()
        .sort_values("working_set_bytes")
    )
    figure, axes = plt.subplots(
        1, 2, figsize=(12.6, 5.2), sharey=False, constrained_layout=True
    )
    markers = {7: "o", 13: "s", 21: "^"}
    colors = {"compression": COMPRESSION, "decompression": DECOMPRESSION}
    cache_edges = [
        ("L1", float(caches["l1"]["bytes"])),
        ("L2", float(caches["l2"]["bytes"])),
        ("L3", float(caches["l3"]["bytes"])),
    ]

    for axis, operation in zip(axes, ("compression", "decompression"), strict=True):
        operation_data = summary[summary["operation"].eq(operation)]
        if operation_data.empty:
            raise PosterDataError(f"No host-memory rows for {operation}")
        xmin = float(operation_data["working_set_bytes"].min())
        xmax = float(operation_data["working_set_bytes"].max())
        boundaries = [xmin] + [edge for _, edge in cache_edges] + [xmax]
        region_names = ["L1", "L2", "L3", "DRAM"]
        for index, region in enumerate(region_names):
            left = max(xmin, boundaries[index])
            right = min(xmax, boundaries[index + 1])
            if right <= left:
                continue
            axis.axvspan(left, right, color=CACHE_FILL, alpha=0.52 if index % 2 == 0 else 0.25)
            axis.text(
                (left * right) ** 0.5,
                0.98,
                region,
                transform=axis.get_xaxis_transform(),
                ha="center",
                va="top",
                color=BASELINE,
                fontsize=8.5,
            )
        for width in (7, 13, 21):
            series = operation_data[
                operation_data["isa"].eq("AVX-512-VBMI")
                & operation_data["bit_width"].eq(width)
            ]
            axis.plot(
                series["working_set_bytes"],
                series["logical_bytes_per_second"] / 1e9,
                color=colors[operation],
                marker=markers[width],
                markersize=5.5,
                linewidth=2.8 if width == 13 else 2.4,
                label=f"N={width}",
                zorder=3,
            )
        cp2k = operation_data[operation_data["isa"].eq("CP2K")]
        if not cp2k.empty:
            axis.plot(
                cp2k["working_set_bytes"],
                cp2k["logical_bytes_per_second"] / 1e9,
                color=BASELINE,
                marker="s",
                markerfacecolor="white",
                markersize=5.0,
                linewidth=1.6,
                label="CP2K, N=13",
                zorder=2,
            )
        if physical_memory_bytes_per_second is not None:
            for width in (7, 13, 21):
                values = 512 // width
                bound = physical_memory_bytes_per_second * (4 * values) / (
                    4 * values + 64
                )
                axis.axhline(
                    bound / 1e9,
                    color=BASELINE,
                    linestyle="--",
                    linewidth=1.5,
                    alpha=0.45,
                    label="Memory bound" if width == 13 else None,
                )
        axis.set_xscale("log", base=2)
        axis.xaxis.set_major_formatter(FuncFormatter(_binary_bytes_label))
        axis.set_title(operation.title())
        axis.set_xlabel("Total working set [KiB / MiB]")
        axis.grid(axis="y", color=GRID, linewidth=0.8)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Original FP32 throughput [GB/s]")
    axes[1].legend(loc="best", fontsize=8.5)
    return figure


def prepare_pcie_data(pcie_results: pd.DataFrame) -> pd.DataFrame:
    """Create poster PCIe rows and retain only the approximately 1 GiB FP32 cases."""

    required = [
        "transfer_direction",
        "implementation",
        "value_type",
        "bit_width",
        "blocks",
        "payload_bytes",
        "real_time",
        "time_unit",
        "iterations",
        "dma_seconds",
        "uncompressed_seconds",
    ]
    _require_columns(pcie_results, required, "PCIe results")
    pcie_results = _iteration_rows(pcie_results)
    data = pcie_results[
        pcie_results["transfer_direction"].isin(["h2d", "d2h"])
        & pcie_results["value_type"].eq("f32")
        & pcie_results["bit_width"].between(2, 24)
    ].copy()
    if data.empty:
        raise PosterDataError("No FP32 PCIe rows for N=2..24 were found")
    data["values_per_block"] = 512 // data["bit_width"]
    data["expected_blocks"] = (2**30 // (4 * data["values_per_block"])).astype(int)
    data = data[data["blocks"].eq(data["expected_blocks"])].copy()
    if data.empty:
        raise PosterDataError(
            "No PCIe rows use B=floor(2^30/(4*k_N)); rerun the poster PCIe cases"
        )
    data["direction"] = data["transfer_direction"].str.upper()
    data["isa"] = data["implementation"].map(_ISA_NAMES).fillna(
        data["implementation"]
    )
    data["original_bytes"] = 4 * data["blocks"] * data["values_per_block"]
    data["compressed_bytes"] = pd.to_numeric(data["payload_bytes"], errors="raise")
    data["total_seconds"] = _elapsed_seconds(data)
    data["transfer_seconds"] = (
        pd.to_numeric(data["dma_seconds"], errors="raise")
        / pd.to_numeric(data["iterations"], errors="raise")
    )
    data["overlap"] = _column_or_default(data, "overlap", 0).fillna(0).astype(bool)
    if data["overlap"].any():
        if "codec_seconds" not in data:
            raise PosterDataError(
                "Overlapped PCIe rows must record codec_seconds separately"
            )
        data["codec_seconds"] = pd.to_numeric(
            data["codec_seconds"], errors="raise"
        )
    else:
        data["codec_seconds"] = (
            data["total_seconds"] - data["transfer_seconds"]
        ).clip(lower=0)
    data["uncompressed_seconds"] = pd.to_numeric(
        data["uncompressed_seconds"], errors="coerce"
    )
    data["pinned_memory"] = _column_or_default(
        data, "pinned_memory", 1
    ).fillna(1).astype(bool)
    data["async_copy"] = _column_or_default(data, "async_copy", 1).fillna(1).astype(bool)
    data["streams"] = pd.to_numeric(
        _column_or_default(data, "streams", 1), errors="coerce"
    ).fillna(1).astype(int)
    data["repetition"] = pd.to_numeric(
        _column_or_default(data, "repetition_index", 0), errors="coerce"
    ).fillna(0).astype(int)
    return data[PCIE_COLUMNS].sort_values(
        ["direction", "isa", "bit_width", "repetition"], ignore_index=True
    )


def validate_pcie_coverage(data: pd.DataFrame, *, isa: str = "AVX-512-VBMI") -> None:
    expected = set(range(2, 25))
    failures: list[str] = []
    for direction in ("H2D", "D2H"):
        subset = data[data["direction"].eq(direction) & data["isa"].eq(isa)]
        if set(subset["bit_width"]) != expected:
            failures.append(f"{direction} {isa}")
        if subset["uncompressed_seconds"].isna().any():
            failures.append(f"{direction} uncompressed baseline")
    if failures:
        raise PosterDataError(
            "Incomplete PCIe poster coverage: " + ", ".join(failures)
        )


def plot_pcie_end_to_end(
    data: pd.DataFrame, *, isa: str = "AVX-512-VBMI"
) -> plt.Figure:
    """Render the two-panel end-to-end PCIe runtime poster figure."""

    validate_pcie_coverage(data, isa=isa)
    configure_poster_style()
    summary = (
        data[data["isa"].eq(isa)]
        .groupby(["direction", "bit_width"], as_index=False, observed=True)
        .agg(
            total_seconds=("total_seconds", "median"),
            codec_seconds=("codec_seconds", "median"),
            transfer_seconds=("transfer_seconds", "median"),
            uncompressed_seconds=("uncompressed_seconds", "median"),
        )
    )
    figure, axes = plt.subplots(
        1, 2, figsize=(12.6, 4.8), sharey=True, constrained_layout=True
    )
    colors = {"H2D": COMPRESSION, "D2H": DECOMPRESSION}
    for axis, direction in zip(axes, ("H2D", "D2H"), strict=True):
        subset = summary[summary["direction"].eq(direction)].sort_values("bit_width")
        widths = subset["bit_width"]
        total_ms = subset["total_seconds"] * 1e3
        codec_ms = subset["codec_seconds"] * 1e3
        transfer_ms = subset["transfer_seconds"] * 1e3
        baseline_ms = float(subset["uncompressed_seconds"].median() * 1e3)
        axis.plot(
            widths,
            total_ms,
            color=colors[direction],
            linewidth=2.4,
            marker="o",
            markersize=5.5,
            label="Compressed pipeline",
        )
        axis.plot(
            widths,
            codec_ms,
            color=BASELINE,
            linewidth=1.6,
            linestyle="--",
            label="CPU codec",
        )
        axis.plot(
            widths,
            transfer_ms,
            color=BASELINE,
            linewidth=1.6,
            linestyle=":",
            label="PCIe transfer",
        )
        axis.axhline(
            baseline_ms,
            color=BASELINE,
            linewidth=1.7,
            label="Uncompressed transfer",
        )
        axis.fill_between(
            widths,
            total_ms,
            baseline_ms,
            where=total_ms < baseline_ms,
            color=colors[direction],
            alpha=0.12,
            interpolate=True,
        )
        axis.set_title(direction)
        axis.set_xlabel("Bit width N")
        axis.set_xticks(range(2, 25, 2))
        axis.grid(axis="y", color=GRID, linewidth=0.8)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("End-to-end runtime [ms]")
    axes[1].legend(loc="best", fontsize=8.5)
    return figure


def export_tidy_csv(frame: pd.DataFrame, destination: str | Path) -> Path:
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


__all__ = [
    "BASELINE",
    "CACHE_FILL",
    "COMPRESSION",
    "DECOMPRESSION",
    "GRID",
    "HOST_MEMORY_COLUMNS",
    "INCORE_COLUMNS",
    "MODEL",
    "PCIE_COLUMNS",
    "PosterDataError",
    "TEXT",
    "cache_metadata",
    "configure_poster_style",
    "export_tidy_csv",
    "plot_host_memory_throughput",
    "plot_incore_throughput",
    "plot_pcie_end_to_end",
    "prepare_host_memory_data",
    "prepare_incore_data",
    "prepare_pcie_data",
    "validate_incore_coverage",
    "validate_pcie_coverage",
]
