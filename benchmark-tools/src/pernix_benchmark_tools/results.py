"""Load and normalize Google Benchmark result files."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Literal

import pandas as pd  # type: ignore[import-untyped]


PERNIX_BENCHMARK_RE = re.compile(
    r"^BM_(?P<direction>compress|decompress)_"
    r"(?P<implementation>[A-Za-z0-9]+?)(?P<value_type>f32|f64)_"
    r"(?P<core_throughput>true|false)_"
    r"(?P<bit_width>\d+)/(?P<blocks>\d+)$"
)
CP2K_BENCHMARK_RE = re.compile(
    r"^BM_cp2k_(?P<direction>compression|decompression)/"
    r"width(?P<bit_width>\d+)/(?P<blocks>\d+)/manual_time$"
)
PERNIX_RESULT_RE = re.compile(
    r"^benchmark_pernix_(?P<implementation>[A-Za-z0-9]+)_results\.json$"
)
PCIE_BENCHMARK_RE = re.compile(
    r"^BM_pcie_(?P<transfer_direction>h2d|d2h)_"
    r"(?P<implementation>[A-Za-z0-9]+?)(?P<value_type>f32|f64)_"
    r"(?P<bit_width>\d+)/(?P<payload_bytes>\d+)$"
)
PCIE_RESULT_RE = re.compile(
    r"^benchmark_pcie_(?P<implementation>[A-Za-z0-9]+)_results\.json$"
)
ResultFamily = Literal["kernel", "pcie"]


def _read_result(path: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read benchmark result {path}: {error}") from error

    benchmarks = payload.get("benchmarks")
    context = payload.get("context")
    if not isinstance(benchmarks, list) or not isinstance(context, dict):
        raise ValueError(
            f"Benchmark result {path} must contain a benchmark list and context object"
        )
    return pd.DataFrame(benchmarks), context


def _require_matches(
    names: pd.Series, extracted: pd.DataFrame, source_file: str
) -> None:
    unmatched = names[extracted.isna().any(axis=1)].astype(str).tolist()
    if unmatched:
        preview = ", ".join(repr(name) for name in unmatched[:5])
        remainder = "" if len(unmatched) <= 5 else f" (+{len(unmatched) - 5} more)"
        raise ValueError(
            f"Unrecognized benchmark name(s) in {source_file}: {preview}{remainder}"
        )


def _names_for_parsing(frame: pd.DataFrame) -> pd.Series:
    """Remove Google Benchmark's aggregate suffix without discarding the row."""

    names = frame["name"].astype(str).copy()
    if "aggregate_name" not in frame:
        return names
    for index, aggregate_name in frame["aggregate_name"].dropna().items():
        suffix = f"_{aggregate_name}"
        if names.at[index].endswith(suffix):
            names.at[index] = names.at[index][: -len(suffix)]
    return names


def _parse_pernix(frame: pd.DataFrame, path: Path, implementation: str) -> pd.DataFrame:
    extracted = _names_for_parsing(frame).str.extract(PERNIX_BENCHMARK_RE)
    _require_matches(frame["name"], extracted, path.name)

    parsed_implementations = set(extracted["implementation"])
    if parsed_implementations != {implementation}:
        raise ValueError(
            f"Implementation encoded by {path.name!r} is {implementation!r}, but "
            f"benchmark names contain {sorted(parsed_implementations)!r}"
        )

    extracted["direction"] = extracted["direction"].map(
        {"compress": "compression", "decompress": "decompression"}
    )
    extracted["core_throughput"] = extracted["core_throughput"].eq("true")
    extracted["memory_mode"] = extracted["core_throughput"].map(
        {True: "core", False: "full"}
    )
    extracted["benchmark_schema"] = "pernix_v1"
    return extracted


def _parse_cp2k(frame: pd.DataFrame, path: Path) -> pd.DataFrame:
    parseable_names = _names_for_parsing(frame)
    modern = parseable_names.str.extract(PERNIX_BENCHMARK_RE)
    if not modern.isna().any(axis=1).any():
        return _parse_pernix(frame, path, "cp2k")

    extracted = parseable_names.str.extract(CP2K_BENCHMARK_RE)
    _require_matches(frame["name"], extracted, path.name)
    extracted["implementation"] = "cp2k"
    extracted["value_type"] = "f64"
    extracted["core_throughput"] = False
    extracted["memory_mode"] = "full"
    extracted["benchmark_schema"] = "cp2k_legacy"
    return extracted


def _parse_pcie(frame: pd.DataFrame, path: Path, implementation: str) -> pd.DataFrame:
    extracted = _names_for_parsing(frame).str.extract(PCIE_BENCHMARK_RE)
    _require_matches(frame["name"], extracted, path.name)
    parsed_implementations = set(extracted["implementation"])
    if parsed_implementations != {implementation}:
        raise ValueError(
            f"Implementation encoded by {path.name!r} is {implementation!r}, but "
            f"benchmark names contain {sorted(parsed_implementations)!r}"
        )
    extracted["direction"] = extracted["transfer_direction"].map(
        {"h2d": "compression", "d2h": "decompression"}
    )
    extracted["core_throughput"] = False
    extracted["memory_mode"] = "pcie"
    extracted["blocks"] = pd.to_numeric(extracted["payload_bytes"], errors="raise") // 64
    extracted["benchmark_schema"] = "pcie_v1"
    return extracted


def _normalize_file(path: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    frame, context = _read_result(path)
    if "name" not in frame:
        raise ValueError(f"Benchmark result {path} contains no name column")

    pcie_match = PCIE_RESULT_RE.fullmatch(path.name)
    pernix_match = PERNIX_RESULT_RE.fullmatch(path.name)
    if pcie_match:
        extracted = _parse_pcie(frame, path, pcie_match.group("implementation"))
    elif pernix_match:
        extracted = _parse_pernix(frame, path, pernix_match.group("implementation"))
    elif path.name == "benchmark_cp2k_results.json":
        extracted = _parse_cp2k(frame, path)
    else:
        raise ValueError(f"Unsupported benchmark result filename: {path.name}")

    normalized = pd.concat(
        [frame.reset_index(drop=True), extracted.reset_index(drop=True)], axis=1
    )
    normalized["source_file"] = path.name
    normalized["host_name"] = context.get("host_name")
    normalized["benchmark_date"] = context.get("date")
    normalized["mhz_per_cpu"] = context.get("mhz_per_cpu")
    normalized["bit_width"] = pd.to_numeric(
        normalized["bit_width"], errors="raise"
    ).astype("int64")
    normalized["blocks"] = pd.to_numeric(normalized["blocks"], errors="raise").astype(
        "int64"
    )
    if "payload_bytes" not in normalized:
        normalized["payload_bytes"] = pd.NA
    normalized["payload_bytes"] = pd.to_numeric(
        normalized["payload_bytes"], errors="coerce"
    ).astype("Int64")
    normalized["bytes_per_second"] = pd.to_numeric(
        normalized["bytes_per_second"], errors="raise"
    )
    normalized["gib_per_second"] = normalized["bytes_per_second"] / 2**30
    return normalized, context


def _result_files(
    result_dir: str | Path,
    family: ResultFamily | None = None,
) -> list[Path]:
    directory = Path(result_dir).expanduser().resolve()
    if not directory.is_dir():
        raise FileNotFoundError(
            f"Benchmark result directory does not exist: {directory}"
        )
    files = sorted(directory.glob("benchmark_*_results.json"))
    if family == "kernel":
        files = [path for path in files if PCIE_RESULT_RE.fullmatch(path.name) is None]
    elif family == "pcie":
        files = [path for path in files if PCIE_RESULT_RE.fullmatch(path.name) is not None]
    if not files:
        description = f"{family} benchmark " if family is not None else "benchmark "
        raise FileNotFoundError(f"No {description}result JSON files found in {directory}")
    return files


def load_benchmark_results(
    result_dir: str | Path,
    *,
    family: ResultFamily | None = None,
) -> pd.DataFrame:
    """Load all supported result files into one normalized dataframe."""

    frames = [
        _normalize_file(path)[0] for path in _result_files(result_dir, family=family)
    ]
    return pd.concat(frames, ignore_index=True).sort_values(
        [
            "direction",
            "value_type",
            "memory_mode",
            "bit_width",
            "blocks",
            "payload_bytes",
            "implementation",
        ],
        ignore_index=True,
    )


def load_benchmark_contexts(
    result_dir: str | Path,
    *,
    family: ResultFamily | None = None,
) -> pd.DataFrame:
    """Load one flattened Google Benchmark context row per result file."""

    rows: list[dict[str, object]] = []
    for path in _result_files(result_dir, family=family):
        _, context = _normalize_file(path)
        implementation = (
            "cp2k"
            if path.name == "benchmark_cp2k_results.json"
            else (PCIE_RESULT_RE.fullmatch(path.name) or PERNIX_RESULT_RE.fullmatch(path.name)).group("implementation")  # type: ignore[union-attr]
        )
        rows.append(
            {
                "implementation": implementation,
                "source_file": path.name,
                **context,
            }
        )
    return pd.json_normalize(rows).sort_values("implementation", ignore_index=True)


__all__ = [
    "CP2K_BENCHMARK_RE",
    "PERNIX_BENCHMARK_RE",
    "PCIE_BENCHMARK_RE",
    "load_benchmark_contexts",
    "load_benchmark_results",
]
