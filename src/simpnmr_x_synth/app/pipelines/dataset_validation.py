"""Compare a completed SimpNMR-X fit with one synthetic ground-truth case."""

from __future__ import annotations

import csv
import json
from pathlib import Path


_TENSOR_COLUMNS = {
    "chi_xx": "chi_xx (Å^3)",
    "chi_xy": "chi_xy (Å^3)",
    "chi_xz": "chi_xz (Å^3)",
    "chi_yy": "chi_yy (Å^3)",
    "chi_yz": "chi_yz (Å^3)",
    "chi_zz": "chi_zz (Å^3)",
}


def validate_dataset_case(case_dir: str | Path) -> Path:
    """Write a factual truth-vs-fit report for one completed replayable case."""
    root = Path(case_dir)
    truth_susceptibility = _read_one_row(root / "DATA" / "CHI" / "susceptibility.csv")
    fitted_dir = root / "SIMULATIONS" / "FITTING" / "simpnmr_x_fitted_output"
    fitted_susceptibility = _read_one_row(fitted_dir / "susceptibility_tensor.csv")
    fitted_linewidth = _read_one_row(_linewidth_output_file(fitted_dir))
    linewidth_truth = _read_dataset_linewidth_truth(root)
    truth = {
        **{name: float(truth_susceptibility[name]) for name in _TENSOR_COLUMNS},
        **linewidth_truth,
    }
    fitted = {
        **{
            name: float(fitted_susceptibility[column])
            for name, column in _TENSOR_COLUMNS.items()
        },
        "linewidth_p1": float(fitted_linewidth["p1"]),
        "linewidth_p2": float(fitted_linewidth["p2"]),
    }
    absolute_error = {
        name: abs(fitted[name] - truth[name]) for name in truth
    }
    report = {
        "status": "compared",
        "truth": truth,
        "fitted": fitted,
        "absolute_error": absolute_error,
        "max_absolute_error": max(absolute_error.values()),
        "moment_score": _optional_moment_score(fitted_dir),
    }
    output_file = root / "validation_report.json"
    output_file.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output_file


def _read_one_row(file_name: Path) -> dict[str, str]:
    with file_name.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(line for line in handle if not line.startswith("#")))
    if len(rows) != 1:
        raise ValueError(f"Expected exactly one data row in {file_name}")
    return rows[0]


def _read_dataset_linewidth_truth(case_dir: Path) -> dict[str, float]:
    """Read one case's linewidth truth from the root ML dataset artifact."""
    dataset_file = case_dir.parent.parent / "dataset.csv"
    with dataset_file.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(line for line in handle if not line.startswith("#")))
    matching_rows = [row for row in rows if row.get("sample_id") == case_dir.name]
    if len(matching_rows) != 1:
        raise ValueError(
            f"Expected exactly one dataset.csv row for sample {case_dir.name!r}"
        )
    row = matching_rows[0]
    return {
        "linewidth_p1": float(row["linewidth_p1"]),
        "linewidth_p2": float(row["linewidth_p2"]),
    }


def _single_file(directory: Path, prefix: str, *, suffix: str) -> Path:
    matches = sorted(directory.glob(f"{prefix}*{suffix}"))
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one {prefix}*{suffix} file in {directory}"
        )
    return matches[0]


def _read_moment_score(file_name: Path) -> float:
    for line in file_name.read_text(encoding="utf-8-sig").splitlines():
        if line.startswith("# score ="):
            return float(line.split("=", maxsplit=1)[1].strip())
    raise ValueError(f"Missing GMM score comment in {file_name}")


def _linewidth_output_file(directory: Path) -> Path:
    """Find the R6 output produced by either fixed or moments fitting."""
    for prefix in ("linewidth_estimate_", "linewidth_model_"):
        matches = sorted(directory.glob(f"{prefix}*.csv"))
        if len(matches) == 1:
            return matches[0]
    raise ValueError(f"Missing unique fitted linewidth output in {directory}")


def _optional_moment_score(directory: Path) -> float | None:
    matches = sorted(directory.glob("moment_fit_diagnostics_*.csv"))
    if not matches:
        return None
    if len(matches) != 1:
        raise ValueError(f"Expected one moment diagnostics file in {directory}")
    return _read_moment_score(matches[0])
