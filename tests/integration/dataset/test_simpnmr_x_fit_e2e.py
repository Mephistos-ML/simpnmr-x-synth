"""Acceptance test for the synthetic case → SimpNMR-X fit → validation loop."""

from __future__ import annotations

import csv
import os
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

from simpnmr_x_synth.app.pipelines.dataset_export import generate_dataset
from simpnmr_x_synth.app.pipelines.dataset_validation import validate_dataset_case
from simpnmr_x_synth.cfg.dataset import DatasetGenerationConfig


@pytest.mark.integration
def test_controlled_synthetic_case_recovers_chi_and_linewidth(tmp_path: Path):
    """Recover one active χ degree of freedom through the real SimpNMR-X CLI.

    The control profile fixes nuisance susceptibility coordinates. It verifies
    the file contract and numerical round-trip without asserting that one
    spectrum identifies every ``isoaxrho_euler`` degree of freedom.
    """
    if shutil.which("simpnmr-x") is None:
        pytest.skip("SimpNMR-X CLI is not installed")
    geometry = tmp_path / "model.xyz"
    geometry.write_text(
        "4\nsynthetic Yb model\nYb 0 0 0\nH 1 0 0\nH 0 1.5 0\nH 0 0 2\n",
        encoding="utf-8",
    )
    diamagnetic = tmp_path / "diamagnetic.csv"
    diamagnetic.write_text(
        "signal_label,shift\nH1,1.0\nH2,2.0\nH3,3.0\n", encoding="utf-8"
    )
    config = DatasetGenerationConfig.from_mapping(
        {
            "project": {"name": "control", "n_cases": 1, "seed": 42},
            "hyperfine": {
                "method": "pdip",
                "file": str(geometry),
                "paramagnetic_centre": [0, 0, 0],
                "spin": 0.5,
                "orbit": 3,
                "total_momentum_J": 3.5,
            },
            "nuclei": {"include": "H"},
            "diamagnetic": {"method": "csv", "file": str(diamagnetic)},
            "experiment": {"temperature_k": 302.15, "magnetic_field_t": 4.7},
            "moments": {"number_of_moments": 6},
            "linewidth": {
                "method": "r6",
            },
            "susceptibility": {
                "model": "isoaxrho_euler",
            },
        }
    )
    root = generate_dataset(config=config, output_dir=tmp_path / "output")
    case_dir = next((root / "cases").iterdir())
    fit_dir = case_dir / "SIMULATIONS" / "FITTING"
    truth = _read_one_row(case_dir / "DATA" / "CHI" / "susceptibility.csv")
    _fix_control_nuisance_variables(fit_dir / "config.yml", truth=truth)
    environment = {
        **os.environ,
        "MPLBACKEND": "Agg",
        "MPLCONFIGDIR": str(tmp_path / "mpl"),
    }
    result = subprocess.run(
        [
            "simpnmr-x",
            "--hide",
            "fit_susc",
            "config.yml",
            "--shift_plots",
            "off",
            "--spread_plots",
            "off",
            "--contrib_plots",
            "off",
            "--isoaxrho_plots",
            "off",
        ],
        cwd=fit_dir,
        env=environment,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    fitted = _read_one_row(
        fit_dir / "simpnmr_x_fitted_output" / "susceptibility_tensor.csv"
    )
    fitted_columns = {
        "chi_xx": "chi_xx (Å^3)",
        "chi_xy": "chi_xy (Å^3)",
        "chi_xz": "chi_xz (Å^3)",
        "chi_yy": "chi_yy (Å^3)",
        "chi_yz": "chi_yz (Å^3)",
        "chi_zz": "chi_zz (Å^3)",
    }
    for name, column in fitted_columns.items():
        assert abs(float(fitted[column]) - float(truth[name])) <= 5e-7
    report = yaml.safe_load(validate_dataset_case(case_dir).read_text(encoding="utf-8"))
    assert report["moment_score"] is None
    assert report["absolute_error"]["linewidth_p1"] <= 5e-7
    assert report["absolute_error"]["linewidth_p2"] <= 5e-7


def _fix_control_nuisance_variables(
    config_file: Path, *, truth: dict[str, str]
) -> None:
    """Fix non-identifiable coordinates to truth for an axiality control."""
    payload = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    variables = payload["susc_fit"]["variables"]
    truth_names = {
        "iso": "chi_iso",
        "rho_over_ax": "rh_over_ax",
        "alpha": "alpha",
        "beta": "beta",
        "gamma": "gamma",
    }
    for name, truth_name in truth_names.items():
        variables[name][0] = "fix"
        variables[name][1] = float(truth[truth_name])
    config_file.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _read_one_row(file_name: Path) -> dict[str, str]:
    with file_name.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(line for line in handle if not line.startswith("#")))
    assert len(rows) == 1
    return rows[0]
