import csv
import json
from pathlib import Path

from simpnmr_x.cfg.config import FitSuscConfig

from simpnmr_x_synth.app.pipelines.dataset_export import generate_dataset
from simpnmr_x_synth.cfg.dataset import DatasetGenerationConfig


def test_generate_dataset_writes_replayable_cases_and_paired_ml_table(tmp_path: Path):
    geometry = tmp_path / "model.xyz"
    geometry.write_text(
        "3\nsynthetic Yb model\nYb 0 0 0\nH 1 0 0\nH 0 1 0\n",
        encoding="utf-8",
    )
    diamagnetic = tmp_path / "diamagnetic.csv"
    diamagnetic.write_text("signal_label,shift\nH1,1.0\nH2,2.0\n", encoding="utf-8")
    config = DatasetGenerationConfig.from_mapping(
        {
            "project": {"name": "yb", "n_cases": 2, "seed": 42},
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
            "moments": {"number_of_moments": 3},
            "linewidth": {"method": "r6"},
            "susceptibility": {"model": "isoaxrho_euler"},
        }
    )

    root = generate_dataset(config=config, output_dir=tmp_path / "output")

    with (root / "dataset.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(line for line in handle if not line.startswith("#")))
    assert len(rows) == 2
    assert set(rows[0]) == {
        "sample_id",
        "m1",
        "m2",
        "m3",
        "chi_xx",
        "chi_xy",
        "chi_xz",
        "chi_yy",
        "chi_yz",
        "chi_zz",
        "linewidth_p1",
        "linewidth_p2",
    }
    case_root = root / "cases" / rows[0]["sample_id"]
    fit_config = case_root / "SIMULATIONS" / "FITTING" / "config.yml"
    gmm_config = case_root / "SIMULATIONS" / "FITTING" / "gmm_config.yml"
    assert fit_config.is_file()
    assert gmm_config.is_file()
    assert (case_root / "DATA" / "HFC" / "geometry.xyz").is_file()
    assert (case_root / "DATA" / "PARA" / "generated_shifts.csv").is_file()
    assert (case_root / "DATA" / "DIA" / "diamagnetic.csv").is_file()
    assert (case_root / "DATA" / "CHI" / "susceptibility.csv").is_file()
    csv_artifacts = [
        root / "dataset.csv",
        case_root / "DATA" / "PARA" / "generated_shifts.csv",
        case_root / "DATA" / "CHI" / "susceptibility.csv",
    ]
    for artifact in csv_artifacts:
        assert artifact.read_text(encoding="utf-8-sig").startswith(
            "# This file was generated with SimpNMR-X-Synth v"
        )
    assert "simpnmr_x_fitted_output" in fit_config.read_text()
    assert FitSuscConfig.from_file(fit_config).assignment_method == "fixed"
    generated_gmm = FitSuscConfig.from_file(gmm_config)
    assert generated_gmm.assignment_method == "moments"
    assert generated_gmm.assignment_max_moment_order == 3
    assert generated_gmm.susc_fit_type == "split"
    assert generated_gmm.susc_fit_variables["iso"] == ["fix", 0.0]
    assert len(generated_gmm.susc_fit_variables) == 6
    assert all(
        value[0] == "fit"
        for name, value in generated_gmm.susc_fit_variables.items()
        if name != "iso"
    )
    assert all(
        value[0] == "fit" for value in generated_gmm.linewidth_variables.values()
    )
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["normalized_config"]["number_of_moments"] == 3
    assert "simpnmr_x_version" in manifest["generator"]


def test_fixed_profile_exports_simpnmr_x_r6_linewidth_estimation(tmp_path: Path):
    geometry = tmp_path / "model.xyz"
    geometry.write_text(
        "3\nsynthetic Yb model\nYb 0 0 0\nH 1 0 0\nH 0 1 0\n",
        encoding="utf-8",
    )
    diamagnetic = tmp_path / "diamagnetic.csv"
    diamagnetic.write_text("signal_label,shift\nH1,1.0\nH2,2.0\n", encoding="utf-8")
    config = DatasetGenerationConfig.from_mapping(
        {
            "project": {"name": "yb", "n_cases": 1, "seed": 42},
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
            "moments": {"number_of_moments": 3},
            "linewidth": {"method": "r6"},
            "susceptibility": {"model": "isoaxrho_euler"},
        }
    )

    root = generate_dataset(config=config, output_dir=tmp_path / "output")
    fit_config = next((root / "cases").glob("*/SIMULATIONS/FITTING/config.yml"))

    parsed = FitSuscConfig.from_file(fit_config)
    assert parsed.linewidth_method == "experimental"
    assert parsed.linewidth_estimate == "p1_p2"


def test_split_model_exports_split_replay_config(tmp_path: Path):
    geometry = tmp_path / "model.xyz"
    geometry.write_text(
        "3\nsynthetic Yb model\nYb 0 0 0\nH 1 0 0\nH 0 1 0\n",
        encoding="utf-8",
    )
    diamagnetic = tmp_path / "diamagnetic.csv"
    diamagnetic.write_text("signal_label,shift\nH1,1.0\nH2,2.0\n", encoding="utf-8")
    config = DatasetGenerationConfig.from_mapping(
        {
            "project": {"name": "yb", "n_cases": 1, "seed": 42},
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
            "moments": {"number_of_moments": 3},
            "linewidth": {"method": "r6"},
            "susceptibility": {"model": "split"},
        }
    )

    root = generate_dataset(config=config, output_dir=tmp_path / "output")
    case_root = next((root / "cases").iterdir())
    replay = FitSuscConfig.from_file(
        case_root / "SIMULATIONS" / "FITTING" / "config.yml"
    )

    assert replay.susc_fit_type == "split"
    assert set(replay.susc_fit_variables) == {"iso", "dxx", "dyy", "dxy", "dxz", "dyz"}
    truth = (case_root / "DATA" / "CHI" / "susceptibility.csv").read_text()
    assert "dxx" in truth
