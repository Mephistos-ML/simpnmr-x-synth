from pathlib import Path

from simpnmr_x_synth.app.pipelines.dataset_generation import (
    generate_case,
    generate_case_artifacts,
    generate_cases,
    prepare_dataset_molecule,
    simulate_peaks,
)
from simpnmr_x_synth.cfg.dataset import DatasetGenerationConfig
from simpnmr_x_synth.core.generators.linewidth import LinewidthLatents
from simpnmr_x_synth.core.generators.susceptibility import IsoAxRhoEulerLatents


def test_prepare_dataset_molecule_attaches_pdip_and_diamagnetic_shifts(tmp_path: Path):
    geometry = tmp_path / "model.xyz"
    geometry.write_text(
        "3\nsynthetic Yb model\nYb 0.0 0.0 0.0\nH 1.0 0.0 0.0\nH 0.0 1.0 0.0\n",
        encoding="utf-8",
    )
    diamagnetic = tmp_path / "diamagnetic.csv"
    diamagnetic.write_text("atom_label,shift\nH1,1.0\nH2,2.0\n", encoding="utf-8")
    config = DatasetGenerationConfig.from_mapping(
        {
            "project": {"name": "test", "n_cases": 1, "seed": 42},
            "hyperfine": {
                "method": "pdip",
                "file": str(geometry),
                "paramagnetic_centre": [0.0, 0.0, 0.0],
                "spin": 0.5,
                "orbit": 3,
                "total_momentum_J": 3.5,
            },
            "nuclei": {"include": "H"},
            "diamagnetic": {"method": "csv", "file": str(diamagnetic)},
            "experiment": {"temperature_k": 302.15, "magnetic_field_t": 4.7},
            "moments": {"number_of_moments": 10},
            "linewidth": {"method": "r6"},
            "susceptibility": {"model": "isoaxrho_euler"},
        }
    )

    molecule, checksum = prepare_dataset_molecule(config)

    assert checksum
    assert [nucleus.label for nucleus in molecule.nuclei] == ["H1", "H2"]
    assert all(nucleus.A.tensor_full is not None for nucleus in molecule.nuclei)
    assert [nucleus.shift.dia for nucleus in molecule.nuclei] == [1.0, 2.0]

    peaks = simulate_peaks(
        molecule=molecule,
        susceptibility=IsoAxRhoEulerLatents(
            iso=0.0,
            ax=0.02,
            rho_over_ax=0.1,
            alpha=0.0,
            beta=0.0,
            gamma=0.0,
        ),
        linewidth=LinewidthLatents(p1=705.05, p2=0.25, p2_hz=50.0),
    )

    assert len(peaks) == 2
    assert all(peak.fwhm_ppm > 0.0 for peak in peaks)

    case = generate_case(
        config=config,
        molecule=molecule,
        geometry_checksum=checksum,
        case_index=0,
    )

    assert tuple(case.moments) == tuple(f"m{index}" for index in range(1, 11))
    assert case.target.linewidth_p1 > 0.0

    artifacts = generate_case_artifacts(
        config=config, molecule=molecule, geometry_checksum=checksum, case_index=0
    )
    assert 0.0 <= artifacts.linewidth.p2_hz <= 50.0
    assert artifacts.linewidth.p2 == artifacts.linewidth.p2_hz / (42.57747844 * 4.7)

    batch = generate_cases(config)

    assert len(batch) == 1
    assert batch[0] == case


def test_synthetic_peaks_apply_chemical_label_averaging(tmp_path: Path):
    geometry = tmp_path / "model.xyz"
    geometry.write_text(
        "3\nsynthetic Yb model\nYb 0 0 0\nH 1 0 0\nH 0 1 0\n",
        encoding="utf-8",
    )
    diamagnetic = tmp_path / "diamagnetic.csv"
    diamagnetic.write_text("atom_label,shift\nH1,1.0\nH2,2.0\n", encoding="utf-8")
    labels = tmp_path / "labels.csv"
    labels.write_text("atom_label,signal_label\nH1,Ha\nH2,Ha\n", encoding="utf-8")
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
            "signal_labels": {"file": str(labels)},
            "diamagnetic": {"method": "csv", "file": str(diamagnetic)},
            "experiment": {"temperature_k": 302.15, "magnetic_field_t": 4.7},
            "moments": {"number_of_moments": 3},
            "linewidth": {"method": "r6"},
            "susceptibility": {"model": "isoaxrho_euler"},
        }
    )
    molecule, checksum = prepare_dataset_molecule(config)

    generated = generate_case_artifacts(
        config=config,
        molecule=molecule,
        geometry_checksum=checksum,
        case_index=0,
    )

    assert len(generated.peaks) == 1
    assert generated.peaks[0].label == "Ha"
    assert generated.peaks[0].area == 2.0
