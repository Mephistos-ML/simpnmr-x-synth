import numpy as np
from simpnmr_x.core.phys.susc import get_spin_only_susc

from simpnmr_x_synth.cfg.dataset import DatasetGenerationConfig
from simpnmr_x_synth.core.generators.susceptibility import (
    IsoAxRhoEulerLatents,
    SplitLatents,
    generate_susceptibility_latents,
)


def test_latent_sampling_is_stable_per_case_and_parameter():
    raw = {
        "project": {"name": "test", "n_cases": 2, "seed": 42},
        "hyperfine": {
            "method": "pdip",
            "file": "test.xyz",
            "paramagnetic_centre": [0, 0, 0],
            "spin": 0.5,
            "orbit": 3,
            "total_momentum_J": 3.5,
        },
        "nuclei": {"include": "H"},
        "diamagnetic": {"method": "csv", "file": "dia.csv"},
        "experiment": {"temperature_k": 302.15, "magnetic_field_t": 4.7},
        "moments": {"number_of_moments": 10},
        "linewidth": {"method": "r6"},
        "susceptibility": {"model": "isoaxrho_euler"},
    }
    config = DatasetGenerationConfig.from_mapping(raw)
    first = generate_susceptibility_latents(
        config=config, geometry_checksum="abc", case_index=0
    )
    repeated = generate_susceptibility_latents(
        config=config, geometry_checksum="abc", case_index=0
    )
    second = generate_susceptibility_latents(
        config=config, geometry_checksum="abc", case_index=1
    )

    assert first == repeated
    assert first != second
    assert isinstance(first, IsoAxRhoEulerLatents)
    assert first.iso == get_spin_only_susc(
        spin=config.hyperfine.spin,
        orbit=config.hyperfine.orbit,
        total_momentum_J=config.hyperfine.total_momentum_j,
        temperature=config.experiment.temperature_k,
    )
    principal_components = (
        first.iso + first.ax * (first.rho_over_ax - 1.0 / 3.0),
        first.iso - first.ax * (first.rho_over_ax + 1.0 / 3.0),
        first.iso + 2.0 * first.ax / 3.0,
    )
    assert min(principal_components) >= 0.0
    assert 0.0 <= first.rho_over_ax <= 1.0 / 3.0


def test_latent_sampling_respects_fixed_tensor_orientation():
    raw = {
        "project": {"name": "test", "n_cases": 1, "seed": 42},
        "hyperfine": {
            "method": "pdip",
            "file": "test.xyz",
            "paramagnetic_centre": [0, 0, 0],
            "spin": 0.5,
            "orbit": 3,
            "total_momentum_J": 3.5,
        },
        "nuclei": {"include": "H"},
        "diamagnetic": {"method": "csv", "file": "dia.csv"},
        "experiment": {"temperature_k": 302.15, "magnetic_field_t": 4.7},
        "moments": {"number_of_moments": 10},
        "linewidth": {"method": "r6"},
        "susceptibility": {
            "model": "isoaxrho_euler",
            "rho_over_ax": 0.0,
            "alpha": 0.0,
            "beta": 0.0,
            "gamma": 0.0,
        },
    }

    latent = generate_susceptibility_latents(
        config=DatasetGenerationConfig.from_mapping(raw),
        geometry_checksum="abc",
        case_index=0,
    )

    assert latent.ax != 0.0
    assert latent.rho_over_ax == latent.alpha == latent.beta == latent.gamma == 0.0


def test_split_latent_sampling_is_deterministic_and_physical():
    raw = {
        "project": {"name": "test", "n_cases": 2, "seed": 42},
        "hyperfine": {
            "method": "pdip",
            "file": "test.xyz",
            "paramagnetic_centre": [0, 0, 0],
            "spin": 0.5,
            "orbit": 3,
            "total_momentum_J": 3.5,
        },
        "nuclei": {"include": "H"},
        "diamagnetic": {"method": "csv", "file": "dia.csv"},
        "experiment": {"temperature_k": 302.15, "magnetic_field_t": 4.7},
        "moments": {"number_of_moments": 10},
        "linewidth": {"method": "r6"},
        "susceptibility": {"model": "split"},
    }
    config = DatasetGenerationConfig.from_mapping(raw)

    first = generate_susceptibility_latents(
        config=config, geometry_checksum="abc", case_index=0
    )
    repeated = generate_susceptibility_latents(
        config=config, geometry_checksum="abc", case_index=0
    )

    assert first == repeated
    assert isinstance(first, SplitLatents)
    assert first.model == "split"
    assert all(
        value is not None
        for value in (first.dxx, first.dyy, first.dxy, first.dxz, first.dyz)
    )
    assert np.all(np.linalg.eigvalsh(first.tensor) >= 0.0)
