"""Deterministic physical R6 linewidth-parameter generation.

The distance-dependent component is derived from SimpNMR-X's point-dipole
Guéron Curie ``R2`` forward calculation. The distance-independent component
is sampled in Hz and converted to SimpNMR-X's internal ppm R6 convention.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from simpnmr_x.core.const.gammas import NUCLEAR_GAMMAS
from simpnmr_x.core.relaxation.gueron import calc_r2_curie

from simpnmr_x_synth.core.generators.deterministic import unit_interval_draw

if TYPE_CHECKING:
    from simpnmr_x.core.domain.mol import Molecule
    from simpnmr_x_synth.cfg.dataset import DatasetGenerationConfig


# Fixed synthetic-generation policy, not a user-facing fit parameter.
CURIE_TAU_R_S = 1.0e-9
P2_HZ_BOUNDS = (0.0, 50.0)


@dataclass(frozen=True, slots=True)
class LinewidthLatents:
    """R6 parameters in SimpNMR-X ppm convention plus human-facing Hz truth."""

    p1: float
    p2: float
    p2_hz: float


def generate_linewidth_latents(
    *,
    config: "DatasetGenerationConfig",
    molecule: "Molecule",
    geometry_checksum: str,
    case_index: int,
) -> LinewidthLatents:
    """Derive Curie ``p1`` and deterministically sample baseline ``p2`` in Hz."""
    p2_hz = P2_HZ_BOUNDS[0] + (P2_HZ_BOUNDS[1] - P2_HZ_BOUNDS[0]) * unit_interval_draw(
        config.project.seed, geometry_checksum, case_index, "p2_hz"
    )
    return LinewidthLatents(
        p1=derive_curie_p1_ppm_a6(config=config, molecule=molecule),
        p2=p2_hz / _larmor_mhz(
            molecule=molecule, magnetic_field_t=config.experiment.magnetic_field_t
        ),
        p2_hz=p2_hz,
    )


def derive_curie_p1_ppm_a6(
    *, config: "DatasetGenerationConfig", molecule: "Molecule"
) -> float:
    """Derive ``p1`` from SimpNMR-X's Curie ``R2`` forward API.

    For one isotope, the point-dipole Curie rate is exactly proportional to
    ``r^-6``. This function evaluates the SimpNMR-X forward API and removes that
    geometric factor instead of duplicating its physical expression.
    """
    _validate_r6_molecule(molecule)
    element = molecule.nuclei[0].label_nn
    gamma_mhz_t = float(NUCLEAR_GAMMAS[element])
    labels = [nucleus.label for nucleus in molecule.nuclei]
    coordinates = {
        nucleus.label: np.asarray(nucleus.coord, dtype=float) for nucleus in molecule.nuclei
    }
    gamma_rad_s_t = gamma_mhz_t * 2.0 * np.pi * 1.0e6
    omega_by_label = {
        label: gamma_rad_s_t * config.experiment.magnetic_field_t for label in labels
    }
    centre = np.asarray(molecule.paramagnetic_centre, dtype=float)
    rates = calc_r2_curie(
        labels,
        coordinates,
        centre,
        omega_by_label,
        config.experiment.temperature_k,
        CURIE_TAU_R_S,
        molecule.electronic.spin_S,
        molecule.electronic.orbit_L,
        molecule.electronic.total_J,
    )
    larmor_hz = _larmor_mhz(
        molecule=molecule, magnetic_field_t=config.experiment.magnetic_field_t
    ) * 1.0e6
    coefficients = []
    for label in labels:
        distance_a = float(np.linalg.norm(coordinates[label] - centre))
        if distance_a <= 1.0e-12:
            raise ValueError(f"Nucleus {label!r} is at the paramagnetic centre")
        linewidth_ppm = float(rates[label]) / np.pi / larmor_hz * 1.0e6
        coefficients.append(linewidth_ppm * distance_a**6)
    if not np.allclose(coefficients, coefficients[0], rtol=1.0e-12, atol=0.0):
        raise ValueError("Curie forward calculation did not produce one R6 coefficient")
    return float(coefficients[0])


def _larmor_mhz(*, molecule: "Molecule", magnetic_field_t: float) -> float:
    """Return the positive Larmor frequency for a single-element dataset."""
    _validate_r6_molecule(molecule)
    frequency = abs(float(NUCLEAR_GAMMAS[molecule.nuclei[0].label_nn]) * magnetic_field_t)
    if frequency == 0.0:
        raise ValueError("R6 linewidth generation requires a non-zero Larmor frequency")
    return frequency


def _validate_r6_molecule(molecule: "Molecule") -> None:
    if molecule.paramagnetic_centre is None:
        raise ValueError("Curie linewidth generation requires a paramagnetic centre")
    if not molecule.nuclei:
        raise ValueError("Curie linewidth generation requires at least one nucleus")
    element = molecule.nuclei[0].label_nn
    if any(nucleus.label_nn != element for nucleus in molecule.nuclei):
        raise ValueError("Curie R6 generation requires one nuclear element per dataset")
    if float(NUCLEAR_GAMMAS[element]) == 0.0:
        raise ValueError(f"No NMR gyromagnetic ratio is available for {element!r}")
