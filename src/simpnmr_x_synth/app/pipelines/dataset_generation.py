"""Prepare SimpNMR-X molecular state for synthetic dataset generation."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from simpnmr_x.app.loaders.paramag_centre_load import load_paramagnetic_centre
from simpnmr_x.app.loaders.labels_load import load_signal_labels_from_csv
from simpnmr_x.app.loaders.dia_load import load_diamagnetic_shifts
from simpnmr_x.core.build.elstate import build_electronic_state
from simpnmr_x.core.build.hfc import build_hfc_from_pdip
from simpnmr_x.core.domain.mol import Molecule
from simpnmr_x.app.policies.linewidth_r6 import resolve_r6_linewidth_inputs
from simpnmr_x.app.policies.averaging import resolve_average_shift_groups
from simpnmr_x.core.fitting.susceptibility.linewidths import predict_r6_widths_by_atom_label
from simpnmr_x.core.fitting.susceptibility.moments.forward import (
    calculated_signal_packages_from_parameters,
    package_linewidths,
    sort_packages_by_center,
)
from simpnmr_x.core.fitting.susceptibility.models.isoaxrho_euler import (
    IsoAxRhoEulerFitter,
)
from simpnmr_x.core.fitting.susceptibility.moments.descriptors import (
    compute_gaussian_mixture_moments,
)
from simpnmr_x.core.fitting.susceptibility.moments.gaussian import (
    gaussian_peak_representation,
)
from simpnmr_x.tools.coords.xyz_fmt import add_label_indices, load_xyz

from simpnmr_x_synth.cfg.dataset import DatasetGenerationConfig
from simpnmr_x_synth.core.dataset.records import DatasetRecord, TensorTarget
from simpnmr_x_synth.core.generators.linewidth import LinewidthLatents, generate_linewidth_latents
from simpnmr_x_synth.core.generators.susceptibility import SusceptibilityLatents, generate_susceptibility_latents


@dataclass(frozen=True, slots=True)
class SyntheticPeak:
    """One synthetic Gaussian peak before moment calculation."""

    label: str
    center_ppm: float
    fwhm_ppm: float
    area: float


@dataclass(frozen=True, slots=True)
class GeneratedCase:
    """Complete synthetic case used for replayable and ML exports."""

    record: DatasetRecord
    susceptibility: SusceptibilityLatents
    linewidth: LinewidthLatents
    peaks: tuple[SyntheticPeak, ...]


def generate_case(
    *,
    config: DatasetGenerationConfig,
    molecule: Molecule,
    geometry_checksum: str,
    case_index: int,
) -> DatasetRecord:
    """Generate one complete in-memory synthetic moment-matching case."""
    return generate_case_artifacts(
        config=config,
        molecule=molecule,
        geometry_checksum=geometry_checksum,
        case_index=case_index,
    ).record


def generate_case_artifacts(
    *,
    config: DatasetGenerationConfig,
    molecule: Molecule,
    geometry_checksum: str,
    case_index: int,
) -> GeneratedCase:
    """Generate all data required to replay one synthetic SimpNMR-X case."""
    susceptibility = generate_susceptibility_latents(
        config=config,
        geometry_checksum=geometry_checksum,
        case_index=case_index,
    )
    linewidth = generate_linewidth_latents(
        config=config,
        molecule=molecule,
        geometry_checksum=geometry_checksum,
        case_index=case_index,
    )
    peaks = simulate_peaks(
        molecule=molecule,
        susceptibility=susceptibility,
        linewidth=linewidth,
        average_labels=_average_labels(config=config, molecule=molecule),
    )
    moments = peaks_to_moments(peaks=peaks, moment_labels=config.moment_labels)
    record = DatasetRecord(
        sample_id=f"{config.project.name}-{geometry_checksum[:12]}-{case_index:06d}",
        temperature_k=config.experiment.temperature_k,
        magnetic_field_t=config.experiment.magnetic_field_t,
        moments=moments,
        target=latents_to_target(susceptibility=susceptibility, linewidth=linewidth),
    )
    return GeneratedCase(
        record=record,
        susceptibility=susceptibility,
        linewidth=linewidth,
        peaks=peaks,
    )


def generate_cases(config: DatasetGenerationConfig) -> tuple[DatasetRecord, ...]:
    """Generate the configured batch of complete synthetic cases."""
    molecule, checksum = prepare_dataset_molecule(config)
    return tuple(
        generate_case(
            config=config,
            molecule=molecule,
            geometry_checksum=checksum,
            case_index=case_index,
        )
        for case_index in range(config.project.n_cases)
    )
def peaks_to_moments(
    *,
    peaks: tuple[SyntheticPeak, ...],
    moment_labels: tuple[str, ...],
) -> dict[str, float]:
    """Calculate dynamic Gaussian-mixture moments through SimpNMR-X."""
    peak_data = gaussian_peak_representation(
        centers=np.asarray([peak.center_ppm for peak in peaks], dtype=float),
        fwhm=np.asarray([peak.fwhm_ppm for peak in peaks], dtype=float),
        areas=np.asarray([peak.area for peak in peaks], dtype=float),
    )
    return compute_gaussian_mixture_moments(
        centers=peak_data["center"],
        sigmas=peak_data["sigma"],
        area_norm=peak_data["area_norm"],
        moment_labels=moment_labels,
    )


def latents_to_target(
    *, susceptibility: SusceptibilityLatents, linewidth: LinewidthLatents
) -> TensorTarget:
    """Build a canonical Cartesian χ/R6 target from sampled latents."""
    tensor = IsoAxRhoEulerFitter.totensor(
        {
            "iso": susceptibility.iso,
            "ax": susceptibility.ax,
            "rho_over_ax": susceptibility.rho_over_ax,
            "alpha": susceptibility.alpha,
            "beta": susceptibility.beta,
            "gamma": susceptibility.gamma,
        }
    )
    return TensorTarget(
        chi_xx=float(tensor[0, 0]),
        chi_xy=float(tensor[0, 1]),
        chi_xz=float(tensor[0, 2]),
        chi_yy=float(tensor[1, 1]),
        chi_yz=float(tensor[1, 2]),
        chi_zz=float(tensor[2, 2]),
        linewidth_p1=linewidth.p1,
        linewidth_p2=linewidth.p2,
    )


def prepare_dataset_molecule(config: DatasetGenerationConfig) -> tuple[Molecule, str]:
    """Load one geometry and attach SimpNMR-X-ready synthetic experiment state."""
    labels, coordinates = load_xyz(config.hyperfine.file)
    indexed_labels = add_label_indices(labels)
    molecule = Molecule.from_labels_coords(
        labels=indexed_labels,
        coords=coordinates,
        elements=config.nuclei_include,
    )
    load_paramagnetic_centre(molecule, list(config.hyperfine.paramagnetic_centre))
    molecule.electronic = build_electronic_state(
        spin_S=config.hyperfine.spin,
        orbit_L=config.hyperfine.orbit,
        total_J=config.hyperfine.total_momentum_j,
    )
    build_hfc_from_pdip(molecule)
    if config.signal_labels_file:
        labels, math_labels = load_signal_labels_from_csv(config.signal_labels_file)
        molecule.apply_signal_labels(labels, math_labels)
    checksum = geometry_checksum(labels=tuple(molecule.labels), coordinates=molecule.coords)
    dia_by_key, key_kind, ref_avg_by_label_nn = load_diamagnetic_shifts(
        file_name=config.diamagnetic.file,
        file_type=config.diamagnetic.method,
        ref_file_name=config.diamagnetic.reference_file,
        ref_file_type=config.diamagnetic.reference_method,
    )
    molecule.apply_diamagnetic_shifts(
        dia_by_key=dia_by_key,
        key_kind=key_kind,
        ref_avg_by_label_nn=ref_avg_by_label_nn,
    )
    return molecule, checksum


def geometry_checksum(*, labels: tuple[str, ...], coordinates: np.ndarray) -> str:
    """Return an order-independent checksum for labelled Cartesian geometry."""
    coordinate_array = np.asarray(coordinates, dtype=float)
    if coordinate_array.shape != (len(labels), 3):
        raise ValueError("coordinates must have shape (len(labels), 3)")
    records = sorted(
        f"{label}|{x:.12g}|{y:.12g}|{z:.12g}"
        for label, (x, y, z) in zip(labels, coordinate_array)
    )
    return hashlib.sha256("\n".join(records).encode("utf-8")).hexdigest()


def simulate_peaks(
    *,
    molecule: Molecule,
    susceptibility: SusceptibilityLatents,
    linewidth: LinewidthLatents,
    average_labels: tuple[tuple[str, ...], ...] = (),
) -> tuple[SyntheticPeak, ...]:
    """Calculate methyl-aware PCS/R6 Gaussian peak descriptors via SimpNMR-X."""
    parameters = {
        "iso": susceptibility.iso,
        "ax": susceptibility.ax,
        "rho_over_ax": susceptibility.rho_over_ax,
        "alpha": susceptibility.alpha,
        "beta": susceptibility.beta,
        "gamma": susceptibility.gamma,
    }
    packages = sort_packages_by_center(
        calculated_signal_packages_from_parameters(
            model=IsoAxRhoEulerFitter,
            parameters=parameters,
            nuclei=molecule.nuclei,
            include_diamagnetic=True,
            average_labels=average_labels,
        )
    )
    linewidth_inputs = resolve_r6_linewidth_inputs(
        molecule=molecule,
        isotope_filter=molecule.nuclei[0].isotope,
        label_kind="atom_label",
    )
    widths = predict_r6_widths_by_atom_label(
        linewidth_inputs=linewidth_inputs,
        linewidth_vars_by_name={"p1": linewidth.p1, "p2": linewidth.p2},
    )
    fwhm_ppm = package_linewidths(packages, widths)
    nucleus_by_label = {nucleus.label: nucleus for nucleus in molecule.nuclei}
    return tuple(
        SyntheticPeak(
            label=nucleus_by_label[package.atom_labels[0]].signal_label,
            center_ppm=package.center,
            fwhm_ppm=float(width),
            area=float(len(package.atom_labels)),
        )
        for package, width in zip(packages, fwhm_ppm)
    )


def _average_labels(
    *, config: DatasetGenerationConfig, molecule: Molecule
) -> tuple[tuple[str, ...], ...]:
    """Resolve synthetic observation groups from an optional label mapping."""
    if not config.signal_labels_file:
        return ()
    return tuple(
        tuple(group)
        for group in resolve_average_shift_groups(
            molecule=molecule,
            average_shifts="all",
        )
    )
