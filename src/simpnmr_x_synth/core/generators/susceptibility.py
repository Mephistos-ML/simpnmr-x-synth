"""Deterministic generation of model-specific susceptibility latents."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Literal, TypeAlias

import numpy as np
from simpnmr_x.core.domain.tensor import (
    SusceptibilityDecomposition,
    decompose_susceptibility_tensor,
)
from simpnmr_x.core.fitting.susceptibility.models.isoaxrho_euler import (
    IsoAxRhoEulerFitter,
)
from simpnmr_x.core.fitting.susceptibility.models.split import SplitFitter
from simpnmr_x.core.phys.susc import get_spin_only_susc

from simpnmr_x_synth.core.generators.deterministic import unit_interval_draw

if TYPE_CHECKING:
    from simpnmr_x_synth.cfg.dataset import DatasetGenerationConfig


Draw = Callable[[str, float, float], float]


@dataclass(frozen=True, slots=True)
class IsoAxRhoEulerLatents:
    """Latents expressed in the iso/ax/rho/ZYZ-Euler basis.

    Args:
        iso: Isotropic susceptibility component in Å³.
        ax: Axial susceptibility component in Å³.
        rho_over_ax: Rhombicity-to-axiality ratio.
        alpha: First ZYZ Euler angle in degrees.
        beta: Second ZYZ Euler angle in degrees.
        gamma: Third ZYZ Euler angle in degrees.
    """

    iso: float
    ax: float
    rho_over_ax: float
    alpha: float
    beta: float
    gamma: float
    model: Literal["isoaxrho_euler"] = field(init=False, default="isoaxrho_euler")

    @property
    def model_parameters(self) -> dict[str, float]:
        """Return the SimpNMR-X isoaxrho-Euler parameter mapping."""
        return {
            "iso": self.iso,
            "ax": self.ax,
            "rho_over_ax": self.rho_over_ax,
            "alpha": self.alpha,
            "beta": self.beta,
            "gamma": self.gamma,
        }

    @property
    def tensor(self) -> np.ndarray:
        """Return the Cartesian susceptibility tensor."""
        return IsoAxRhoEulerFitter.totensor(self.model_parameters)


@dataclass(frozen=True, slots=True)
class SplitLatents:
    """Latents expressed in the Cartesian isotropic/deviatoric basis.

    Args:
        iso: Isotropic susceptibility component in Å³.
        dxx: Deviatoric xx component in Å³.
        dyy: Deviatoric yy component in Å³.
        dxy: Deviatoric xy component in Å³.
        dxz: Deviatoric xz component in Å³.
        dyz: Deviatoric yz component in Å³.
    """

    iso: float
    dxx: float
    dyy: float
    dxy: float
    dxz: float
    dyz: float
    model: Literal["split"] = field(init=False, default="split")

    @property
    def model_parameters(self) -> dict[str, float]:
        """Return the SimpNMR-X Cartesian split parameter mapping."""
        return {
            "iso": self.iso,
            "dxx": self.dxx,
            "dyy": self.dyy,
            "dxy": self.dxy,
            "dxz": self.dxz,
            "dyz": self.dyz,
        }

    @property
    def tensor(self) -> np.ndarray:
        """Return the Cartesian susceptibility tensor."""
        return SplitFitter.totensor(self.model_parameters)

    @property
    def decomposition(self) -> SusceptibilityDecomposition:
        """Return the derived canonical isoaxrho-Euler representation."""
        return decompose_susceptibility_tensor(self.tensor)


GeneratedSusceptibility: TypeAlias = IsoAxRhoEulerLatents | SplitLatents


def generate_susceptibility_latents(
    *, config: "DatasetGenerationConfig", geometry_checksum: str, case_index: int
) -> GeneratedSusceptibility:
    """Generate deterministic model-specific susceptibility latents.

    Args:
        config: Validated dataset-generation configuration.
        geometry_checksum: Stable checksum of the source molecular geometry.
        case_index: Zero-based deterministic sample index.

    Returns:
        Isoaxrho-Euler or Cartesian split latents selected by the configuration.
    """

    def draw(name: str, lower: float, upper: float) -> float:
        return lower + (upper - lower) * unit_interval_draw(
            config.project.seed, geometry_checksum, case_index, name
        )

    iso = get_spin_only_susc(
        spin=config.hyperfine.spin,
        orbit=config.hyperfine.orbit,
        total_momentum_J=config.hyperfine.total_momentum_j,
        temperature=config.experiment.temperature_k,
    )
    if config.susceptibility.model == "isoaxrho_euler":
        return _generate_isoaxrho_euler_latents(config=config, iso=iso, draw=draw)
    return _generate_split_latents(config=config, iso=iso, draw=draw)


def _generate_isoaxrho_euler_latents(
    *, config: "DatasetGenerationConfig", iso: float, draw: Draw
) -> IsoAxRhoEulerLatents:
    """Generate physical isoaxrho-Euler latents."""
    rho_over_ax = config.susceptibility.rho_over_ax
    if rho_over_ax is None:
        rho_over_ax = draw("rho_over_ax", 0.0, 1.0 / 3.0)
    ax_lower, ax_upper = _ax_bounds(iso=iso, rho_over_ax=rho_over_ax)
    return IsoAxRhoEulerLatents(
        iso=iso,
        ax=draw("ax", ax_lower, ax_upper),
        rho_over_ax=rho_over_ax,
        alpha=_fixed_or_draw(config.susceptibility.alpha, "alpha", 0.0, 360.0, draw),
        beta=_fixed_or_draw(config.susceptibility.beta, "beta", 0.0, 180.0, draw),
        gamma=_fixed_or_draw(config.susceptibility.gamma, "gamma", 0.0, 360.0, draw),
    )


def _generate_split_latents(
    *, config: "DatasetGenerationConfig", iso: float, draw: Draw
) -> SplitLatents:
    """Generate physical Cartesian split latents."""
    configured = {
        name: getattr(config.susceptibility, name)
        for name in ("dxx", "dyy", "dxy", "dxz", "dyz")
    }
    components = (
        configured
        if all(value is not None for value in configured.values())
        else _sample_split_components(iso=iso, draw=draw)
    )
    return SplitLatents(
        iso=iso,
        dxx=float(components["dxx"]),
        dyy=float(components["dyy"]),
        dxy=float(components["dxy"]),
        dxz=float(components["dxz"]),
        dyz=float(components["dyz"]),
    )


def _sample_split_components(*, iso: float, draw: Draw) -> dict[str, float]:
    """Sample bounded deviations that preserve non-negative tensor eigenvalues."""
    if iso <= 0.0:
        raise ValueError("split susceptibility generation requires positive chi_iso")
    raw = {name: draw(name, -1.0, 1.0) for name in ("dxx", "dyy", "dxy", "dxz", "dyz")}
    raw_tensor = SplitFitter.totensor({"iso": 0.0, **raw})
    magnitude = float(np.max(np.abs(np.linalg.eigvalsh(raw_tensor))))
    if np.isclose(magnitude, 0.0):
        return {name: 0.0 for name in raw}
    scale = 0.95 * iso / magnitude
    return {name: float(value * scale) for name, value in raw.items()}


def _ax_bounds(*, iso: float, rho_over_ax: float) -> tuple[float, float]:
    """Return axiality bounds that keep all principal χ components non-negative."""
    return -1.5 * iso, iso / (rho_over_ax + 1.0 / 3.0)


def _fixed_or_draw(
    value: float | None,
    name: str,
    lower: float,
    upper: float,
    draw: Draw,
) -> float:
    """Return a configured Euler angle or one deterministic draw."""
    return draw(name, lower, upper) if value is None else value
