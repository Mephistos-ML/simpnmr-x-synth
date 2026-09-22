"""Deterministic susceptibility-latent generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from simpnmr_x.core.phys.susc import get_spin_only_susc
from simpnmr_x_synth.core.generators.deterministic import unit_interval_draw

if TYPE_CHECKING:
    from simpnmr_x_synth.cfg.dataset import DatasetGenerationConfig


@dataclass(frozen=True, slots=True)
class SusceptibilityLatents:
    """Sampled iso/ax/rho/Euler susceptibility parameters."""

    iso: float
    ax: float
    rho_over_ax: float
    alpha: float
    beta: float
    gamma: float


def generate_susceptibility_latents(
    *, config: "DatasetGenerationConfig", geometry_checksum: str, case_index: int
) -> SusceptibilityLatents:
    """Generate deterministic susceptibility parameters for one case."""
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
    rho_over_ax = config.susceptibility.rho_over_ax
    if rho_over_ax is None:
        rho_over_ax = draw("rho_over_ax", 0.0, 1.0 / 3.0)
    ax_lower, ax_upper = _ax_bounds(iso=iso, rho_over_ax=rho_over_ax)
    return SusceptibilityLatents(
        iso=iso,
        ax=draw("ax", ax_lower, ax_upper),
        rho_over_ax=rho_over_ax,
        alpha=_fixed_or_draw(config.susceptibility.alpha, "alpha", 0.0, 360.0, draw),
        beta=_fixed_or_draw(config.susceptibility.beta, "beta", 0.0, 180.0, draw),
        gamma=_fixed_or_draw(config.susceptibility.gamma, "gamma", 0.0, 360.0, draw),
    )


def _ax_bounds(*, iso: float, rho_over_ax: float) -> tuple[float, float]:
    """Return axiality bounds that keep all principal χ components non-negative."""
    return -1.5 * iso, iso / (rho_over_ax + 1.0 / 3.0)


def _fixed_or_draw(
    value: float | None,
    name: str,
    lower: float,
    upper: float,
    draw,
) -> float:
    """Return a configured Euler angle or one deterministic draw."""
    return draw(name, lower, upper) if value is None else value
