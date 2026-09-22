"""Typed configuration section models."""

from __future__ import annotations

from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class ProjectConfig:
    name: str
    n_cases: int
    seed: int


@dataclass(frozen=True, slots=True)
class HyperfineConfig:
    file: str
    paramagnetic_centre: tuple[float, float, float]
    spin: float
    orbit: float
    total_momentum_j: float


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    temperature_k: float
    magnetic_field_t: float


@dataclass(frozen=True, slots=True)
class DiamagneticConfig:
    method: str
    file: str
    reference_method: str = ""
    reference_file: str = ""


@dataclass(frozen=True, slots=True)
class LinewidthConfig:
    method: str


@dataclass(frozen=True, slots=True)
class SusceptibilityConfig:
    model: str
    rho_over_ax: float | None = None
    alpha: float | None = None
    beta: float | None = None
    gamma: float | None = None
