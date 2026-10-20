"""Magnetic Feshbach resonances: the pure computation.

Van der Waals scales (β₆, ā) from C₆ and the mass, the isolated-resonance
scattering length a(B) = a_bg (1 − Δ/(B − B₀)) (Chin, Grimm, Julienne &
Tiesinga, Rev. Mod. Phys. 82, 1225 (2010)), the product of all resonances, and
the elastic cross section. No files, prints or plots: the
``feshbach`` pipeline loads the literature table, C₆ and mass and passes them in.

Known issues (PHYSICS_BUGS.md): ā uses the R_vdW factor with β₆ (P1); the
combined a(B) multiplies resonances from different channels (P24); σ_el
lacks the identical-boson factor 2 (P25). Kept as is so results stay
comparable until they are fixed deliberately.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from linak.core.constants import (
    A0_M,
    A_BAR_FACTOR,
    AMU_KG,
    AU_TO_SI_C6,
    HBAR_SI,
    K_B,
)


# ============================================================================
# PHYSICS
# ============================================================================

def reduced_mass_kg(mass_amu):
    """Identical-atom reduced mass μ = m/2."""
    m = mass_amu * AMU_KG
    return 0.5 * m


def beta6_a0(C6_au, mass_amu):
    """
    van der Waals length β₆ = (2 μ C₆ / ℏ²)^{1/4} in units of a₀.

    Working in SI then converting to a₀ keeps units transparent.
    """
    mu = reduced_mass_kg(mass_amu)
    C6_SI = C6_au * AU_TO_SI_C6
    beta_m = (2.0 * mu * C6_SI / HBAR_SI**2) ** 0.25
    return beta_m / A0_M


def a_bar_a0(beta6):
    """Mean scattering length ā ≈ 0.956 β₆ (Gribakin–Flambaum)."""
    return A_BAR_FACTOR * beta6


def scattering_length(B, a_bg, B0, Delta):
    """
    a(B) = a_bg (1 − Δ/(B − B₀)).

    Returns NaN exactly at B = B₀ (pole).
    """
    B = np.asarray(B, dtype=float)
    denom = B - B0
    with np.errstate(divide='ignore', invalid='ignore'):
        a = a_bg * (1.0 - Delta / denom)
    a = np.where(np.abs(denom) < 1e-12, np.nan, a)
    return a


def wave_number_a0(kT_uK, mass_amu):
    """
    Thermal wave number k = √(2 μ k_B T) / ℏ, returned in a₀⁻¹.

    Uses E = k_B T (not 3/2 kT) — convention for quoting a single energy scale.
    """
    mu = reduced_mass_kg(mass_amu)
    E = K_B * (kT_uK * 1e-6)
    k_SI = math.sqrt(2.0 * mu * E) / HBAR_SI
    return k_SI * A0_M   # 1/m · m/a₀⁻¹ → a₀⁻¹


def sigma_el_a0(a_a0, k_a0inv):
    """σ_el / a₀² = 4π a² / (1 + k² a²)."""
    a = np.asarray(a_a0, dtype=float)
    ka2 = (k_a0inv * a) ** 2
    with np.errstate(invalid='ignore'):
        return 4.0 * np.pi * a**2 / (1.0 + ka2)


@dataclass(frozen=True)
class Resonance:
    """One literature resonance (a row of ``<El>_feshbach.json``)."""

    label: str
    channel: str
    partial_wave: str
    B0_G: float
    Delta_G: float
    a_bg_a0: float | None   # None: use the table's default background
    ref: str

    @classmethod
    def from_json(cls, row: Mapping) -> Resonance:
        B0 = float(row['B0_G'])
        a_bg = row.get('a_bg_a0')
        return cls(
            label=row.get('label', f"B0={B0}G"),
            channel=row.get('channel', ''),
            partial_wave=row.get('partial_wave', 's'),
            B0_G=B0,
            Delta_G=float(row['Delta_G']),
            a_bg_a0=None if a_bg is None else float(a_bg),
            ref=row.get('ref', ''),
        )

    def background(self, a_bg_default: float) -> float:
        return self.a_bg_a0 if self.a_bg_a0 is not None else a_bg_default


@dataclass(frozen=True)
class ResonanceCurve:
    """a(B) and σ(B) of one resonance on its own (isolated single pole)."""

    resonance: Resonance
    a_bg_a0: float
    a_a0: np.ndarray
    sigma_a02: np.ndarray


@dataclass(frozen=True)
class FeshbachResult:
    C6_au: float
    mass_amu: float
    beta6_a0: float
    a_bar_a0: float
    a_bg_default_a0: float
    kT_uK: float
    k_a0inv: float
    B_G: np.ndarray
    a_combined_a0: np.ndarray
    sigma_combined_a02: np.ndarray
    curves: list[ResonanceCurve]


def multi_resonance_a(B, resonances: Sequence[Resonance], a_bg_default):
    """
    Product of isolated-resonance factors (valid when resonances are well
    separated compared to their widths):

        a(B)/a_bg = Π_i (1 − Δ_i / (B − B₀,i))

    The overall a_bg is the first resonance's own, else *a_bg_default*
    (standard when all share one open channel; see P24).
    """
    B = np.asarray(B, dtype=float)
    a_bg = resonances[0].background(a_bg_default)
    factor = np.ones_like(B, dtype=float)
    for res in resonances:
        B0 = res.B0_G
        Delta = res.Delta_G
        denom = B - B0
        with np.errstate(divide='ignore', invalid='ignore'):
            f = 1.0 - Delta / denom
        f = np.where(np.abs(denom) < 1e-12, np.nan, f)
        factor = factor * f
    return a_bg * factor


def default_B_max(resonances: Sequence[Resonance]) -> float:
    """15 % past the highest resonance, at least 500 G."""
    return max(max(r.B0_G for r in resonances) * 1.15, 500.0)


def compute(
    resonances: Sequence[Resonance], a_bg_default_a0: float, C6_au: float, mass_amu: float,
    *, kT_uK: float = 1.0, B_max: float | None = None, B_points: int = 2000,
) -> FeshbachResult:
    """Van der Waals scales, combined and per-resonance a(B) and σ(B) on 0..B_max."""
    beta6 = beta6_a0(C6_au, mass_amu)
    k_th = wave_number_a0(kT_uK, mass_amu)
    B = np.linspace(0.0, default_B_max(resonances) if B_max is None else B_max, B_points)
    a_comb = multi_resonance_a(B, resonances, a_bg_default_a0)
    curves = []
    for res in resonances:
        a_bg = res.background(a_bg_default_a0)
        a = scattering_length(B, a_bg, res.B0_G, res.Delta_G)
        curves.append(ResonanceCurve(res, a_bg, a, sigma_el_a0(a, k_th)))
    return FeshbachResult(
        C6_au=C6_au,
        mass_amu=mass_amu,
        beta6_a0=beta6,
        a_bar_a0=a_bar_a0(beta6),
        a_bg_default_a0=a_bg_default_a0,
        kT_uK=kT_uK,
        k_a0inv=k_th,
        B_G=B,
        a_combined_a0=a_comb,
        sigma_combined_a02=sigma_el_a0(a_comb, k_th),
        curves=curves,
    )
