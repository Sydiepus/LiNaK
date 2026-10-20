"""Hyperfine structure, Zeeman and Breit-Rabi: the pure computation.

For each state |n l J⟩ of an isotope with nuclear spin I::

    E(F) = (A/2)·K + B·[3K(K+1)/4 − I(I+1)J(J+1)] / [2I(2I−1)·2J(2J−1)]
           + C·(octupole term)          K = F(F+1) − I(I+1) − J(J+1)

    g_F from g_J (Landé, or an experimental override) and g_I; the
    Breit-Rabi levels diagonalise H_hf + (g_J μ_B J_z − g_I μ_N I_z)·B.

A state without measured constants borrows them from a measured state with
the same (l, J), scaled as A ∝ n*⁻³ with the quantum defect of l.

Known issues (PHYSICS_BUGS.md): A = 0 doubles as "not measured" (P17); the
Breit-Rabi quadrupole is diagonal-only (P4); the clock interval uses A alone
(P7) and belongs to the first state given (P40); the octupole term is wrong
(P30, latent: every bundled C = 0). Kept as is so results stay comparable
until they are fixed deliberately.

No files, prints or plots here: the ``hyperfine`` pipeline loads the nuclear
moments, the constants table and the quantum defects and passes them in.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from linak.core.constants import MU_B_MHz_G, MU_N_MHz_G
from linak.physics.polarizability_core import L_CHAR

CLOCK_NOTE = 'Zero-field ground ΔF=1 interval (m_F=0 clock to first order in B)'


@dataclass(frozen=True)
class Nucleus:
    """One isotope's nuclear moments (a row of ``nuclear_data.json``)."""

    A: int
    I: float
    mu_I: float     # nuclear magnetons
    Q_barn: float
    gI: float       # μ_I / I

    @classmethod
    def from_json(cls, row: Mapping) -> Nucleus:
        # An unknown moment reads as 0: no nuclear Zeeman term.
        return cls(A=row['A'], I=row['I'], mu_I=row['mu_I'] or 0.0,
                   Q_barn=row['Q_barn'], gI=row['g_I'] or 0.0)


@dataclass(frozen=True)
class HfConstants:
    """Measured constants of one state (a row of ``<El>_hf_constants.json``)."""

    A_MHz: float
    B_MHz: float = 0.0    # electric quadrupole; absent reads as 0
    C_kHz: float = 0.0    # magnetic octupole; absent reads as 0
    gJ_exp: float | None = None   # overrides the Landé g_J
    S_exp: float | None = None    # effective spin for Landé (default 1/2)

    @classmethod
    def from_json(cls, row: Mapping) -> HfConstants:
        B = row.get('B_MHz')
        C = row.get('C_kHz')
        return cls(
            A_MHz=float(row['A_MHz']),
            B_MHz=0.0 if B is None else float(B),
            C_kHz=0.0 if C is None else float(C),
            gJ_exp=float(row['gJ_exp']) if 'gJ_exp' in row else None,
            S_exp=float(row['S_exp']) if 'S_exp' in row else None,
        )


@dataclass(frozen=True)
class FLevel:
    """One hyperfine level F of a state."""

    F: float
    E_MHz: float        # relative to the state's centroid
    gF: float
    mF: list[float]     # +F … −F


@dataclass(frozen=True)
class HyperfineState:
    label: str
    n: int
    l: str
    J: float
    gJ: float
    gJ_source: str      # 'experimental' or 'Lande'
    A_MHz: float
    B_MHz: float
    C_kHz: float
    extrapolated_from: str | None   # the measured state A, B, C were scaled from
    levels: list[FLevel]
    BR_energies: np.ndarray         # (B point, level) Breit-Rabi eigenvalues, MHz
    BR_labels: list[tuple[float, float]]   # (F, m_F) of each level at B = 0


@dataclass(frozen=True)
class MicrowaveClock:
    """The zero-field ΔF = 1 interval between a state's two highest F levels."""

    F_upper: float
    F_lower: float
    frequency_MHz: float
    frequency_GHz: float
    frequency_Hz: float
    wavelength_m: float | None    # None for degenerate levels (A = 0)
    wavelength_cm: float | None
    note: str
    state: str


@dataclass(frozen=True)
class HyperfineResult:
    nucleus: Nucleus
    B_G: np.ndarray
    states: list[HyperfineState]
    clock: MicrowaveClock | None    # of the first state (P40)
    skipped: list[str]              # labels that are not of the form '3p3/2'


# ============================================================================
# STATE LABELS AND F LEVELS
# ============================================================================

def parse_state_label(label):
    """
    Parse a state label like '3s1/2', '3p3/2', '4d5/2', '3s' into (n, l, J).
    Returns (n, l_char, J_float) or None.
    """
    m = re.match(r'^(\d+)([spdfgh])([\d/]+)?$', label.strip())
    if not m:
        return None
    n      = int(m.group(1))
    l_char = m.group(2)
    j_str  = m.group(3)
    if j_str:
        if '/' in j_str:
            jn, jd = j_str.split('/')
            J = int(jn) / int(jd)
        else:
            J = float(j_str)
    else:
        # Infer J from l and ground-state convention
        l_int = L_CHAR[l_char]
        J = l_int + 0.5   # default to J = l + 1/2 for doublet
    return n, l_char, J


def F_levels(I, J):
    """Return list of allowed F quantum numbers: |I-J|, ..., I+J."""
    F_min = abs(I - J)
    F_max = I + J
    Fs = []
    F = F_min
    while F <= F_max + 1e-9:
        Fs.append(round(F * 2) / 2)   # keep half-integer precision
        F += 1
    return Fs


# ============================================================================
# HYPERFINE ENERGIES AND g FACTORS
# ============================================================================

def hyperfine_energy(F, I, J, A_MHz, B_MHz=0.0, C_kHz=0.0):
    """
    Hyperfine energy of state |I, J, F⟩ in MHz relative to the centroid.

    Includes three multipole terms:
      Magnetic dipole (A):
        E_A = (A/2) · K
      Electric quadrupole (B):
        E_B = B · [3K(K+1)/4 − I(I+1)J(J+1)] / [2I(2I−1)·2J(2J−1)]
      Magnetic octupole (C):
        E_C = C · K(K+1)(K+2) · [5K − 6I(I+1) − 6J(J+1) + ... ] / denom_oct
        Simplified:  E_C = C · [5K³ + 3K²(1−2I(I+1)−2J(J+1)) + ...] / denom_oct
      where K = F(F+1) − I(I+1) − J(J+1)

    A in MHz, B in MHz, C in kHz (converted internally to MHz).
    B term requires I >= 1 AND J >= 1.
    C term requires I >= 3/2 AND J >= 3/2 (else identically zero).

    Reference for octupole formula:
      Armstrong (1971) Adv. At. Mol. Phys. 7:149
      Woodgate (1966) Proc. Roy. Soc. A293:117
    """
    K = F*(F+1) - I*(I+1) - J*(J+1)
    E = 0.5 * A_MHz * K

    # Quadrupole term — requires I >= 1 AND J >= 1
    if B_MHz != 0.0 and I >= 1.0 - 1e-9 and J >= 1.0 - 1e-9:
        denom_IJ = (2*I - 1) * (2*I) * (2*J - 1) * (2*J)
        if abs(denom_IJ) > 1e-12:
            E += B_MHz * (1.5*K*(K+1) - 2*I*(I+1)*J*(J+1)) / denom_IJ

    # Octupole term — requires I >= 3/2 AND J >= 3/2
    # Uses the Schwartz (1955) / Armstrong (1971) denominator:
    #   (2I−1)(2I)(2I+1)(2I+2) × (2J−1)(2J)(2J+1)(2J+2)
    # which is non-zero for I >= 3/2, J >= 3/2 (unlike the naive 2I-3 form).
    # Typical magnitude: sub-kHz shifts, informational for all but optical clocks.
    if C_kHz != 0.0 and I >= 1.5 - 1e-9 and J >= 1.5 - 1e-9:
        C_MHz = C_kHz * 1e-3
        denom_oct = (
            (2*I - 1) * (2*I) * (2*I + 1) * (2*I + 2) *
            (2*J - 1) * (2*J) * (2*J + 1) * (2*J + 2)
        )
        if abs(denom_oct) > 1e-12:
            II  = I * (I + 1)
            JJ  = J * (J + 1)
            # Armstrong (1971) Adv. At. Mol. Phys. 7:149, eq. (3.5)
            E_oct = (10*K*(K+1)*(K+2)
                     - 12*(K+1)*II*JJ
                     + 3*(K+1)*(II + JJ))
            E += C_MHz * E_oct / denom_oct
    return E


def lande_gJ(L, S, J, gJ_exp=None):
    """
    Landé g_J factor.
    g_J = 1 + [J(J+1) + S(S+1) − L(L+1)] / [2J(J+1)]

    For non-alkali atoms or when higher accuracy is needed, pass
    gJ_exp (experimental value from NIST or literature) to override
    the Landé formula. This handles:
      - Heavy atoms where LS coupling breaks down (jj coupling)
      - Multi-electron atoms (Yb, Sr, Ca, Hg, etc.)
      - States where QED corrections matter (>0.1% level)
    For J=0 returns 0 regardless.
    """
    if J < 1e-9:
        return 0.0
    if gJ_exp is not None:
        return float(gJ_exp)
    return 1.0 + (J*(J+1) + S*(S+1) - L*(L+1)) / (2*J*(J+1))


def gF_factor(F, I, J, gJ, gI=0.0):
    """
    g_F factor in the weak-field Zeeman regime.
    g_F = g_J·[F(F+1)+J(J+1)−I(I+1)] / [2F(F+1)]
          − g_I·(μ_N/μ_B)·[F(F+1)+I(I+1)−J(J+1)] / [2F(F+1)]
    μ_N/μ_B ≈ 1/1836.15 (ratio of nuclear to Bohr magneton)
    """
    if F < 1e-9:
        return 0.0
    FF = F*(F+1)
    JJ = J*(J+1)
    II = I*(I+1)
    gF  =  gJ * (FF + JJ - II) / (2*FF)
    gF -= gI * (1/1836.15267) * (FF + II - JJ) / (2*FF)
    return gF


# ============================================================================
# BREIT-RABI DIAGONALISATION
# ============================================================================

def breit_rabi_energies(I, J, A_MHz, B_MHz_quad, gJ, gI,
                        B_gauss_array):
    """
    Exact Breit-Rabi diagonalisation for arbitrary B field.

    Builds the full (2I+1)(2J+1) × (2I+1)(2J+1) Hamiltonian matrix
    H = H_hf + H_Z   at each B value and diagonalises it.

    H_hf couples all |m_I, m_J⟩ states.
    H_Z  = (g_J·μ_B·m_J − g_I·μ_N·m_I)·B   (diagonal in |m_I, m_J⟩ basis)

    Basis: |m_I, m_J⟩ ordered as:
      m_I = −I, −I+1, ..., +I
      m_J = −J, −J+1, ..., +J
    Combined index: idx = (m_I_index)*(2J+1) + m_J_index

    Returns:
      energies[B_idx, level_idx]  — sorted eigenvalues in MHz
      labels[level_idx]           — approximate (F, m_F) label at B→0
    """
    dim_I = round(2*I + 1)
    dim_J = round(2*J + 1)
    dim   = dim_I * dim_J

    mI_vals = np.array([-I + i for i in range(dim_I)])
    mJ_vals = np.array([-J + j for j in range(dim_J)])

    def idx(mi_i, mj_i):
        return mi_i * dim_J + mj_i

    # ── Build H_hf  (I·J operator in |m_I, m_J⟩ basis) ─────────────────────
    # I·J = I_z·J_z + (1/2)(I_+·J_- + I_-·J_+)
    # Matrix elements:
    #   ⟨m_I, m_J|I_z·J_z|m_I, m_J⟩ = m_I·m_J
    #   ⟨m_I+1, m_J-1|I_+·J_-|m_I, m_J⟩ = sqrt[(I-m_I)(I+m_I+1)(J+m_J)(J-m_J+1)]

    H_hf = np.zeros((dim, dim))

    # Diagonal: A/2 * K where K = F(F+1)-I(I+1)-J(J+1)
    # But in |m_I,m_J⟩ basis H_hf is not diagonal.
    # Build it directly from I·J:
    for mi_i, mI in enumerate(mI_vals):
        for mj_i, mJ in enumerate(mJ_vals):
            r = idx(mi_i, mj_i)
            # I_z J_z term
            H_hf[r, r] += A_MHz * mI * mJ

            # Quadrupole B term — diagonal in |m_I, m_J⟩ basis:
            # 3(I_z)²·(J_z)² - I²J²  ...  complex in this basis, use full tensor
            # For simplicity include leading quadrupole correction diagonally:
            # B·[3(m_I·m_J)² + 3(m_I·m_J)/2 - I(I+1)J(J+1)] / denom
            if B_MHz_quad != 0.0 and I >= 1 and J >= 1:
                denom = (2*I-1)*2*I*(2*J-1)*2*J
                if abs(denom) > 1e-12:
                    u = mI * mJ
                    H_hf[r, r] += B_MHz_quad * (3*u*u + 1.5*u - I*(I+1)*J*(J+1)) / denom

            # I_+ J_- term: ⟨m_I+1, m_J-1|...|m_I, m_J⟩
            if mi_i + 1 < dim_I and mj_i - 1 >= 0:
                r2   = idx(mi_i+1, mj_i-1)
                me   = (math.sqrt((I - mI)*(I + mI + 1)) *
                        math.sqrt((J + mJ)*(J - mJ + 1)))
                H_hf[r2, r] += 0.5 * A_MHz * me
                H_hf[r, r2] += 0.5 * A_MHz * me   # Hermitian

    # ── Compute H_Z diagonal at each B ──────────────────────────────────────
    # H_Z[r] = (gJ·μ_B·mJ − gI·μ_N·mI) · B
    H_Z_diag = np.array([
        gJ * MU_B_MHz_G * mJ_vals[mj_i]
        - gI * MU_N_MHz_G * mI_vals[mi_i]
        for mi_i in range(dim_I)
        for mj_i in range(dim_J)
    ])

    # ── Build |F, m_F⟩ basis ordering at B=0 for labelling ─────────────────
    # Eigenvectors of H_hf at B=0 → identify (F, m_F) quantum numbers
    evals0, evecs0 = np.linalg.eigh(H_hf)
    # For each eigenvector find the dominant |m_I, m_J⟩ and infer F, m_F
    labels = []
    for k in range(dim):
        v      = evecs0[:, k]
        dom_r  = int(np.argmax(np.abs(v)**2))
        mi_i   = dom_r // dim_J
        mj_i   = dom_r %  dim_J
        mI     = mI_vals[mi_i]
        mJ     = mJ_vals[mj_i]
        mF_approx = round(2*(mI + mJ)) / 2   # m_F = m_I + m_J
        # Estimate F from energy: E ≈ A/2·K
        # For the ground state family, use closest F level energy
        Fs   = F_levels(I, J)
        best_F = min(Fs, key=lambda F_: abs(evals0[k] - hyperfine_energy(F_, I, J, A_MHz, B_MHz_quad)))
        labels.append((best_F, mF_approx))

    # ── Diagonalise at each B ────────────────────────────────────────────────
    all_energies = np.zeros((len(B_gauss_array), dim))

    for bi, B in enumerate(B_gauss_array):
        H = H_hf.copy()
        np.fill_diagonal(H, H.diagonal() + H_Z_diag * B)
        evals = np.linalg.eigvalsh(H)
        all_energies[bi] = evals

    return all_energies, labels


# ============================================================================
# CLOCK INTERVAL
# ============================================================================

def microwave_clock(state: HyperfineState, I: float) -> MicrowaveClock | None:
    """The F_max ↔ F_max − 1 interval of *state*, or None with a single F.

    Only A enters (P7): exact for the J = 1/2 alkali ground states.
    """
    Fs = [lv.F for lv in state.levels]
    if len(Fs) < 2:
        return None
    F1, F2 = Fs[-1], Fs[-2]
    dE = (hyperfine_energy(F1, I, state.J, state.A_MHz)
          - hyperfine_energy(F2, I, state.J, state.A_MHz))
    nu_MHz = abs(dE)
    return MicrowaveClock(
        F_upper=F1,
        F_lower=F2,
        frequency_MHz=nu_MHz,
        frequency_GHz=nu_MHz / 1e3,
        frequency_Hz=nu_MHz * 1e6,
        wavelength_m=3e8 / (nu_MHz * 1e6) if nu_MHz > 0 else None,
        wavelength_cm=3e4 / nu_MHz if nu_MHz > 0 else None,
        note=CLOCK_NOTE,
        state=state.label,
    )


# ============================================================================
# ONE STATE, ONE ISOTOPE
# ============================================================================

def _extrapolated(n, l_char, J, constants: Mapping[str, HfConstants],
                  quantum_defects: Mapping[str, float]):
    """(reference label, A, B, C) scaled from the first measured state with the
    same (l, J) as A ∝ n*⁻³, or None."""
    for ref_label, ref in constants.items():
        parsed = parse_state_label(ref_label)
        if parsed and parsed[1] == l_char and abs(parsed[2] - J) < 0.01 and ref.A_MHz != 0:
            delta_l = quantum_defects.get(l_char, 0.0)
            n_eff_ref = parsed[0] - delta_l
            n_eff_cur = n - delta_l
            if n_eff_ref > 0 and n_eff_cur > 0:
                scale = (n_eff_ref / n_eff_cur) ** 3
                return ref_label, ref.A_MHz * scale, ref.B_MHz * scale, ref.C_kHz * scale
            return None
    return None


def compute_state(label: str, nucleus: Nucleus, constants: Mapping[str, HfConstants],
                  quantum_defects: Mapping[str, float], B_G: np.ndarray) -> HyperfineState | None:
    """F levels, g factors and Breit-Rabi levels of one state; None for a bad label."""
    parsed = parse_state_label(label)
    if parsed is None:
        return None
    n, l_char, J = parsed
    I, gI = nucleus.I, nucleus.gI

    hfc = constants.get(label)
    gJ_exp = hfc.gJ_exp if hfc else None
    S = hfc.S_exp if hfc and hfc.S_exp is not None else 0.5   # electron spin
    gJ = lande_gJ(L_CHAR.get(l_char, 0), S, J, gJ_exp=gJ_exp)

    A_MHz, B_MHz, C_kHz = (hfc.A_MHz, hfc.B_MHz, hfc.C_kHz) if hfc else (0.0, 0.0, 0.0)
    source = None
    if A_MHz == 0.0:   # not measured — or a measured A = 0 (P17)
        found = _extrapolated(n, l_char, J, constants, quantum_defects)
        if found:
            source, A_MHz, B_MHz, C_kHz = found

    levels = [
        FLevel(F=F,
               E_MHz=hyperfine_energy(F, I, J, A_MHz, B_MHz, C_kHz),
               gF=gF_factor(F, I, J, gJ, gI),
               mF=[round((F-k)*2)/2 for k in range(round(2*F)+1)])
        for F in F_levels(I, J)
    ]
    BR_energies, BR_labels = breit_rabi_energies(I, J, A_MHz, B_MHz, gJ, gI, B_G)
    return HyperfineState(
        label=label, n=n, l=l_char, J=J,
        gJ=gJ,
        gJ_source='experimental' if gJ_exp is not None else 'Lande',
        A_MHz=A_MHz, B_MHz=B_MHz, C_kHz=C_kHz,
        extrapolated_from=source,
        levels=levels,
        BR_energies=BR_energies,
        BR_labels=BR_labels,
    )


def compute(
    nucleus: Nucleus, constants: Mapping[str, HfConstants], state_labels: Sequence[str],
    quantum_defects: Mapping[str, float], *, B_max: float = 500.0, B_points: int = 500,
) -> HyperfineResult:
    """Every state's hyperfine structure on 0..B_max G, plus the first state's clock."""
    B_G = np.linspace(0, B_max, B_points)
    states, skipped = [], []
    for label in state_labels:
        state = compute_state(label, nucleus, constants, quantum_defects, B_G)
        if state is None:
            skipped.append(label)
        else:
            states.append(state)
    clock = microwave_clock(states[0], nucleus.I) if states else None
    return HyperfineResult(nucleus=nucleus, B_G=B_G, states=states, clock=clock,
                           skipped=skipped)
