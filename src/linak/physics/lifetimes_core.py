"""Einstein A coefficients, radiative lifetimes and line parameters: the pure computation.

For every dipole transition the absorption oscillator strength f_lu comes from the
first source that has it:

  1. NIST (term + J for ions, J-resolved lines with LS 6j completion, or the multiplet
     resolved into fine structure with LS 6j weights), 1b. the curated literature
     overlay, 2. ORCA TD-DFT for ground-state lines matched within 0.10 eV, 3. ORCA
     EOM-CCSD excited → excited moments, 4. Numerov radial integrals calibrated on
     NIST, 5. the Coulomb approximation.

Metals use the published NIST gA / f of each line only. Lines below the Numerov gate
(n* ≤ 2.5) are recovered by extrapolating NIST series as f n*³ = a + b/n*².

  A_ul = (EINSTEIN_PREFACTOR / λ²) (g_l / g_u) f_lu,   τ = 1 / Σ A_ul

and, for each state's dominant channel, the Einstein B, natural and Doppler widths,
peak cross sections, saturation intensities, recoil, Doppler limit and capture velocity.

Known issues (PHYSICS_BUGS.md): the peak cross section uses 1/Δν (P22: σ ×π/2,
I_sat ×2/π); ``coulomb_fosc`` fails the hydrogen limit (P26). Kept as is so results
stay comparable until they are fixed deliberately.

No files or prints here: the ``lifetimes`` pipeline loads the transitions, the
NIST / literature / ORCA f-values and passes them in.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as _np

from linak.core.constants import AMU_KG, C_M_S, EINSTEIN_PREFACTOR, H_J_S, KB_J_K
from linak.physics.polarizability_core import ACC_UNCERTAINTY, parse_term_from_label
from linak.physics.qdt_radial import (
    L_CHAR,
    allowed_J2,
    build_true_multiplet_f,
    fit_numerov_scales_by_type,
    ls_group_branching,
    ls_line_weight,
    numerov_fosc_scaled,
    numerov_scale_for,
    parse_J2_set,
)

# Reference temperature for Doppler width (typical vapour cell)
T_DOPPLER_K = 300.0   # K

_EXTRAP_K = 6
_EXTRAP_MAX_RMS = 0.05


# ============================================================================
# RECORDS
# ============================================================================

@dataclass(frozen=True)
class FValueSources:
    """Everything the oscillator-strength priority list can draw on."""

    nist: Mapping                   # polarizability_core.load_nist_fvalues(...)
    literature: Mapping             # polarizability_core.load_literature_fvalues(...)
    # ORCA ground → excited f, summed per excitation energy (eV, 4 decimals)
    orca_fosc: Mapping[float, float] = field(default_factory=dict)
    # ORCA EOM-CCSD excited → excited f by (from, to) state number (1-based)
    ee_fosc: Mapping[tuple[int, int], float] = field(default_factory=dict)
    orca_excitations: Sequence[Mapping] = ()


@dataclass(frozen=True)
class TransitionA:
    """One line with an Einstein A (a row of ``transitions`` in the lifetimes JSON)."""

    upper_label: str
    lower_label: str
    upper_n: int | None
    lower_n: int | None
    upper_l: str | None
    lower_l: str | None
    wavelength_nm: float
    delta_E_eV: float
    region: str
    fosc: float
    fosc_source: str
    fosc_acc: str | None     # NIST Acc string or None
    fosc_delta: float        # fractional 1-sigma on fosc
    g_upper: float
    g_lower: float
    A_s: float
    sigma_A_s: float


@dataclass(frozen=True)
class Dropped:
    """A transition that got no A, and why."""

    transition: Mapping
    reason: str


@dataclass(frozen=True)
class SeriesFit:
    """A NIST series fit f n*³ = a + b/n*² used to extrapolate below the Numerov gate."""

    n_fit_max: int
    n_points: int
    a: float
    b: float
    rms: float


@dataclass(frozen=True)
class Lifetime:
    """One upper state: τ, its decay channels, and its dominant channel's line parameters."""

    state: str
    n: int | None
    l: str | None
    A_total_s: float
    sigma_A_s: float
    lifetime_s: float
    sigma_tau_s: float
    quality_acc: str | None    # accuracy of the least accurate channel above 5 % of A
    quality_delta: float
    channels: list[TransitionA]   # strongest first
    # The dominant (strongest) channel:
    B21_m3_J_s2: float
    B12_m3_J_s2: float
    delta_nu_nat_Hz: float
    sigma_peak_nat_m2: float
    I_sat_nat_W_m2: float
    mass_amu: float
    delta_nu_dop_Hz: float     # at T_DOPPLER_K
    sigma_peak_dop_m2: float
    I_sat_dop_W_m2: float
    v_rec_ms: float
    E_rec_J: float
    E_rec_nK: float
    T_dop_uK: float
    v_cap_ms: float

    @property
    def dominant(self) -> TransitionA:
        return self.channels[0]


@dataclass(frozen=True)
class LifetimesResult:
    numerov_calibration: dict | None      # None on the metal path
    transitions: list[TransitionA]        # input order, then the extrapolated lines
    dropped: list[Dropped]
    n_extrapolated: int
    series_fits: dict[str, SeriesFit]     # '<lower>-><l><J>'
    lifetimes: list[Lifetime]


@dataclass(frozen=True)
class _Context:
    sources: FValueSources
    qd: dict[str, float]
    z_core: float
    IE: float
    gs_label: str
    is_ion: bool
    is_metal: bool
    orca_only: bool
    nist_only: bool
    nist_true: Mapping                  # multiplet f rebuilt from the NIST lines
    numerov_cal: dict | None


# ============================================================================
# OSCILLATOR-STRENGTH HELPERS
# ============================================================================

def coulomb_fosc(n_upper, l_upper, n_lower, l_lower, delta_upper, delta_lower):
    """
    Coulomb approximation for oscillator strength (Bates & Damgaard 1949).

    For a transition n_lower l_lower -> n_upper l_upper with quantum defects:
        n*_u = n_upper - delta_upper
        n*_l = n_lower - delta_lower

    The absorption oscillator strength scales as:
        f_lu ∝ n*_l^2 * n*_u^2 / (n*_u^2 - n*_l^2)^3

    We use the semi-empirical formula for alkali-like transitions:
        f_lu = C_l * (n*_l * n*_u)^3 / (n*_u^2 - n*_l^2)^3

    where C_l accounts for the angular momentum factor and is derived
    from the known hydrogenic limit.

    Angular factors for Δl = ±1 (only allowed E1):
        l_max = max(l_upper, l_lower)
        C = (2/3) * l_max / (2*l_lower + 1)

    Reference: Cowan, "The Theory of Atomic Structure and Spectra" (1981)
    """
    n_eff_u = n_upper  - delta_upper
    n_eff_l = n_lower  - delta_lower

    if n_eff_u <= n_eff_l or n_eff_u <= 0 or n_eff_l <= 0:
        return 0.0

    # Guard against near-degenerate n_eff (same-n transitions like 4f->4d)
    # When n_eff_u ≈ n_eff_l the denominator (n*_u² - n*_l²) → 0 and the
    # Coulomb approximation diverges. The actual matrix element for same-n
    # transitions is small and can't be computed this way — return 0.
    if (n_eff_u**2 - n_eff_l**2) < 0.5 * n_eff_u:
        return 0.0

    l_max = max(l_upper, l_lower)

    # Angular momentum factor
    C_ang = (2.0/3.0) * l_max / (2*l_lower + 1)

    # Energy difference in Rydberg units
    delta_E_ry = 1.0/n_eff_l**2 - 1.0/n_eff_u**2

    # Radial matrix element squared in atomic units (Coulomb approximation)
    # |<n*_l l | r | n*_u l'>|^2 ≈ (n*_u * n*_l)^3 / (n*_u^2 - n*_l^2)^3
    # × correction for non-integer n* (slowly varying, ~1 for high n*)
    n_prod = n_eff_u * n_eff_l
    n_diff_sq = n_eff_u**2 - n_eff_l**2
    R_sq = n_prod**3 / n_diff_sq**3

    # f = (g_upper / g_lower) * (4/3) * delta_E_ry * C_ang * R_sq
    # Note: this is f_lu (absorption), g ratio already in C_ang
    f = (4.0/3.0) * delta_E_ry * C_ang * R_sq * (2*l_upper + 1) / (2*l_lower + 1)

    return max(f, 0.0)


def resolve_multiplet_f(f_mult, lu, ll, J_up, J_lo, wl):
    """
    Multiplet f (nl -> n'l', lower-multiplet averaged) -> (fosc, g_upper, g_lower) for
    the channel upper-group -> lower-group, using LS 6j line strengths. A level label
    without J is an unresolved multiplet averaged statistically over its fine-structure
    levels (so no double counting); returns None if the J combination is forbidden.
        A(Ju -> Jl) = A_mult (2Lu+1)(2Jl+1){Lu Ju 1/2; Jl Ll 1}^2,
        A_mult = PREF/lambda^2 (2Ll+1)/(2Lu+1) f_mult
    """
    ratio, gU, gL = ls_group_branching(lu, parse_J2_set(J_up, lu), ll, parse_J2_set(J_lo, ll))
    if ratio <= 1e-12:
        return None
    A_mult = (EINSTEIN_PREFACTOR / wl**2) * (2*ll + 1) / (2*lu + 1) * f_mult
    return A_mult * ratio * wl**2 / EINSTEIN_PREFACTOR * gU / gL, gU, gL


def nist_line_f(lo_nl, up_nl, lu, J_up, ll, J_lo, fosc_jl):
    """
    f of one line (L_l, J_l) -> (L_u, J_u): the NIST line itself if listed, else the
    value implied by another listed lower-J line of the same upper level through the
    LS 6j ratio  f ~ w(Ju,Jl)/(2Jl+1).  Returns (f, acc, reconstructed) or None when
    not applicable (lower s level, merged J labels, ions, no line data).
    """
    if lu < 0 or ll < 1 or not J_up or not J_lo or ',' in str(J_up) or ',' in str(J_lo):
        return None
    ju = parse_J2_set(J_up, lu)
    jl = parse_J2_set(J_lo, ll)
    if len(ju) != 1 or len(jl) != 1:
        return None
    ju, jl = ju[0], jl[0]
    w_this = ls_line_weight(lu, ju, ll, jl)
    if w_this <= 1e-12:
        return (0.0, None, False)
    hit = fosc_jl.get((lo_nl, up_nl, J_up, J_lo))
    if hit:
        return (hit[0], hit[1], False)
    for j2 in allowed_J2(ll):
        if j2 == jl:
            continue
        w_k = ls_line_weight(lu, ju, ll, j2)
        if w_k <= 1e-12:
            continue
        Jk = f"{j2}/2"
        other = fosc_jl.get((lo_nl, up_nl, J_up, Jk))
        if other:
            f = other[0] * (w_this / (jl + 1)) / (w_k / (j2 + 1))
            return (f, other[1], True)
    return None


def degeneracy(l):
    """Degeneracy of orbital l ignoring spin-orbit: 2l+1."""
    return 2 * l + 1


def parse_J_label(label):
    """Extract J from '3p1/2' or ion multiplet key '3s_1Po_1'."""
    if not label:
        return None
    m = re.search(r'(\d+/\d+)$', label)
    if m:
        return m.group(1)
    m = re.search(r'_(\d+h\d+|\d+)$', str(label))
    return m.group(1) if m else None


def J_degeneracy(J_str):
    """Statistical weight g = 2J+1 from a J string like '1/2'."""
    if not J_str:
        return None
    try:
        if '/' in J_str:
            Jn, Jd = (int(x) for x in J_str.split('/'))
            return 2 * Jn // Jd + 1
        return 2 * int(J_str) + 1
    except (ValueError, ZeroDivisionError):
        return None


def default_lower_J(l_char):
    """Default J when the lower label has no J suffix (alkali ns ground → ²S₁/₂)."""
    return '1/2' if l_char == 's' else None


def parse_state_label(label):
    """Parse state label like '3p', '10d', '13s' into (n, l) for sorting."""
    digits = ''.join(filter(str.isdigit, label))
    letters = ''.join(filter(str.isalpha, label))
    return (int(digits) if digits else 0, letters)


def calibration_fvalues(nist_fosc, nist_true):
    """The NIST multiplet f the Numerov scale is fitted on.

    The loader's multiplet f is 1.45-1.48x too large whenever BOTH levels have two J
    (p->d, d->f, d->p, f->d) and exact for s->p and p->s, so it is kept only for pairs
    involving an s level; the rest comes from the multiplets rebuilt from the lines.
    """
    cal = dict(nist_true)
    for k, v in (nist_fosc or {}).items():
        if k in cal or not isinstance(k, tuple):
            continue
        a = re.match(r'^\d+([spdfg])$', k[0])
        b = re.match(r'^\d+([spdfg])$', k[1])
        if a and b and 's' in (a.group(1), b.group(1)):
            cal[k] = v
    return cal


# ============================================================================
# ONE TRANSITION
# ============================================================================

def _row(t, fosc, fosc_source, fosc_acc, fosc_delta, g_upper, g_lower, A, sigma_A):
    return TransitionA(
        upper_label=t['upper_label'], lower_label=t['lower_label'],
        upper_n=t['upper_n'], lower_n=t['lower_n'],
        upper_l=t['upper_l'], lower_l=t['lower_l'],
        wavelength_nm=t['wavelength_nm'], delta_E_eV=t['delta_E_eV'],
        region=t['region'],
        fosc=fosc, fosc_source=fosc_source, fosc_acc=fosc_acc, fosc_delta=fosc_delta,
        g_upper=g_upper, g_lower=g_lower,
        A_s=A, sigma_A_s=sigma_A,
    )


def _metal_row(t, c: _Context) -> TransitionA | None:
    """Metal path: the published NIST gA / f of the line only (no Numerov / Coulomb)."""
    nist = c.sources.nist
    nist_fosc_line = nist.get('fosc_line') or {}
    nist_acc_line = nist.get('acc_line') or {}
    nist_gA_line = nist.get('gA_line') or {}

    wl = t['wavelength_nm']
    lu = L_CHAR.get(t['upper_l'], -1) if t['upper_l'] else -1
    ll_int = L_CHAR.get(t['lower_l'], -1) if t['lower_l'] else -1
    g_upper = degeneracy(lu) if lu >= 0 else 1
    g_lower = degeneracy(ll_int) if ll_int >= 0 else 1
    _gJ_u = J_degeneracy(t.get('upper_J'))
    _gJ_l = J_degeneracy(t.get('lower_J'))
    if _gJ_u:
        g_upper = _gJ_u
    if _gJ_l:
        g_lower = _gJ_l
    _key = (t['lower_label'], t['upper_label'])
    _gA = t.get('gA')
    if _gA is None:
        _gA = nist_gA_line.get(_key)
    _fik = t.get('fik')
    _f_line = t.get('fosc') or nist_fosc_line.get(_key)
    _acc = t.get('Acc') or nist_acc_line.get(_key, 'unknown')
    _delta = ACC_UNCERTAINTY.get(_acc, 0.50) if _acc else 0.50
    if _gA and _gA > 0 and g_upper:
        A = float(_gA) / g_upper
        if _f_line and _f_line > 0:
            fosc = float(_f_line)
        else:
            # f_lu from A: A = PREF/λ² * (g_l/g_u) * f
            fosc = A * (wl ** 2) / EINSTEIN_PREFACTOR * (g_upper / max(g_lower, 1))
    elif _fik and _fik > 0:
        fosc = float(_fik)
        A = EINSTEIN_PREFACTOR / (wl ** 2) * (g_lower / g_upper) * fosc
    elif _f_line and _f_line > 0:
        fosc = float(_f_line)
        A = EINSTEIN_PREFACTOR / (wl ** 2) * (g_lower / g_upper) * fosc
    else:
        return None
    if A <= 0 or fosc <= 0:
        return None
    return _row(t, fosc, 'NIST', _acc, _delta, g_upper, g_lower, A, A * (_delta or 0.50))


def _line_row(t, c: _Context) -> TransitionA | str:
    """The line with its A, or the reason it gets none."""
    nist = c.sources.nist
    nist_fosc = nist['fosc']
    nist_fosc_j = nist['fosc_j']
    nist_fosc_jt = nist.get('fosc_jt') or {}
    nist_acc = nist['acc']
    nist_acc_j = nist['acc_j']
    nist_acc_jt = nist.get('acc_jt') or {}
    nist_fosc_jl = nist.get('fosc_jl') or {}
    nist_fosc_true = c.nist_true
    literature_fosc = c.sources.literature['fosc']
    literature_fosc_j = c.sources.literature['fosc_j']
    literature_acc_j = c.sources.literature['acc_j']
    orca_fosc = c.sources.orca_fosc
    ee_fosc = c.sources.ee_fosc
    qd = c.qd
    gs_label = c.gs_label
    _is_ion = c.is_ion

    _drop_reason = None
    wl  = t['wavelength_nm']
    dE  = t['delta_E_eV']
    ul  = t['upper_label']
    ll  = t['lower_label']
    nu  = t['upper_n']
    nl  = t['lower_n']
    lu_char = t['upper_l']
    ll_char = t['lower_l']
    lu  = L_CHAR.get(lu_char, -1) if lu_char else -1
    ll_int = L_CHAR.get(ll_char, -1) if ll_char else -1

    g_upper = degeneracy(lu) if lu >= 0 else 1
    g_lower = degeneracy(ll_int) if ll_int >= 0 else 1

    fosc        = None
    fosc_source = None
    fosc_acc    = None    # NIST Acc string or None
    fosc_delta  = None    # fractional 1-sigma uncertainty

    # Priority 1: NIST experimental f-values
    # Prefer J-resolved lookup when the transition label carries J (e.g. 3p3/2).
    # Ions: term+J first so ¹P₁ / ³P₁ do not share one f.
    # Fall back to nl-averaged multiplet f_eff for states without J labels.
    _ll_nl = re.sub(r'(\d+[spdfg]).*', r'\1', ll)
    _ul_nl = re.sub(r'(\d+[spdfg]).*', r'\1', ul)
    _J_up  = t.get('upper_J') or parse_J_label(ul)
    _J_lo  = t.get('lower_J') or parse_J_label(ll) or default_lower_J(ll_char)
    _term_up = parse_term_from_label(ul) if _is_ion else None
    _term_lo = parse_term_from_label(ll) if _is_ion else None
    if (nist_fosc_jt or nist_fosc_j or nist_fosc) and not c.orca_only:
        _f = None
        if _is_ion and nist_fosc_jt and _J_up and _term_up:
            # Exact (term_lo, term_up) key; if lower has no term (e.g. '2p'), unique-match on (nl,nl,*,term_up,J).
            _jtkey = (_ll_nl, _ul_nl, _term_lo or '', _term_up, _J_up)
            _f = nist_fosc_jt.get(_jtkey)
            _hit_key = _jtkey if _f else None
            if not _f and not _term_lo:
                _cands = [k for k in nist_fosc_jt
                          if k[0] == _ll_nl and k[1] == _ul_nl
                          and k[3] == _term_up and k[4] == _J_up]
                if len(_cands) == 1:
                    _hit_key = _cands[0]
                    _f = nist_fosc_jt[_hit_key]
            if _f and _f > 0:
                fosc        = _f
                fosc_acc    = nist_acc_jt.get(_hit_key, 'unknown')
                fosc_source = 'NIST'
                fosc_delta  = ACC_UNCERTAINTY.get(fosc_acc, 0.50)
                _gJ_u = J_degeneracy(_J_up)
                _gJ_l = J_degeneracy(_J_lo)
                if _gJ_u:
                    g_upper = _gJ_u
                if _gJ_l:
                    g_lower = _gJ_l
        # Skip plain J lookup for ion multiplet labels, same J merges ¹P/³P.
        if fosc is None and _J_up and nist_fosc_j and not _term_up:
            _jkey = (_ll_nl, _ul_nl, _J_up)
            _f = nist_fosc_j.get(_jkey)
            if _f and _f > 0:
                fosc        = _f
                fosc_acc    = nist_acc_j.get(_jkey, 'unknown')
                fosc_source = 'precision' if fosc_acc == 'precision' else 'NIST'
                fosc_delta  = (0.01 if fosc_acc == 'precision'
                               else ACC_UNCERTAINTY.get(fosc_acc, 0.50))
                # Line-level NIST f + LS 6j completion over the lower fine-structure levels.
                if not _is_ion and fosc_source == 'NIST':
                    _fl = nist_line_f(_ll_nl, _ul_nl, lu, _J_up, ll_int, _J_lo, nist_fosc_jl)
                    if _fl is not None:
                        _f_line, _acc_line, _recon = _fl
                        fosc = _f_line
                        if _acc_line:
                            fosc_acc = _acc_line
                        fosc_delta = ACC_UNCERTAINTY.get(fosc_acc, 0.50)
                        if _recon:   # completed with the LS 6j ratio: add 5 % model error
                            fosc_delta = (fosc_delta**2 + 0.05**2) ** 0.5
                        if _f_line <= 0:
                            _drop_reason = 'forbidden_dJ (|J_u-J_l|>1)'
                _gJ_u = J_degeneracy(_J_up)
                _gJ_l = J_degeneracy(_J_lo)
                if _gJ_u:
                    g_upper = _gJ_u
                if _gJ_l:
                    g_lower = _gJ_l
        # Multiplet nl average also merges ion terms, skip when term is known.
        if fosc is None and nist_fosc and not _term_up:
            _key  = (_ll_nl, _ul_nl)
            _key2 = (_ul_nl, _ll_nl)
            _f    = nist_fosc.get(_key) or nist_fosc.get(_key2)
            _k    = _key if _key in nist_fosc else _key2
            _resolved = None
            if _f and _f > 0 and not _is_ion and lu >= 0 and ll_int >= 0:
                _ft = nist_fosc_true.get(_key) or nist_fosc_true.get(_key2)
                if _ft:
                    _f = _ft
                elif lu >= 1 and ll_int >= 1:
                    _f = None                      # overcounted multiplet, cannot be corrected
                if _f and _f > 0:
                    _resolved = resolve_multiplet_f(_f, lu, ll_int, _J_up, _J_lo, wl)
                    if _resolved is None:
                        _f = None
                        _drop_reason = 'forbidden_dJ (|J_u-J_l|>1 within the LS multiplet)'
            if _f and _f > 0:
                fosc        = _f
                fosc_source = 'NIST'
                fosc_acc    = nist_acc.get(_k, 'unknown')
                fosc_delta  = (0.01 if fosc_acc == 'precision'
                               else ACC_UNCERTAINTY.get(fosc_acc, 0.50))
                if _resolved is not None:
                    fosc, g_upper, g_lower = _resolved

    # Priority 1b: curated literature overlay
    if fosc is None and (literature_fosc_j or literature_fosc) and not c.orca_only:
        if _J_up and literature_fosc_j:
            _jkey = (_ll_nl, _ul_nl, _J_up)
            _f = literature_fosc_j.get(_jkey)
            if _f and _f > 0:
                fosc        = _f
                fosc_source = 'literature'
                fosc_acc    = literature_acc_j.get(_jkey, 'literature')
                fosc_delta  = ACC_UNCERTAINTY.get(fosc_acc, 0.20)
                _gJ_u = J_degeneracy(_J_up)
                _gJ_l = J_degeneracy(_J_lo)
                if _gJ_u:
                    g_upper = _gJ_u
                if _gJ_l:
                    g_lower = _gJ_l
        if fosc is None and literature_fosc:
            _key  = (_ll_nl, _ul_nl)
            _key2 = (_ul_nl, _ll_nl)
            _f    = literature_fosc.get(_key) or literature_fosc.get(_key2)
            if _f and _f > 0:
                fosc        = _f
                fosc_source = 'literature'
                fosc_acc    = 'literature'
                fosc_delta  = 0.20

    # Priority 2: ORCA TD-DFT for ground state transitions
    # Match ONLY when a TD-DFT root lies close in energy to the spectroscopic level.
    # A fractional window that grows toward IE (old: up to 0.5 eV) wrongly
    # assigns high-n Rydberg np→gs lines to the highest valence/continuum-like
    # ORCA root, freezing their lifetime near ~0.4 us. Absolute 0.10 eV covers
    # typical CAM-B3LYP valence shifts (Na 3p ~0.07 eV, 4p ~0.08 eV) while
    # rejecting Rydberg latch-on. Roots within 0.25 eV of IE are discarded
    # standard bases do not describe those diffuse states.
    if fosc is None and not c.nist_only and ll == gs_label and orca_fosc:
        exc_ev = dE
        tolerance = 0.10  # eV, absolute
        continuum_cut = c.IE - 0.25
        candidates = [e for e in orca_fosc.keys() if e <= continuum_cut]
        best_match = min(candidates, key=lambda x: abs(x - exc_ev),
                         default=None) if candidates else None
        if best_match is not None and abs(best_match - exc_ev) < tolerance:
            fosc        = orca_fosc[best_match]
            fosc_source = 'ORCA'
            fosc_delta  = 0.05   # ~5% fosc accuracy for CAM-B3LYP

    # EOM excited->excited fosc (EOM-CCSD only)
    # Match upper and lower states by energy to their ORCA state indices, then look up
    # the excited->excited transition moment.
    if fosc is None and ee_fosc and ll != gs_label:
        upper_E = t['upper_energy_eV'] + c.IE    # convert abs to excitation eV
        lower_E = t['lower_energy_eV'] + c.IE

        # Find ORCA state indices closest in energy to upper and lower levels
        tol = max(0.1, min(0.4, 0.15 * abs(upper_E)))
        upper_idx = None
        lower_idx = None
        for idx, ex in enumerate(c.sources.orca_excitations):
            if abs(ex['energy_eV'] - upper_E) < tol:
                upper_idx = idx + 1   # 1-indexed
            if abs(ex['energy_eV'] - lower_E) < tol:
                lower_idx = idx + 1

        if upper_idx and lower_idx:
            f_ee = ee_fosc.get((upper_idx, lower_idx)) or \
                   ee_fosc.get((lower_idx, upper_idx))
            if f_ee and f_ee > 0:
                fosc        = f_ee
                fosc_source = 'EOM-CCSD'
                fosc_delta  = 0.05

    # QD Numerov radial (preferred theoretical fallback; alkalis only)
    # The Numerov f is a MULTIPLET (LS-averaged) absorption f for nl -> n'l'.
    # It is resolved into fine structure with LS 6j line strengths:
    #   A(Ju -> Jl) = A_mult * (2Lu+1)(2Jl+1){Lu Ju 1/2; Jl Ll 1}^2
    # where A_mult = PREF/lambda^2 * (2Ll+1)/(2Lu+1) * f is the rate of any one
    # upper level of the multiplet to the whole lower multiplet. A level label
    # without J (ns, 21p, 21d, ...) is an unresolved multiplet and is averaged
    # statistically over its fine-structure levels (no double counting).
    if fosc is None and not c.orca_only:
        delta_u = qd.get(lu_char, 0.0)
        delta_l = qd.get(ll_char, 0.0)
        n_eff_u = nu - delta_u
        n_eff_l = nl - delta_l
        f_num = 0.0
        if n_eff_u > 2.5 and n_eff_l > 2.5 and lu >= 0 and ll_int >= 0:
            try:
                f_num = numerov_fosc_scaled(
                    nu, lu, nl, ll_int, delta_u, delta_l,
                    scale=numerov_scale_for(c.numerov_cal or {}, ll_char, lu_char),
                    z_core=c.z_core)
            except Exception:
                f_num = 0.0
        if f_num and f_num > 0:
            fosc_source = 'Numerov'
            fosc_delta  = 0.15 if (n_eff_u > 4 and n_eff_l > 4) else 0.25
            _res = resolve_multiplet_f(f_num, lu, ll_int, _J_up, _J_lo, wl)
            if _res is None:
                fosc = None
                _drop_reason = 'forbidden_dJ (|J_u-J_l|>1 within the LS multiplet)'
            else:
                fosc, g_upper, g_lower = _res

    # Coulomb approximation (last resort; alkalis only)
    if fosc is None and _drop_reason is None and not c.orca_only:
        delta_u = qd.get(lu_char, 0.0)
        delta_l = qd.get(ll_char, 0.0)
        n_eff_u = nu - delta_u
        n_eff_l = nl - delta_l
        if n_eff_u > 4 and n_eff_l > 4:
            fosc        = coulomb_fosc(nu, lu, nl, ll_int, delta_u, delta_l)
            fosc_source = 'Coulomb'
            fosc_delta  = 0.30
        elif n_eff_u > 2.5 and n_eff_l > 2.5 and ll != gs_label:
            fosc        = coulomb_fosc(nu, lu, nl, ll_int, delta_u, delta_l)
            fosc_source = 'Coulomb(approx)'
            fosc_delta  = 0.50
        else:
            fosc        = 0.0
            fosc_source = 'unknown'
            fosc_delta  = 1.00

    if fosc is None or fosc <= 0:
        if _drop_reason is None:
            _du = qd.get(lu_char, 0.0) if lu_char else 0.0
            _dl = qd.get(ll_char, 0.0) if ll_char else 0.0
            _nsu, _nsl = nu - _du, nl - _dl
            if lu < 0 or ll_int < 0 or abs(lu - ll_int) != 1:
                _drop_reason = 'not_E1_or_unlabelled'
            elif not (_nsu > 2.5 and _nsl > 2.5):
                _drop_reason = 'below_numerov_gate (n*<=2.5, no NIST f)'
            elif (_nsu**2 - _nsl**2) < 0.5 * _nsu:
                _drop_reason = 'near_degenerate_guard (Coulomb/Numerov)'
            else:
                _drop_reason = 'f<=0 or radial integral failed'
        return _drop_reason

    # A_ul = PREFACTOR / lambda_nm^2 * (g_l/g_u) * f_lu
    A       = EINSTEIN_PREFACTOR / (wl**2) * (g_lower / g_upper) * fosc
    sigma_A = A * (fosc_delta or 0.50)   # 1-sigma uncertainty on A
    return _row(t, fosc, fosc_source, fosc_acc, fosc_delta, g_upper, g_lower, A, sigma_A)


# ============================================================================
# NIST SERIES EXTRAPOLATION BELOW THE NUMEROV GATE
# ============================================================================

def _split_J_tag(label, n):
    m = re.match(rf'^{n}[spdfg](\d+/\d+)$', label or '')
    return m.group(1) if m else ''


def _build_extrap_fits(lowers, transition_table: Sequence[TransitionA], qd):
    """Fit y = f n*^3 = a + b/n*^2 for every (lower_label, upper_l, J_tag) NIST series."""
    series: dict = defaultdict(dict)
    for tt in transition_table:
        if tt.fosc_source not in ('NIST', 'precision') or tt.lower_label not in lowers:
            continue
        lc = tt.upper_l
        if lc not in qd or tt.upper_n is None:
            continue
        Jt = _split_J_tag(tt.upper_label, tt.upper_n) if lc != 's' else ''
        series[(tt.lower_label, lc, Jt)][tt.upper_n] = (tt.fosc, tt.fosc_delta or 0.25)
    fits = {}
    for key, d in series.items():
        ns_ = sorted(d)
        if len(ns_) < _EXTRAP_K:
            continue
        ns_ = ns_[-_EXTRAP_K:]
        x = _np.array([n - qd[key[1]] for n in ns_])
        y = _np.array([d[n][0] * (n - qd[key[1]])**3 for n in ns_])
        M = _np.vstack([_np.ones_like(x), 1.0 / x**2]).T
        c = _np.linalg.lstsq(M, y, rcond=None)[0]
        rms = float(_np.sqrt(_np.mean((M @ c / y - 1.0)**2)))
        if rms < _EXTRAP_MAX_RMS:
            fits[key] = dict(c=c, n_max=max(ns_), rms=rms, delta=d[ns_[-1]][1],
                             n_pts=len(ns_))
    return fits


def _extrapolated_row(tt, fits, qd) -> TransitionA | None:
    """The A of a below-gate line from its lower level's NIST series fits, or None."""
    ul_, ll_, n_, lc_ = tt['upper_label'], tt['lower_label'], tt['upper_n'], tt['upper_l']
    wl_ = tt['wavelength_nm']
    lu_ = L_CHAR.get(lc_, -1)
    if lu_ < 0 or lc_ not in qd:
        return None
    Jtag = _split_J_tag(ul_, n_) if lc_ != 's' else ''
    if lc_ == 's':
        comps = [('', 2)]
    elif Jtag:
        comps = [(Jtag, J_degeneracy(Jtag))]
    else:
        comps = [(f'{j}/2', j + 1) for j in allowed_J2(lu_)]
    _J_lo_ = tt.get('lower_J') or parse_J_label(ll_) or default_lower_J(tt['lower_l'])
    g_l_ = J_degeneracy(_J_lo_) or (2 * L_CHAR.get(tt['lower_l'], 0) + 1)
    ns_u = n_ - qd[lc_]
    A_sum, used = 0.0, []
    for Jc, gu_c in comps:
        fit_key = (ll_, lc_, Jc)
        fit = fits.get(fit_key)
        if fit is None or n_ <= fit['n_max']:
            continue
        y = fit['c'][0] + fit['c'][1] / ns_u**2
        if y <= 0:
            continue
        f_c = y / ns_u**3
        A_c = EINSTEIN_PREFACTOR / wl_**2 * (g_l_ / gu_c) * f_c
        w_c = gu_c if not Jtag and lc_ != 's' else 1.0
        A_sum += A_c * w_c
        used.append((Jc, fit))
    if not used:
        return None
    if not Jtag and lc_ != 's':
        g_up = 2 * (2 * lu_ + 1)
        A_ch = A_sum / g_up
    else:
        g_up = comps[0][1]
        A_ch = A_sum
    f_eff = A_ch * wl_**2 / EINSTEIN_PREFACTOR * g_up / g_l_
    rms = max(fit['rms'] for _, fit in used)
    dnist = max(fit['delta'] for _, fit in used)
    d_ex = (dnist**2 + rms**2)**0.5
    acc = f"extrap(n>{max(f['n_max'] for _, f in used)}, rms {100*rms:.1f}%)"
    return _row(tt, f_eff, 'NIST-extrap', acc, d_ex, g_up, g_l_, A_ch, A_ch * d_ex)


def extrapolate_below_gate(rows: Sequence[TransitionA], dropped: Sequence[Dropped], qd):
    """(new rows, the drops still without an A, the series fits) for the lines dropped
    below the Numerov gate, from the NIST series of their lower levels."""
    gate = [d for d in dropped if d.reason.startswith('below_numerov_gate')]
    if not gate:
        return [], list(dropped), {}
    fits = _build_extrap_fits({d.transition['lower_label'] for d in gate}, rows, qd)
    new_rows, still_dropped = [], []
    for d in dropped:
        row = (_extrapolated_row(d.transition, fits, qd)
               if d.reason.startswith('below_numerov_gate') else None)
        if row is None:
            still_dropped.append(d)
        else:
            new_rows.append(row)
    series = {f"{lo_}->{lc_}{Jc}": SeriesFit(n_fit_max=int(fit['n_max']), n_points=fit['n_pts'],
                                             a=float(fit['c'][0]), b=float(fit['c'][1]),
                                             rms=fit['rms'])
              for (lo_, lc_, Jc), fit in fits.items()}
    return new_rows, still_dropped, series


# ============================================================================
# LIFETIMES AND LINE PARAMETERS
# ============================================================================

def lifetime_of(state_label, tlist: Sequence[TransitionA], mass_amu: float) -> Lifetime | None:
    """τ = 1/ΣA of one upper state and the line parameters of its dominant channel."""
    A_total     = sum(t.A_s for t in tlist)
    var_A_total = sum(t.sigma_A_s**2 for t in tlist)   # sum of variances
    if A_total <= 0:
        return None

    tau_s   = 1.0 / A_total
    sigma_A = var_A_total**0.5
    # σ(τ) = τ² × σ(A)   [error propagation: τ=1/A, dτ/dA=-1/A²]
    sigma_tau_s = sigma_A / A_total**2

    dominant = max(tlist, key=lambda t: t.A_s)

    # Overall quality grade: worst (largest delta) among channels weighted by A fraction
    # Channels contributing >5% of total A drive the uncertainty
    quality_acc   = None
    quality_delta = 0.0
    for ch in tlist:
        if ch.A_s / A_total > 0.05:
            d = ch.fosc_delta
            if d > quality_delta:
                quality_delta = d
                quality_acc   = ch.fosc_acc or ch.fosc_source

    # Einstein B coefficients
    # Computed for the DOMINANT decay channel (highest A).
    # B₂₁ = A₂₁ · c³ / (8πhν³)          stimulated emission [m³ J⁻¹ s⁻²]
    # B₁₂ = (g₂/g₁) · B₂₁                absorption
    # Both in SI: spectral energy density ρ(ν) units [J m⁻³ Hz⁻¹]
    #
    # Peak absorption cross-section at line centre (natural broadening):
    #   σ_peak = (λ²/8π) · (g₂/g₁) · A₂₁ / (Δν_nat)
    #   Δν_nat = A_total / (2π)             [Hz]  natural linewidth (FWHM)
    #
    # Saturation intensity (two-level approximation):
    #   I_sat = hν · A₂₁² · (1/2) / (σ_peak · A_total)
    #   Simplified for a closed two-level: I_sat = hν/(2·σ_peak·τ)  [W m⁻²]
    #   (factor of 2 for equal degeneracies; exact prefactor depends on g ratio)

    dom_A    = dominant.A_s                 # s⁻¹
    dom_wl_m = dominant.wavelength_nm * 1e-9   # m
    dom_nu   = C_M_S / dom_wl_m          # Hz
    dom_E_J  = H_J_S * dom_nu            # J  (photon energy)
    g_upper  = dominant.g_upper
    g_lower  = dominant.g_lower

    # B₂₁  [m³ J⁻¹ s⁻²]
    B21_SI   = dom_A * C_M_S**3 / (8 * math.pi * H_J_S * dom_nu**3)

    # B₁₂  [m³ J⁻¹ s⁻²]
    B12_SI   = (g_upper / g_lower) * B21_SI

    # Natural linewidth Δν [Hz] = A_total / (2π)
    delta_nu_nat = A_total / (2 * math.pi)

    # Peak cross-section at line centre [m²] — natural broadening (Lorentzian)
    sigma_peak_m2 = (dom_wl_m**2 / (8 * math.pi)) * (g_upper / g_lower) * (dom_A / delta_nu_nat)

    # Saturation intensity [W m⁻²] — two-level closed transition approximation
    # I_sat = hν / (2 · σ_peak · τ)
    I_sat_Wm2  = dom_E_J / (2.0 * sigma_peak_m2 * tau_s)

    # Doppler-broadened quantities
    # The Doppler FWHM is:
    #   Δν_D = (ν₀/c) · √(8·k_B·T·ln2 / m)
    # Peak cross-section with Doppler broadening (CompTable / compare_elements):
    #   σ_peak_dop = σ_peak_nat · (Δν_nat / Δν_D) · √(ln2/π)
    # with σ_peak_nat from the pipeline natural convention above.
    # Note: a unit-normalised Gaussian lineshape peak is larger by a factor 2
    # (φ(0)=2√(ln2/π)/Δν_D rather than √(ln2/π)/Δν_D); the pipeline keeps the
    # CompTable scaling so lifetimes JSON and comparison_table.html agree.
    #
    # Saturation intensity with Doppler broadening (same formula, different σ):
    #   I_sat_dop = hν / (2 · σ_peak_dop · τ)
    # This is always LARGER than I_sat_nat because σ_peak_dop < σ_peak_nat.
    #
    # Recoil quantities (per absorbed photon):
    #   v_rec = ħk/m              recoil velocity [m/s]
    #   E_rec = ħ²k²/(2m) = ½m·v_rec²   recoil energy [J] → nK
    #
    # Doppler cooling limit:
    #   T_Dop = ħ·Γ/(2·k_B)      minimum temperature from Doppler cooling [K] → μK
    #
    # Capture velocity:
    #   v_cap = Γ/k               maximum velocity that can be captured [m/s]
    mass_kg = mass_amu * AMU_KG
    dom_k   = 2 * math.pi / dom_wl_m   # wave vector [rad/m]

    # Doppler FWHM [Hz]
    delta_nu_dop = (dom_nu / C_M_S) * math.sqrt(
        8 * KB_J_K * T_DOPPLER_K * math.log(2) / mass_kg
    )

    # Peak cross-section with Doppler broadening [m²]
    # Matches compare_elements.sigma_peak_doppler:
    #   σ_D = σ_nat · (Δν_nat / Δν_D) · √(ln 2 / π)
    # (scales the pipeline natural peak; see Overview §07 for the
    # proper-Gaussian factor-of-two relative to this convention).
    sigma_peak_dop_m2  = (
        sigma_peak_m2 * (delta_nu_nat / delta_nu_dop)
        * math.sqrt(math.log(2) / math.pi)
    )

    # Saturation intensity with Doppler broadening [W/m²]
    I_sat_dop_Wm2   = dom_E_J / (2.0 * sigma_peak_dop_m2 * tau_s)

    # Recoil velocity [m/s] and energy
    v_rec_ms   = (H_J_S / (2 * math.pi)) * dom_k / mass_kg    # ħk/m
    E_rec_J    = 0.5 * mass_kg * v_rec_ms**2

    # Doppler cooling temperature limit
    Gamma_s  = A_total
    T_dop_K  = (H_J_S / (2 * math.pi)) * Gamma_s / (2 * KB_J_K)

    return Lifetime(
        state=state_label,
        n=tlist[0].upper_n,
        l=tlist[0].upper_l,
        A_total_s=A_total,
        sigma_A_s=sigma_A,
        lifetime_s=tau_s,
        sigma_tau_s=sigma_tau_s,
        quality_acc=quality_acc,
        quality_delta=quality_delta,
        channels=sorted(tlist, key=lambda t: -t.A_s),
        B21_m3_J_s2=B21_SI,
        B12_m3_J_s2=B12_SI,
        delta_nu_nat_Hz=delta_nu_nat,
        sigma_peak_nat_m2=sigma_peak_m2,
        I_sat_nat_W_m2=I_sat_Wm2,
        mass_amu=mass_amu,
        delta_nu_dop_Hz=delta_nu_dop,
        sigma_peak_dop_m2=sigma_peak_dop_m2,
        I_sat_dop_W_m2=I_sat_dop_Wm2,
        v_rec_ms=v_rec_ms,
        E_rec_J=E_rec_J,
        E_rec_nK=E_rec_J / KB_J_K * 1e9,
        T_dop_uK=T_dop_K * 1e6,
        v_cap_ms=Gamma_s / dom_k,
    )


# ============================================================================
# ONE SPECIES
# ============================================================================

def compute(
    transitions: Sequence[Mapping], sources: FValueSources, *,
    qd: dict[str, float], z_core: float, IE: float, gs_label: str, mass_amu: float,
    is_ion: bool = False, is_metal: bool = False,
    orca_only: bool = False, nist_only: bool = False,
) -> LifetimesResult:
    """Einstein A of every transition, then τ and line parameters per upper state."""
    nist = sources.nist
    nist_true = {} if (is_ion or is_metal) else build_true_multiplet_f(nist.get('fosc_jl') or {})
    numerov_cal = None
    if not is_metal:
        # Per (l_lower, l_upper) scale, fitted only on NIST pairs inside the Numerov
        # domain (n*_l, n*_u > 2.5); types with <5 inlier pairs use the pooled median.
        numerov_cal = fit_numerov_scales_by_type(
            qd, calibration_fvalues(nist['fosc'], nist_true), z_core=z_core)
    c = _Context(sources=sources, qd=qd, z_core=z_core, IE=IE, gs_label=gs_label,
                 is_ion=is_ion, is_metal=is_metal, orca_only=orca_only, nist_only=nist_only,
                 nist_true=nist_true, numerov_cal=numerov_cal)

    rows: list[TransitionA] = []
    dropped: list[Dropped] = []
    for t in transitions:
        if is_metal:
            metal_row = _metal_row(t, c)
            if metal_row is not None:
                rows.append(metal_row)
            continue
        line = _line_row(t, c)
        if isinstance(line, str):
            dropped.append(Dropped(t, line))
        else:
            rows.append(line)

    new_rows, dropped, series = extrapolate_below_gate(rows, dropped, qd)
    rows.extend(new_rows)

    # τ = 1 / Σ A_ul over the channels of each upper state
    A_by_upper: dict[str, list[TransitionA]] = defaultdict(list)
    for row in rows:
        A_by_upper[row.upper_label].append(row)
    lifetimes = []
    for state_label, tlist in sorted(A_by_upper.items(),
                                     key=lambda x: parse_state_label(x[0])):
        lt = lifetime_of(state_label, tlist, mass_amu)
        if lt is not None:
            lifetimes.append(lt)

    return LifetimesResult(numerov_calibration=numerov_cal, transitions=rows,
                           dropped=dropped, n_extrapolated=len(new_rows),
                           series_fits=series, lifetimes=lifetimes)
