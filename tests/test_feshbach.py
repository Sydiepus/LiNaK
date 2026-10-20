# -*- coding: utf-8 -*-
"""Na magnetic Feshbach a(B) formula and literature-parameter anchors."""
from __future__ import annotations

import math

import numpy as np
import pytest

from linak.physics.feshbach import Resonance, multi_resonance_a, scattering_length

from helpers import write_result_summary


def test_feshbach_literature_params(na_feshbach_params, anchors):
    a = anchors["feshbach"]
    assert float(na_feshbach_params["a_bg_default_a0"]) == a["a_bg_a0"]
    by_label = {r["label"]: r for r in na_feshbach_params["resonances"]}
    for exp in a["resonances"]:
        row = by_label[exp["label"]]
        assert float(row["B0_G"]) == exp["B0_G"]
        assert float(row["Delta_G"]) == exp["Delta_G"]
        assert float(row["a_bg_a0"]) == exp["a_bg_a0"]


def test_scattering_length_zero_and_background(anchors, results_dir):
    """a(B0 + Delta) = 0 and a far from resonance -> a_bg."""
    a = anchors["feshbach"]
    res = a["resonances"][0]  # 905 G, Delta=1.04
    a_bg = res["a_bg_a0"]
    B0 = res["B0_G"]
    Delta = res["Delta_G"]

    B_zero = B0 + Delta
    a_zero = float(scattering_length(B_zero, a_bg, B0, Delta))
    assert abs(a_zero - 0.0) <= a["tol_abs_a0"]

    a_far = float(scattering_length(-1.0e4, a_bg, B0, Delta))
    assert abs(a_far - a_bg) / abs(a_bg) < 1e-4

    # Pole is NaN
    assert math.isnan(float(scattering_length(B0, a_bg, B0, Delta)))

    write_result_summary(
        results_dir,
        "feshbach_formula",
        {
            "B_zero": B_zero,
            "a_zero": a_zero,
            "a_far_B0": a_far,
            "a_bg": a_bg,
        },
    )


def test_multi_resonance_product_near_background(na_feshbach_params):
    """Far below all poles, multi-resonance a(B) recovers a_bg."""
    a_bg = float(na_feshbach_params["a_bg_default_a0"])
    B = np.array([0.0, 10.0, 50.0])
    resonances = [Resonance.from_json(row) for row in na_feshbach_params["resonances"]]
    vals = multi_resonance_a(B, resonances, a_bg)
    for v in vals:
        assert abs(float(v) - a_bg) / abs(a_bg) < 0.02


# --- engine (physics.feshbach) --------------------------------------------


def test_van_der_waals_length_matches_chin_2010_for_rb87():
    """R_vdW = β₆/2 = 82.6 a₀ for ⁸⁷Rb with C₆ = 4698 au (Chin et al., RMP 82, 1225)."""
    from linak.physics.feshbach import beta6_a0

    assert beta6_a0(4698.0, 86.909180527) / 2 == pytest.approx(82.6, rel=2e-3)


def test_compute_single_resonance_combined_equals_isolated():
    from linak.physics.feshbach import Resonance, compute

    res = Resonance.from_json({"B0_G": 907.0, "Delta_G": 1.0, "a_bg_a0": 63.0})
    result = compute([res], 63.0, 1556.0, 22.98976928, kT_uK=1.0, B_points=500)
    assert result.B_G[-1] == pytest.approx(907.0 * 1.15), "default range: 15 % past B₀"
    np.testing.assert_allclose(result.a_combined_a0, result.curves[0].a_a0, equal_nan=True)
    assert result.curves[0].resonance.label == "B0=907.0G"
    assert result.curves[0].a_bg_a0 == 63.0


def test_cross_section_is_4_pi_a_squared_at_low_energy():
    """σ_el → 4πa² for k·a ≪ 1 (distinguishable-particle form, see P25)."""
    from linak.physics.feshbach import sigma_el_a0

    assert float(sigma_el_a0(50.0, 1e-8)) == pytest.approx(4 * math.pi * 50.0**2, rel=1e-9)


def test_resonance_without_own_background_uses_the_table_default():
    from linak.physics.feshbach import Resonance

    res = Resonance.from_json({"label": "x", "B0_G": 10, "Delta_G": 0.5})
    assert res.a_bg_a0 is None and res.background(54.5) == 54.5
    assert (res.partial_wave, res.channel, res.ref) == ("s", "", "")
