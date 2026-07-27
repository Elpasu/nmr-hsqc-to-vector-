# coding: utf-8
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from adapter import MULT_H, build_inputs, parse_formula, true_vector  # noqa: E402

CLASS_NAMES = [
    "CH3", "CH2", "CH", "Cq", "CH3-O", "CH2-O", "CH-O", "Cq-O", "CH3-N",
    "CH2-N", "CH-N", "Cq-N", "=CH2", "=CH/Ar", "Cqsp2", "Aldeh", "Imina",
    "C-2X", "C-3X",
]
NORM = {"c13_ppm_min": 0, "c13_ppm_max": 220, "h1_ppm_min": -1,
        "h1_ppm_max": 15, "amp_ch0_scale": 3.0}


def test_parse_formula_chon():
    f = parse_formula("C10H12N2O")
    assert f == {"C": 10, "H": 12, "N": 2, "O": 1, "S": 0, "Hal": 0}


def test_parse_formula_implicit_one_and_missing():
    f = parse_formula("CH4")
    assert f["C"] == 1 and f["H"] == 4 and f["N"] == 0 and f["O"] == 0


def test_parse_formula_halogens_sum():
    f = parse_formula("C2H4ClBr")
    assert f["Hal"] == 2 and f["C"] == 2


def _mol_etanol():
    # CH3-CH2-OH: CH3 (d 1.2), CH2-O (d 3.7), 2 carbonos protonados, sin Cq.
    return [
        {"delta_c": 18.0, "delta_h": 1.2, "mult": "CH3", "clase": "CH3"},
        {"delta_c": 58.0, "delta_h": 3.7, "mult": "CH2", "clase": "CH2-O"},
    ]


def test_amp_ch2_is_negative_two():
    peaks = [{"delta_c": 58.0, "delta_h": 3.7, "mult": "CH2"}]
    peaks_ch, _, _, _, _ = build_inputs(peaks, parse_formula("C2H6O"), NORM)
    # amp_ch0 crudo = -2, normalizado /3.0 -> -0.6667
    assert np.isclose(peaks_ch[0, 2], -2.0 / 3.0, atol=1e-4)
    assert np.isclose(peaks_ch[0, 3], 2.0 / 3.0, atol=1e-4)  # amp_ch1 = mult/3


def test_amp_ch3_is_plus_three():
    peaks = [{"delta_c": 18.0, "delta_h": 1.2, "mult": "CH3"}]
    peaks_ch, _, _, _, _ = build_inputs(peaks, parse_formula("C2H6O"), NORM)
    assert np.isclose(peaks_ch[0, 2], 3.0 / 3.0, atol=1e-4)   # +3 /3 = 1.0
    assert np.isclose(peaks_ch[0, 3], 3.0 / 3.0, atol=1e-4)


def test_normalization_matches_config():
    peaks = [{"delta_c": 110.0, "delta_h": 7.0, "mult": "CH"}]
    peaks_ch, _, _, _, _ = build_inputs(peaks, parse_formula("C6H6"), NORM)
    assert np.isclose(peaks_ch[0, 0], 110.0 / 220.0, atol=1e-4)   # dC/220
    assert np.isclose(peaks_ch[0, 1], (7.0 + 1.0) / 16.0, atol=1e-4)  # (dH+1)/16


def test_cq_no_crosspeak_but_in_13c():
    peaks = [
        {"delta_c": 40.0, "delta_h": 1.5, "mult": "CH2"},
        {"delta_c": 150.0, "delta_h": None, "mult": "Cq"},
    ]
    peaks_ch, mask_ch, peaks_13c, mask_13c, _ = build_inputs(
        peaks, parse_formula("C3H6"), NORM)
    assert peaks_ch.shape[0] == 1          # solo el CH2 genera crosspeak
    assert peaks_13c.shape[0] == 2         # ambos carbonos en 13C
    assert mask_ch.sum() == 1 and mask_13c.sum() == 2


def test_cond_derived_from_spectrum_and_formula():
    _, _, _, _, cond = build_inputs(_mol_etanol(), parse_formula("C2H6O"), NORM)
    # [total_senales, total_CH2, C,H,N,O,S,Hal]
    assert cond[0] == 2      # total senales = nro filas
    assert cond[1] == 1      # un CH2 (el CH2-O del etanol)
    assert cond[2] == 2 and cond[4] == 0 and cond[5] == 1   # C=2, N=0, O=1


def test_true_vector_histogram():
    tv = true_vector(_mol_etanol(), CLASS_NAMES)
    assert tv.sum() == 2
    assert tv[0] == 1        # CH3 en indice 0
    assert tv[5] == 1        # CH2-O en indice 5


def _expect_valueerror(fn):
    try:
        fn()
    except ValueError:
        return
    raise AssertionError("se esperaba ValueError y no se lanzo")


def test_parse_formula_rejects_unknown_element():
    _expect_valueerror(lambda: parse_formula("C10H12P"))   # P no soportado
    _expect_valueerror(lambda: parse_formula("C6H6Si"))    # Si no soportado


def test_parse_formula_rejects_empty_and_junk():
    _expect_valueerror(lambda: parse_formula(""))
    _expect_valueerror(lambda: parse_formula("   "))
    _expect_valueerror(lambda: parse_formula("xyz"))       # sin mayuscula inicial


def test_build_inputs_rejects_empty_peaks():
    _expect_valueerror(lambda: build_inputs([], parse_formula("C2H6O"), NORM))


def test_build_inputs_rejects_bad_mult():
    peaks = [{"delta_c": 40.0, "delta_h": 1.5, "mult": "CH4"}]
    _expect_valueerror(lambda: build_inputs(peaks, parse_formula("C2H6"), NORM))


def test_build_inputs_rejects_cq_with_dh_and_ch_without_dh():
    cq_dh = [{"delta_c": 150.0, "delta_h": 7.0, "mult": "Cq"}]
    _expect_valueerror(lambda: build_inputs(cq_dh, parse_formula("C3H6"), NORM))
    ch_no_dh = [{"delta_c": 110.0, "delta_h": None, "mult": "CH"}]
    _expect_valueerror(lambda: build_inputs(ch_no_dh, parse_formula("C6H6"), NORM))


def test_true_vector_missing_clase_raises():
    peaks = [{"delta_c": 18.0, "delta_h": 1.2, "mult": "CH3"}]  # sin 'clase'
    _expect_valueerror(lambda: true_vector(peaks, CLASS_NAMES))


if __name__ == "__main__":
    import traceback
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    n = 0
    for fn in fns:
        try:
            fn(); n += 1
        except Exception:
            print(f"FALLO: {fn.__name__}"); traceback.print_exc(); sys.exit(1)
    print(f">>> {n} TESTS OK <<<")
