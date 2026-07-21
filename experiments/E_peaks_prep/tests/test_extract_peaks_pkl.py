# coding: ascii
"""Tests de extract_peaks_pkl.py -- corre localmente, requiere rdkit."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from extract_peaks_pkl import extract_peaks_from_pkl_molecule


def test_ethanol_one_peak_per_carbon_with_diastereotopic_average():
    # Etanol CCO -> AddHs: atomo 0=C(CH3, H en 3,4,5), 1=C(CH2, H en 6,7),
    # 2=O (indices verificados con RDKit directamente, ver Task 1).
    # H diastereotopicos en el CH2 (6 y 7) con shifts DISTINTOS a proposito,
    # para confirmar que se promedian en vez de generar 2 picos.
    nmr_shifts = {
        0: 18.0,   # C del CH3
        1: 58.0,   # C del CH2
        3: 1.2, 4: 1.2, 5: 1.2,   # H del CH3 (isocronos)
        6: 3.5, 7: 3.7,           # H del CH2 (diastereotopicos, shifts distintos)
    }

    peaks = extract_peaks_from_pkl_molecule("CCO", nmr_shifts)
    assert len(peaks) == 2, f"esperados 2 picos (1 por carbono), salieron {len(peaks)}"

    peaks_by_c = {round(p[0], 3): p for p in peaks}
    assert 18.0 in peaks_by_c, peaks_by_c
    assert 58.0 in peaks_by_c, peaks_by_c

    delta_c, delta_h, amp_ch0, amp_ch1 = peaks_by_c[18.0]
    assert abs(delta_h - 1.2) < 1e-9, delta_h
    assert amp_ch0 == 3.0   # CH3: fase +1 * mult 3
    assert abs(amp_ch1 - 1.0) < 1e-9   # mult 3 / 3

    delta_c, delta_h, amp_ch0, amp_ch1 = peaks_by_c[58.0]
    assert abs(delta_h - 3.6) < 1e-9, delta_h   # promedio de 3.5 y 3.7
    assert amp_ch0 == -2.0   # CH2: fase -1 * mult 2
    assert abs(amp_ch1 - (2.0 / 3.0)) < 1e-9

    print(f"[OK] test_ethanol_one_peak_per_carbon_with_diastereotopic_average -> {peaks}")


def test_carbon_without_shift_is_dropped():
    # Falta el shift del carbono del CH2 (atomo 1) -- ese carbono no debe
    # generar pico (sin delta_c no hay pico posible).
    nmr_shifts = {
        0: 18.0,
        3: 1.2, 4: 1.2, 5: 1.2,
        6: 3.5, 7: 3.7,
    }
    peaks = extract_peaks_from_pkl_molecule("CCO", nmr_shifts)
    assert len(peaks) == 1, f"esperado 1 pico (CH2 sin delta_c se descarta), salieron {len(peaks)}"
    assert abs(peaks[0][0] - 18.0) < 1e-9
    print(f"[OK] test_carbon_without_shift_is_dropped -> {peaks}")


def test_invalid_smiles_returns_empty():
    peaks = extract_peaks_from_pkl_molecule("no_es_un_smiles_valido()", {0: 18.0})
    assert peaks == []
    print("[OK] test_invalid_smiles_returns_empty")


def test_symmetric_carbons_with_identical_shift_collapse_to_one_peak():
    # Propano CCC -> AddHs: atomo 0=C(CH3, H en 3,4,5), 1=C(CH2, H en 6,7),
    # 2=C(CH3, H en 8,9,10) -- verificado con RDKit directamente. Los dos
    # CH3 (0 y 2) son quimicamente equivalentes por simetria -- un DFT real
    # les asigna el MISMO shift (caso real observado en datos de produccion,
    # ej. anillos para-sustituidos). En un HSQC real dan una sola senal
    # (son indistinguibles), asi que el label los cuenta una vez -- la
    # extraccion debe deduplicar picos con (delta_c, delta_h) identicos
    # en vez de generar un pico por cada atomo.
    nmr_shifts = {
        0: 15.5, 2: 15.5,      # los dos C del CH3, shift IDENTICO (simetria)
        1: 16.2,               # C del CH2
        3: 0.9, 4: 0.9, 5: 0.9,       # H del CH3 (atomo 0)
        6: 1.3, 7: 1.3,               # H del CH2
        8: 0.9, 9: 0.9, 10: 0.9,      # H del CH3 (atomo 2, mismo shift que el 0)
    }

    peaks = extract_peaks_from_pkl_molecule("CCC", nmr_shifts)
    assert len(peaks) == 2, f"esperados 2 picos (CH3 simetrico colapsado + CH2), salieron {len(peaks)}"

    peaks_by_c = {round(p[0], 3): p for p in peaks}
    assert 15.5 in peaks_by_c, peaks_by_c
    assert 16.2 in peaks_by_c, peaks_by_c
    print(f"[OK] test_symmetric_carbons_with_identical_shift_collapse_to_one_peak -> {peaks}")


if __name__ == "__main__":
    test_ethanol_one_peak_per_carbon_with_diastereotopic_average()
    test_carbon_without_shift_is_dropped()
    test_invalid_smiles_returns_empty()
    test_symmetric_carbons_with_identical_shift_collapse_to_one_peak()
    print("\n>>> test_extract_peaks_pkl.py OK <<<")
