# coding: ascii
"""5a feature de los crosspeaks: la DEGENERACION (cuantos carbonos comparten
esa senal). El dato ya existia en el pipeline de Fase 1b y se descartaba:
_dedupe_symmetric_peaks tiraba los duplicados en vez de contarlos."""
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from make_peaks_degeneracion import (
    agrupar_con_degeneracion, build_padded_arrays_n,
    extract_peaks_deg_from_pkl_molecule,
)


def test_agrupar_cuenta_en_vez_de_descartar():
    """Tres picos, dos con el MISMO (delta_c, delta_h): salen 2 picos, el
    primero con degeneracion 2."""
    peaks = [
        (128.5, 7.20, 1.0, 0.333),
        (128.5, 7.20, 1.0, 0.333),   # equivalente por simetria
        (21.4, 2.35, 3.0, 1.0),
    ]
    out = agrupar_con_degeneracion(peaks)
    assert len(out) == 2, out
    assert out[0] == (128.5, 7.20, 1.0, 0.333, 2.0), out[0]
    assert out[1] == (21.4, 2.35, 3.0, 1.0, 1.0), out[1]
    print("[OK] los duplicados se cuentan (degeneracion), no se descartan")


def test_agrupar_conserva_el_orden_de_aparicion():
    peaks = [(10.0, 1.0, 3.0, 1.0), (20.0, 2.0, 1.0, 0.333), (10.0, 1.0, 3.0, 1.0)]
    out = agrupar_con_degeneracion(peaks)
    assert [p[0] for p in out] == [10.0, 20.0], out
    assert out[0][4] == 2.0 and out[1][4] == 1.0
    print("[OK] se conserva el orden de aparicion del primer pico de cada grupo")


def test_agrupar_sin_simetria_da_todo_uno():
    peaks = [(10.0, 1.0, 3.0, 1.0), (20.0, 2.0, 1.0, 0.333)]
    out = agrupar_con_degeneracion(peaks)
    assert all(p[4] == 1.0 for p in out), out
    print("[OK] sin simetria, toda la degeneracion vale 1")


def test_benceno_da_un_pico_con_degeneracion_seis():
    """Los 6 CH del benceno comparten shift: UNA senal, degeneracion 6.
    Integracion equivalente = 6 carbonos x 1 H = 6H, que es lo que se lee."""
    shifts = {}
    from rdkit import Chem
    mol = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1"))
    for a in mol.GetAtoms():
        shifts[a.GetIdx()] = 128.5 if a.GetAtomicNum() == 6 else 7.26
    peaks = extract_peaks_deg_from_pkl_molecule("c1ccccc1", shifts)
    assert len(peaks) == 1, peaks
    assert peaks[0][4] == 6.0, peaks[0]
    print("[OK] benceno -> 1 crosspeak con degeneracion 6")


def test_degeneracion_por_mult_reconstruye_los_H():
    """Sum(degeneracion x mult) sobre los crosspeaks == numero de H sobre
    carbono. Es la relacion que hace que la integracion sea el dato correcto."""
    from rdkit import Chem
    mol = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1"))
    shifts = {a.GetIdx(): (128.5 if a.GetAtomicNum() == 6 else 7.26)
              for a in mol.GetAtoms()}
    peaks = extract_peaks_deg_from_pkl_molecule("c1ccccc1", shifts)
    # amp_ch1 = mult / 3  ->  mult = amp_ch1 * 3
    h_total = sum(p[4] * round(p[3] * 3) for p in peaks)
    assert h_total == 6, h_total
    print("[OK] Sum(degeneracion x mult) == 6 H del benceno")


def test_build_padded_arrays_cinco_columnas():
    peaks_per_mol = [
        [(1.0, 2.0, 3.0, 4.0, 5.0), (6.0, 7.0, 8.0, 9.0, 1.0)],
        [(1.5, 2.5, 3.5, 4.5, 2.0)],
    ]
    arr, mask = build_padded_arrays_n(peaks_per_mol, 5)
    assert arr.shape == (2, 2, 5), arr.shape
    assert mask.tolist() == [[True, True], [True, False]]
    assert arr[1, 1].tolist() == [0.0, 0.0, 0.0, 0.0, 0.0]
    assert arr[0, 0, 4] == 5.0
    print("[OK] padding a 5 columnas, mascara correcta")


def test_molecula_sin_shifts_da_lista_vacia():
    assert extract_peaks_deg_from_pkl_molecule("c1ccccc1", {}) == []
    print("[OK] sin shifts en el pkl -> lista vacia (no excepcion)")


def test_smiles_invalido_da_lista_vacia():
    assert extract_peaks_deg_from_pkl_molecule("no-es-smiles", {}) == []
    print("[OK] SMILES invalido -> lista vacia (mismo contrato que Fase 1b)")


if __name__ == "__main__":
    test_agrupar_cuenta_en_vez_de_descartar()
    test_agrupar_conserva_el_orden_de_aparicion()
    test_agrupar_sin_simetria_da_todo_uno()
    test_benceno_da_un_pico_con_degeneracion_seis()
    test_degeneracion_por_mult_reconstruye_los_H()
    test_build_padded_arrays_cinco_columnas()
    test_molecula_sin_shifts_da_lista_vacia()
    test_smiles_invalido_da_lista_vacia()
    print("\n>>> PEAKS DEGENERACION OK <<<")
