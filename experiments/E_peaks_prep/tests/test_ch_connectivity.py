# coding: ascii
"""Tests de ch_connectivity.py -- corre localmente, requiere rdkit
(disponible en la maquina local de Lucas)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rdkit import Chem

from ch_connectivity import get_carbon_multiplicity, get_ch_connectivity_with_multiplicity


def test_ethanol_connectivity():
    # Etanol CCO -> AddHs: atomo 0=C(CH3), 1=C(CH2), 2=O, 3-5=H de CH3,
    # 6-7=H de CH2, 8=H del OH. Verificado corriendo RDKit directamente.
    mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))

    mult_ch3 = get_carbon_multiplicity(mol, 0)
    mult_ch2 = get_carbon_multiplicity(mol, 1)
    assert mult_ch3 == 3, f"esperado CH3 (mult=3), salio {mult_ch3}"
    assert mult_ch2 == 2, f"esperado CH2 (mult=2), salio {mult_ch2}"

    ch_pairs = get_ch_connectivity_with_multiplicity(mol)
    pairs_ch3 = [p for p in ch_pairs if p["c_idx"] == 0]
    pairs_ch2 = [p for p in ch_pairs if p["c_idx"] == 1]

    assert len(pairs_ch3) == 3, f"esperados 3 pares C-H para el CH3, salio {len(pairs_ch3)}"
    assert {p["h_idx"] for p in pairs_ch3} == {3, 4, 5}
    assert all(p["multiplicity"] == 3 for p in pairs_ch3)

    assert len(pairs_ch2) == 2, f"esperados 2 pares C-H para el CH2, salio {len(pairs_ch2)}"
    assert {p["h_idx"] for p in pairs_ch2} == {6, 7}
    assert all(p["multiplicity"] == 2 for p in pairs_ch2)
    print(f"[OK] test_ethanol_connectivity -> CH3 pairs={pairs_ch3} CH2 pairs={pairs_ch2}")


def test_oxygen_has_no_multiplicity():
    mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    mult_o = get_carbon_multiplicity(mol, 2)
    assert mult_o == -1, f"esperado -1 para atomo no-carbono, salio {mult_o}"
    print("[OK] test_oxygen_has_no_multiplicity")


if __name__ == "__main__":
    test_ethanol_connectivity()
    test_oxygen_has_no_multiplicity()
    print("\n>>> test_ch_connectivity.py OK <<<")
