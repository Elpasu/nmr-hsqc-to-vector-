# coding: ascii
"""Labels de carbonos totales (Exp J). El vector cuenta CARBONOS, no senales:
el benceno da =CH/Ar=6, no 1. La clasificacion de cada carbono es la MISMA que
la del esquema viejo (classify_carbon portado de Gen_vector.py); lo unico que
cambia es que no se colapsan los equivalentes por simetria."""
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from make_labels_totales import vector_carbonos_totales, vector_con_simetria

SHORT_NAMES = [
    "CH3", "CH2", "CH", "Cq", "CH3-O", "CH2-O", "CH-O", "Cq-O", "CH3-N",
    "CH2-N", "CH-N", "Cq-N", "=CH2", "=CH/Ar", "Cqsp2", "Aldeh", "Imina",
    "C-2X", "C-3X",
]
IDX = {n: i for i, n in enumerate(SHORT_NAMES)}


def _dic(vec):
    return {SHORT_NAMES[i]: int(vec[i]) for i in range(19) if vec[i]}


def test_benceno_da_seis():
    """El caso que motiva todo el experimento: 6 carbonos, no 1 senal."""
    v = vector_carbonos_totales("c1ccccc1")
    assert _dic(v) == {"=CH/Ar": 6}, _dic(v)
    print("[OK] benceno -> =CH/Ar: 6")


def test_tolueno_desglosa_orto_meta_para():
    """3 senales aromaticas CH (orto/meta/para) pero 5 carbonos: 2+2+1."""
    v = vector_carbonos_totales("Cc1ccccc1")
    assert _dic(v) == {"CH3": 1, "=CH/Ar": 5, "Cqsp2": 1}, _dic(v)
    print("[OK] tolueno -> CH3:1, =CH/Ar:5, Cqsp2:1")


def test_isopropanol_cuenta_los_dos_metilos():
    v = vector_carbonos_totales("CC(C)O")
    assert _dic(v) == {"CH3": 2, "CH-O": 1}, _dic(v)
    print("[OK] isopropanol -> CH3:2 (los dos metilos equivalentes), CH-O:1")


def test_suma_igual_a_carbonos_de_la_formula():
    from rdkit import Chem
    for smi in ["c1ccccc1", "Cc1ccccc1", "CC(C)O", "CCO", "O=C(N)c1ccccc1",
                "C=CCN1CC(=O)NC1=O", "ClC(Cl)Cl"]:
        m = Chem.MolFromSmiles(smi)
        n_c = sum(1 for a in m.GetAtoms() if a.GetAtomicNum() == 6)
        assert vector_carbonos_totales(smi).sum() == n_c, smi
    print("[OK] sum(vector) == C de la formula en todos los casos")


def test_totales_siempre_mayor_o_igual_que_con_simetria():
    """El vector nuevo nunca puede ser MENOR que el viejo clase por clase: el
    colapso solo puede esconder carbonos, nunca inventarlos."""
    for smi in ["c1ccccc1", "Cc1ccccc1", "CC(C)O", "CCO", "c1ccc2ccccc2c1"]:
        tot = vector_carbonos_totales(smi)
        sim = vector_con_simetria(smi)
        assert np.all(tot >= sim), (smi, tot.tolist(), sim.tolist())
    print("[OK] vector de totales >= vector con simetria, clase por clase")


def test_con_simetria_reproduce_el_esquema_viejo():
    """vector_con_simetria es la referencia contra la que se valida el puerto.
    Con colapso, el benceno tiene que volver a dar 1."""
    assert _dic(vector_con_simetria("c1ccccc1")) == {"=CH/Ar": 1}
    assert _dic(vector_con_simetria("CC(C)O")) == {"CH3": 1, "CH-O": 1}
    print("[OK] vector_con_simetria reproduce el conteo por senales")


def test_smiles_invalido_levanta_valueerror():
    try:
        vector_carbonos_totales("no-es-un-smiles")
    except ValueError:
        print("[OK] SMILES invalido -> ValueError (no un vector de ceros silencioso)")
        return
    raise AssertionError("se esperaba ValueError")


if __name__ == "__main__":
    test_benceno_da_seis()
    test_tolueno_desglosa_orto_meta_para()
    test_isopropanol_cuenta_los_dos_metilos()
    test_suma_igual_a_carbonos_de_la_formula()
    test_totales_siempre_mayor_o_igual_que_con_simetria()
    test_con_simetria_reproduce_el_esquema_viejo()
    test_smiles_invalido_levanta_valueerror()
    print("\n>>> LABELS TOTALES OK <<<")
