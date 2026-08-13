# Exp J — Vector de carbonos totales — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Dejar listas para `sbatch` en login-1/A10 dos corridas (J-A y J-0) que entrenan el Set
Transformer contra un target nuevo — el vector de **carbonos totales** en vez de señales — usando la
degeneración derivada de la integración de protones como 5ª feature de cada crosspeak.

**Architecture:** Carpeta autocontenida `experiments/J_carbonos_totales/` con copias del E3 y tres
deltas: labels sin colapso de simetría, crosspeaks con una 5ª columna (degeneración), y
`peak_features` configurable para que **un solo archivo de modelo** sirva a la corrida experimental
(5 features) y a la de control (4). Los datos se regeneran localmente desde los pkl DFT; el
entrenamiento va al cluster.

**Tech Stack:** Python 3, numpy, PyYAML, RDKit, PyTorch (CPU local para smoke tests, CUDA en el
cluster), SLURM.

**Spec:** `docs/superpowers/specs/2026-08-12-vector-carbonos-totales-design.md` (commit `18519cb`).

## Global Constraints

- **Regla dura 1:** `num_workers: 0` en los dos configs. Nunca subirlo (h5py/DataLoader deadlockea).
- **Regla dura 2:** SLURM usa `#SBATCH --gres=gpu:1`, **NO** `--gpus=1`.
- **Regla dura 3:** nada hardcodeado — rutas, escalas de normalización y constantes salen del config.
- **Regla dura 4:** los `.py` de este repo llevan cabecera `# coding: ascii` y comentarios **sin
  tildes ni ñ**. Los `.md` y `.yaml` sí pueden llevar tildes. Seguir el estilo de
  `experiments/E3_dos_conjuntos/*.py`.
- **Regla dura 5:** smoke test local antes de cualquier `sbatch`.
- **Regla dura 6:** scheduler `patience=8, factor=0.7` en los dos configs.
- **Regla dura 7:** `num_classes=19` y el orden de clases de `config/db.yaml` — no se tocan. Un
  desalineamiento de labels entrena basura **sin tirar error**.
- **Regla dura 8:** val congelado y seed idénticos entre J-A y J-0, o sus EMAs no son comparables.
- **NO pisar los archivos del checkpoint congelado.** `vectors_13c_19v_202465.npy` y
  `peaks_pkl_202465.npz` son el ground truth del modelo en producción: los archivos nuevos llevan
  nombres nuevos (`vectors_19v_totales_202465.npy`, `peaks_pkl_deg_202465.npz`).
- **Cluster objetivo: login-1 / A10 ("capitán")** — user `lpassaglia.iquir`, env `NMR_env`, partición
  `gpua10_hi`, `base_dir` `/home/lpassaglia.iquir/DB_200k`. No se genera script para Clementina.
- **Claude Code no lanza SLURM ni lee logs del cluster.** El entregable es "listo para `sbatch`".
- **Entorno local:** torch 2.13.0+cpu, numpy, PyYAML, rdkit, matplotlib. Los datos crudos están en
  `E:/Proyectos/SciTrix/ScitrixDB/DB_nmr_to_vector/` (`144K/`, `58K/`, `202K_suma/`), así que **las
  validaciones de datos y los smoke tests son ejecutables localmente de verdad**.
- **Convención de tests del repo:** NO se usa pytest. Son scripts con funciones `test_*` y un bloque
  `if __name__ == "__main__":` que las corre en orden e imprime `>>> ... OK <<<`. Se ejecutan con
  `python tests/test_x.py`.

---

## File Structure

**Se crean** (nada se modifica fuera de `RESULTS.md`):

| Archivo | Responsabilidad | Task |
|---|---|---|
| `experiments/J_carbonos_totales/prep/config_prep.yaml` | Rutas locales de los datos crudos | 1 |
| `experiments/J_carbonos_totales/prep/make_labels_totales.py` | Labels sin colapso de simetría + gate de verificación | 1 |
| `experiments/J_carbonos_totales/prep/tests/test_labels_totales.py` | Casos conocidos + suma == C | 1 |
| `experiments/J_carbonos_totales/prep/make_peaks_degeneracion.py` | Crosspeaks con 5ª feature | 2 |
| `experiments/J_carbonos_totales/prep/tests/test_peaks_degeneracion.py` | Degeneración correcta y consistente | 2 |
| `experiments/J_carbonos_totales/{oraculo,split_utils,config_utils,device_utils}.py` | Copias **sin cambios** del E3 | 3 |
| `experiments/J_carbonos_totales/model_j_settransformer.py` | Copia del E3 + `peak_features` configurable | 3 |
| `experiments/J_carbonos_totales/tests/test_forward_j.py` | Forward con 4 y con 5 features | 3 |
| `experiments/J_carbonos_totales/dataset_j.py` | 5ª feature normalizada + recorte por `peak_features` | 4 |
| `experiments/J_carbonos_totales/tests/test_dataset_j.py` | Normalización, recorte y `cond` | 4 |
| `experiments/J_carbonos_totales/train.py` | Copia del E3, imports a `dataset_j`/`model_j` | 5 |
| `experiments/J_carbonos_totales/evaluate.py` | Copia del E3, imports a `dataset_j` | 5 |
| `experiments/J_carbonos_totales/config_j_a.yaml` | Corrida J-A (`peak_features: 5`) | 5 |
| `experiments/J_carbonos_totales/config_j_0.yaml` | Corrida J-0 (`peak_features: 4`) | 5 |
| `experiments/J_carbonos_totales/run_train_j.sh` | 1 job SLURM = train + eval | 5 |
| `experiments/J_carbonos_totales/tests/test_configs_j.py` | Invariantes entre los dos configs | 5 |
| `experiments/J_carbonos_totales/README.md` · `RATIONALE.md` | Cómo se corre y por qué | 6 |
| `docs/Runs/RESULTS.md` | Sección placeholder del Exp J (**se modifica**) | 6 |

---

## Task 1: Labels de carbonos totales

**Files:**
- Create: `experiments/J_carbonos_totales/prep/config_prep.yaml`
- Create: `experiments/J_carbonos_totales/prep/make_labels_totales.py`
- Test: `experiments/J_carbonos_totales/prep/tests/test_labels_totales.py`

**Interfaces:**
- Consumes: nada de tareas anteriores. Reusa `classify_carbon` y `CAT_INDEX` de
  `experiments/H_inferencia_experimental/smiles_classifier.py` (puerto fiel ya verificado).
- Produces:
  - `vector_carbonos_totales(smiles) -> np.ndarray shape (19,) dtype int32` — sin colapso de simetría.
  - `vector_con_simetria(smiles) -> np.ndarray shape (19,) dtype int32` — CON colapso (el esquema
    viejo), usado solo por el gate de verificación.
  - `verificar_puerto(smiles_array, labels_viejos, n_muestra=None) -> (n_ok, n_total, primer_fallo)`
  - Archivo `vectors_19v_totales_202465.npy` en el `base_dir_202k` del config.

- [ ] **Step 1: Escribir el config de rutas**

Crear `experiments/J_carbonos_totales/prep/config_prep.yaml`:

```yaml
# experiments/J_carbonos_totales/prep/config_prep.yaml
#
# Exp J: rutas LOCALES de los datos crudos (PC Windows de Lucas), no las del
# cluster. Mismo patron que experiments/E_peaks_prep/config_pkl.yaml.
#
# Los archivos de SALIDA llevan nombres NUEVOS a proposito: no se pisan
# vectors_13c_19v_202465.npy ni peaks_pkl_202465.npz, que son el ground truth
# del checkpoint congelado en produccion.

paths:
  base_dir_144k: "E:/Proyectos/SciTrix/ScitrixDB/DB_nmr_to_vector/144K"
  pkl_144k: "nmr_calculated_data_scaled_144K.pkl"
  mol_ids_144k: "mol_ids_144280.npy"
  smiles_144k: "smiles_144280.npy"
  base_dir_58k: "E:/Proyectos/SciTrix/ScitrixDB/DB_nmr_to_vector/58K"
  pkl_58k: "nmr_calculated_data_scaled_58k.pkl"
  mol_ids_58k: "mol_ids_58185.npy"
  smiles_58k: "smiles_58185.npy"
  base_dir_202k: "E:/Proyectos/SciTrix/ScitrixDB/DB_nmr_to_vector/202K_suma"
  smiles_202465: "smiles_202465.npy"
  labels_viejos_202465: "vectors_13c_19v_202465.npy"
  labels_totales_output: "vectors_19v_totales_202465.npy"
  peaks_deg_output: "peaks_pkl_deg_202465.npz"
```

- [ ] **Step 2: Escribir el test que falla**

Crear `experiments/J_carbonos_totales/prep/tests/test_labels_totales.py`:

```python
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
```

- [ ] **Step 3: Correr el test y verificar que falla**

```bash
cd experiments/J_carbonos_totales/prep && python tests/test_labels_totales.py
```

Esperado: FALLA con `ModuleNotFoundError: No module named 'make_labels_totales'`.

- [ ] **Step 4: Escribir el generador de labels**

Crear `experiments/J_carbonos_totales/prep/make_labels_totales.py`:

```python
# coding: ascii
"""make_labels_totales.py -- Exp J: genera el vector de CARBONOS TOTALES.

Diferencia unica respecto del esquema historico: no se colapsan los carbonos
equivalentes por simetria. classify_carbon es exactamente el mismo (puerto fiel
de Gen_vector.py que vive en experiments/H_inferencia_experimental), asi que la
clasificacion de cada carbono no cambia -- solo cuantas veces se cuenta.

Antes de escribir nada, main() corre un GATE: regenera los labels VIEJOS (con
colapso) y los compara contra vectors_13c_19v_202465.npy. Si no coinciden
exactamente, aborta. Sin ese gate, un puerto sutilmente distinto produciria un
ground truth nuevo corrupto sin tirar ningun error (regla dura 7).

Uso:
    python make_labels_totales.py --config config_prep.yaml
    python make_labels_totales.py --config config_prep.yaml --solo-verificar
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import yaml

# El puerto fiel del clasificador vive en Exp H. Se IMPORTA en vez de copiarse:
# es la unica fuente de verdad de las 19 clases y una copia divergente
# corromperia el ground truth en silencio (regla dura 7).
_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO / "experiments" / "H_inferencia_experimental"))

from smiles_classifier import CAT_INDEX, N_CLASSES, classify_carbon  # noqa: E402

from rdkit import Chem, RDLogger  # noqa: E402

RDLogger.DisableLog('rdApp.*')


def _mol_con_h(smiles):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        raise ValueError(f"SMILES invalido: {smiles!r}")
    return Chem.AddHs(mol)


def vector_carbonos_totales(smiles):
    """Vector de 19 clases contando TODOS los carbonos (sin colapso de
    simetria). sum(vector) == numero de carbonos de la formula."""
    mol = _mol_con_h(smiles)
    vec = np.zeros(N_CLASSES, dtype=np.int32)
    for atom in mol.GetAtoms():
        if atom.GetSymbol() != 'C':
            continue
        idx = CAT_INDEX.get(classify_carbon(atom))
        if idx is not None:
            vec[idx] += 1
    return vec


def vector_con_simetria(smiles):
    """Vector del esquema HISTORICO: un carbono por clase de equivalencia
    (CanonicalRankAtoms con breakTies=False). Existe solo para el gate de
    verificacion -- es lo que tiene que reproducir vectors_13c_19v_202465.npy."""
    mol = _mol_con_h(smiles)
    ranks = Chem.CanonicalRankAtoms(mol, breakTies=False)
    vec = np.zeros(N_CLASSES, dtype=np.int32)
    vistos = set()
    for atom, rank in zip(mol.GetAtoms(), ranks):
        if atom.GetSymbol() != 'C':
            continue
        if rank in vistos:
            continue
        vistos.add(rank)
        idx = CAT_INDEX.get(classify_carbon(atom))
        if idx is not None:
            vec[idx] += 1
    return vec


def verificar_puerto(smiles_array, labels_viejos, n_muestra=None):
    """GATE. Regenera los labels viejos con vector_con_simetria y los compara
    con los existentes. Devuelve (n_ok, n_total, primer_fallo) donde
    primer_fallo es None si no hubo ninguno, o (indice, smiles, viejo, nuevo)."""
    n = len(smiles_array)
    indices = range(n) if n_muestra is None else np.random.default_rng(7).choice(
        n, min(n_muestra, n), replace=False)
    n_ok = 0
    primer_fallo = None
    for i in indices:
        try:
            v = vector_con_simetria(smiles_array[i])
        except ValueError:
            v = np.zeros(N_CLASSES, dtype=np.int32)
        if np.array_equal(v, labels_viejos[i]):
            n_ok += 1
        elif primer_fallo is None:
            primer_fallo = (int(i), str(smiles_array[i]),
                            labels_viejos[i].tolist(), v.tolist())
    return n_ok, len(list(indices)) if n_muestra is not None else n, primer_fallo


def main(config_path, solo_verificar=False):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    p = cfg["paths"]
    base_202k = Path(p["base_dir_202k"])

    print("=" * 62)
    print("  EXP J: labels de CARBONOS TOTALES (sin colapso de simetria)")
    print("=" * 62)

    smiles = np.load(base_202k / p["smiles_202465"], allow_pickle=True)
    labels_viejos = np.load(base_202k / p["labels_viejos_202465"]).astype(int)
    print(f"-> Moleculas: {len(smiles)}")

    print("\n[GATE] Verificando que el clasificador reproduce los labels viejos...")
    n_ok, n_tot, fallo = verificar_puerto(smiles, labels_viejos)
    pct = 100.0 * n_ok / n_tot
    print(f"       coincidencia exacta: {n_ok}/{n_tot} ({pct:.4f}%)")
    if fallo is not None:
        i, smi, viejo, nuevo = fallo
        print(f"\n[ABORT] El clasificador NO reproduce el ground truth existente.")
        print(f"        primer fallo en idx={i}: {smi}")
        print(f"        label existente: {viejo}")
        print(f"        regenerado     : {nuevo}")
        print("        Generar labels nuevos con un clasificador que no reproduce")
        print("        los viejos corromperia el target sin avisar (regla dura 7).")
        sys.exit(1)
    print("[GATE OK] el clasificador es fiel; se puede confiar en la version sin colapso.")

    if solo_verificar:
        print("\n(--solo-verificar: no se escribe nada)")
        return

    print("\n-> Generando el vector de carbonos totales...")
    labels_totales = np.zeros((len(smiles), N_CLASSES), dtype=np.int32)
    n_invalidos = 0
    for i, smi in enumerate(smiles):
        try:
            labels_totales[i] = vector_carbonos_totales(smi)
        except ValueError:
            n_invalidos += 1
        if (i + 1) % 25000 == 0:
            print(f"   procesadas {i + 1}/{len(smiles)}")

    n_c = np.array([
        sum(1 for a in Chem.MolFromSmiles(str(s)).GetAtoms() if a.GetAtomicNum() == 6)
        if Chem.MolFromSmiles(str(s)) is not None else 0
        for s in smiles
    ])
    suma = labels_totales.sum(axis=1)
    n_suma_ok = int((suma == n_c).sum())
    escondidos = suma - labels_viejos.sum(axis=1)

    print(f"\n-> SMILES invalidos: {n_invalidos}")
    print(f"-> sum(vector) == C de la formula: {n_suma_ok}/{len(smiles)} "
          f"({100.0 * n_suma_ok / len(smiles):.4f}%)")
    print(f"-> carbonos escondidos por simetria: promedio {escondidos.mean():.2f}, "
          f"max {escondidos.max()}")
    print(f"-> moleculas con simetria: {100.0 * (escondidos > 0).mean():.1f}%")
    if int((escondidos < 0).sum()) != 0:
        print("[ABORT] hay moleculas con MENOS carbonos que senales: imposible.")
        sys.exit(1)

    out = base_202k / p["labels_totales_output"]
    np.save(out, labels_totales)
    print(f"\n[SAVE] {out}")
    print(">>> EXP J make_labels_totales.py OK <<<")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Exp J: labels de carbonos totales")
    ap.add_argument("--config", type=str, default="config_prep.yaml")
    ap.add_argument("--solo-verificar", action="store_true",
                    help="corre solo el gate, no escribe el .npy")
    args = ap.parse_args()
    main(args.config, solo_verificar=args.solo_verificar)
```

- [ ] **Step 5: Correr el test y verificar que pasa**

```bash
cd experiments/J_carbonos_totales/prep && python tests/test_labels_totales.py
```

Esperado: `>>> LABELS TOTALES OK <<<` con 7 líneas `[OK]`.

- [ ] **Step 6: Correr el gate sobre las 202 465 completas**

```bash
cd experiments/J_carbonos_totales/prep && python make_labels_totales.py --config config_prep.yaml --solo-verificar
```

Esperado: `coincidencia exacta: 202465/202465 (100.0000%)` y `[GATE OK]`.

**Si esto no da 100 %, PARAR y reportar BLOCKED.** No seguir, no "arreglar" el clasificador: significa
que el puerto no es fiel y todo el experimento quedaría construido sobre un ground truth dudoso.

- [ ] **Step 7: Generar los labels**

```bash
cd experiments/J_carbonos_totales/prep && python make_labels_totales.py --config config_prep.yaml
```

Esperado: `sum(vector) == C de la formula: 202465/202465 (100.0000%)`, ~62 % de moléculas con
simetría, y `[SAVE] .../vectors_19v_totales_202465.npy`.

- [ ] **Step 8: Commit**

```bash
git add experiments/J_carbonos_totales/prep/
git commit -m "exp J: labels de carbonos totales (sin colapso de simetria) + gate de verificacion"
```

---

## Task 2: Crosspeaks con degeneración

**Files:**
- Create: `experiments/J_carbonos_totales/prep/make_peaks_degeneracion.py`
- Test: `experiments/J_carbonos_totales/prep/tests/test_peaks_degeneracion.py`

**Interfaces:**
- Consumes: `config_prep.yaml` de la Task 1 (mismas rutas). Reusa
  `get_ch_connectivity_with_multiplicity` de `experiments/E_peaks_prep/ch_connectivity.py` y
  `verify_smiles_alignment` de `experiments/E_peaks_prep/extract_peaks_pkl.py`, sin modificarlas.
- Produces:
  - `agrupar_con_degeneracion(peaks) -> list[tuple]` — recibe 4-tuplas, devuelve 5-tuplas.
  - `extract_peaks_deg_from_pkl_molecule(smiles, nmr_shifts) -> list[tuple de 5 floats]`
  - `build_padded_arrays_n(peaks_per_molecule, n_features) -> (np.float32 (N, max, n_features), np.bool_ (N, max))`
  - Archivo `peaks_pkl_deg_202465.npz` con las claves `peaks` (N, 32, 5) y `peaks_mask` (N, 32).

- [ ] **Step 1: Escribir el test que falla**

Crear `experiments/J_carbonos_totales/prep/tests/test_peaks_degeneracion.py`:

```python
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
```

- [ ] **Step 2: Correr el test y verificar que falla**

```bash
cd experiments/J_carbonos_totales/prep && python tests/test_peaks_degeneracion.py
```

Esperado: FALLA con `ModuleNotFoundError: No module named 'make_peaks_degeneracion'`.

- [ ] **Step 3: Escribir el generador de crosspeaks**

Crear `experiments/J_carbonos_totales/prep/make_peaks_degeneracion.py`:

```python
# coding: ascii
"""make_peaks_degeneracion.py -- Exp J: crosspeaks con una 5a feature, la
DEGENERACION de cada senal (cuantos carbonos la comparten).

Es exactamente el pipeline de Fase 1b (extract_peaks_pkl.py) con un solo
cambio: donde _dedupe_symmetric_peaks DESCARTABA los picos con el mismo
(delta_c, delta_h), aca se los CUENTA. El dato ya estaba y se tiraba.

Experimentalmente esa degeneracion es lo que se lee de la integracion del 1H:
integracion_en_H = degeneracion x H_por_carbono. Para el benceno, 1 senal con
integracion 6H -> 6 carbonos equivalentes.

Uso:
    python make_peaks_degeneracion.py --config config_prep.yaml
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import yaml
from rdkit import Chem, RDLogger

# Se reusa la maquinaria ya probada de Fase 1b sin tocarla.
_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO / "experiments" / "E_peaks_prep"))

from ch_connectivity import get_ch_connectivity_with_multiplicity  # noqa: E402
from extract_peaks_pkl import verify_smiles_alignment  # noqa: E402

RDLogger.DisableLog('rdApp.*')

N_FEATURES = 5   # delta_c, delta_h, amp_ch0, amp_ch1, degeneracion


def agrupar_con_degeneracion(peaks):
    """peaks: lista de (delta_c, delta_h, amp_ch0, amp_ch1). Agrupa por
    (delta_c, delta_h) redondeado a 6 decimales -- IDENTICO al criterio de
    _dedupe_symmetric_peaks de Fase 1b -- pero en vez de descartar los
    duplicados devuelve el tamano del grupo como 5a feature.

    Conserva el orden de aparicion del primer pico de cada grupo, igual que
    Fase 1b, para que los dos .npz sean comparables fila por fila.

    Nota: el agrupamiento es por COINCIDENCIA DE SHIFT, no por simetria de
    RDKit. Dos carbonos distintos con shifts accidentalmente iguales cuentan
    como degeneracion 2 -- y eso es lo correcto: en un espectro real esa
    coincidencia es indistinguible de la simetria (se ve una sola senal con el
    doble de integral). Es la colision del 2.19% ya documentada en Fase 1b."""
    orden = []
    grupos = {}
    for peak in peaks:
        clave = (round(peak[0], 6), round(peak[1], 6))
        if clave not in grupos:
            grupos[clave] = [peak, 0]
            orden.append(clave)
        grupos[clave][1] += 1
    return [tuple(grupos[c][0]) + (float(grupos[c][1]),) for c in orden]


def extract_peaks_deg_from_pkl_molecule(smiles, nmr_shifts):
    """Copia de extract_peaks_from_pkl_molecule (Fase 1b) con agrupacion que
    cuenta. smiles: str. nmr_shifts: dict {atom_idx: shift}, indices POST
    AddHs. Devuelve lista de (delta_c, delta_h, amp_ch0, amp_ch1, degeneracion)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []
    mol = Chem.AddHs(mol)
    ch_pairs = get_ch_connectivity_with_multiplicity(mol)

    groups = {}
    for pair in ch_pairs:
        c_idx = pair["c_idx"]
        if c_idx not in groups:
            groups[c_idx] = {"mult": pair["multiplicity"], "h_idxs": []}
        groups[c_idx]["h_idxs"].append(pair["h_idx"])

    peaks = []
    for c_idx, group in groups.items():
        if c_idx not in nmr_shifts:
            continue
        h_shifts = [nmr_shifts[h] for h in group["h_idxs"] if h in nmr_shifts]
        if not h_shifts:
            continue
        delta_c = float(nmr_shifts[c_idx])
        delta_h = float(sum(h_shifts) / len(h_shifts))
        mult = group["mult"]
        phase = -1.0 if mult == 2 else 1.0
        peaks.append((delta_c, delta_h, phase * float(mult), float(mult) / 3.0))
    return agrupar_con_degeneracion(peaks)


def build_padded_arrays_n(peaks_per_molecule, n_features):
    """Version generalizada de build_padded_arrays (Fase 1) que no hardcodea 4
    columnas. Devuelve (peaks (N, max, n_features) float32, mask (N, max) bool)."""
    n = len(peaks_per_molecule)
    max_peaks = max((len(p) for p in peaks_per_molecule), default=0)
    peaks_array = np.zeros((n, max_peaks, n_features), dtype=np.float32)
    mask_array = np.zeros((n, max_peaks), dtype=bool)
    for i, peaks in enumerate(peaks_per_molecule):
        for j, peak in enumerate(peaks):
            peaks_array[i, j] = peak
            mask_array[i, j] = True
    return peaks_array, mask_array


def main(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    p = cfg["paths"]
    base_144 = Path(p["base_dir_144k"])
    base_58 = Path(p["base_dir_58k"])
    base_202 = Path(p["base_dir_202k"])

    print("=" * 62)
    print("  EXP J: crosspeaks con degeneracion (5a feature)")
    print("=" * 62)

    smiles_144 = np.load(base_144 / p["smiles_144k"], allow_pickle=True)
    mol_ids_144 = np.load(base_144 / p["mol_ids_144k"], allow_pickle=True)
    smiles_58 = np.load(base_58 / p["smiles_58k"], allow_pickle=True)
    mol_ids_58 = np.load(base_58 / p["mol_ids_58k"], allow_pickle=True)
    smiles_real = np.load(base_202 / p["smiles_202465"], allow_pickle=True)

    smiles_local = np.concatenate([smiles_144, smiles_58])
    mol_ids_local = np.concatenate([mol_ids_144, mol_ids_58])
    print(f"-> Moleculas locales: {len(smiles_local)} | reales: {len(smiles_real)}")

    ok, idx_malo = verify_smiles_alignment(smiles_local, smiles_real)
    if not ok:
        print(f"[ABORT] desajuste de alineacion (idx={idx_malo}). Los picos"
              f" quedarian pegados a la molecula equivocada.")
        sys.exit(1)
    print("[OK] alineacion verificada: SMILES canonicos coinciden fila por fila")

    with open(base_144 / p["pkl_144k"], "rb") as f:
        pkl_144 = pickle.load(f)
    with open(base_58 / p["pkl_58k"], "rb") as f:
        pkl_58 = pickle.load(f)

    n_total = len(smiles_local)
    n_144 = len(smiles_144)
    peaks_per_molecule = []
    for i in range(n_total):
        pkl = pkl_144 if i < n_144 else pkl_58
        shifts = pkl.get(str(mol_ids_local[i]), {})
        peaks_per_molecule.append(
            extract_peaks_deg_from_pkl_molecule(str(smiles_local[i]), shifts))
        if (i + 1) % 25000 == 0:
            print(f"   procesadas {i + 1}/{n_total}")

    peaks_array, mask_array = build_padded_arrays_n(peaks_per_molecule, N_FEATURES)
    n_picos = mask_array.sum(axis=1)
    deg = peaks_array[:, :, 4]
    deg_validas = deg[mask_array]

    print(f"\n-> shape: {peaks_array.shape}")
    print(f"-> picos por molecula: min={n_picos.min()} max={n_picos.max()} "
          f"promedio={n_picos.mean():.2f}")
    print(f"-> degeneracion: min={deg_validas.min():.0f} max={deg_validas.max():.0f} "
          f"promedio={deg_validas.mean():.2f}")
    print(f"-> senales con degeneracion > 1: "
          f"{100.0 * (deg_validas > 1).mean():.1f}%")

    # Validacion: Sum(degeneracion x mult) <= H sobre carbono. Igualdad solo si
    # el pkl tiene shift para TODOS los H (ver spec 9.1.4): un pkl incompleto da
    # estrictamente menos, y eso no es un test que haya que relajar sino un dato
    # sobre la calidad de los datos.
    mult = np.rint(peaks_array[:, :, 3] * 3.0)
    h_reconstruidos = (deg * mult * mask_array).sum(axis=1)

    h_reales = np.zeros(n_total, dtype=np.float64)
    for i, s in enumerate(smiles_local):
        mol = Chem.MolFromSmiles(str(s))       # una sola vez por molecula:
        if mol is None:                        # parsear dos veces sobre 202465
            continue                           # cuesta varios minutos de mas
        mol = Chem.AddHs(mol)
        h_reales[i] = sum(
            1 for a in mol.GetAtoms()
            if a.GetAtomicNum() == 1
            and any(nb.GetAtomicNum() == 6 for nb in a.GetNeighbors()))
    n_igual = int((h_reconstruidos == h_reales).sum())
    n_exceso = int((h_reconstruidos > h_reales).sum())
    print(f"\n-> Sum(degeneracion x mult) == H sobre carbono: "
          f"{n_igual}/{n_total} ({100.0 * n_igual / n_total:.2f}%)")
    print(f"-> casos con EXCESO (imposible, seria un bug): {n_exceso}")
    if n_exceso > 0:
        print("[ABORT] la degeneracion reconstruye MAS H de los que tiene la"
              " molecula: hay un error en el agrupamiento.")
        sys.exit(1)

    out = base_202 / p["peaks_deg_output"]
    np.savez(out, peaks=peaks_array, peaks_mask=mask_array)
    print(f"\n[SAVE] {out}")
    print(">>> EXP J make_peaks_degeneracion.py OK <<<")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Exp J: crosspeaks con degeneracion")
    ap.add_argument("--config", type=str, default="config_prep.yaml")
    args = ap.parse_args()
    main(args.config)
```

- [ ] **Step 4: Correr el test y verificar que pasa**

```bash
cd experiments/J_carbonos_totales/prep && python tests/test_peaks_degeneracion.py
```

Esperado: `>>> PEAKS DEGENERACION OK <<<` con 8 líneas `[OK]`.

- [ ] **Step 5: Generar los crosspeaks**

```bash
cd experiments/J_carbonos_totales/prep && python make_peaks_degeneracion.py --config config_prep.yaml
```

Esperado: `[OK] alineacion verificada`, shape `(202465, 32, 5)`, `casos con EXCESO: 0`, y
`[SAVE] .../peaks_pkl_deg_202465.npz`.

Nota: el porcentaje de igualdad exacta de H depende de la completitud del pkl. Anotar el número que
salga en el reporte — si es bajo (< 90 %), **reportar DONE_WITH_CONCERNS** con el valor, no seguir
como si nada.

- [ ] **Step 6: Verificar los picos ¹³C contra los labels nuevos**

Con el vector viejo, `sum(label)` era igual al número de picos ¹³C (~100 %). Con carbonos totales
`sum(label)` pasa a ser C, que es **mayor o igual**, y la diferencia es exactamente el número de
carbonos escondidos por simetría. Confirmarlo:

```bash
cd experiments/J_carbonos_totales/prep && python -c "
import numpy as np, yaml
from pathlib import Path
cfg = yaml.safe_load(open('config_prep.yaml', encoding='utf-8'))['paths']
base = Path(cfg['base_dir_202k'])
labels = np.load(base / cfg['labels_totales_output']).astype(int)
n13 = np.load(base / 'peaks_13c_202465.npz')['mask_13c'].sum(axis=1)
suma = labels.sum(axis=1)
print('n_picos_13C <= sum(label): %d/%d (%.2f%%)' % ((n13 <= suma).sum(), len(suma), 100*(n13 <= suma).mean()))
print('violaciones (picos > carbonos, imposible):', int((n13 > suma).sum()))
print('carbonos escondidos = sum(label) - n_picos_13C: promedio %.2f, max %d' % ((suma-n13).mean(), (suma-n13).max()))
"
```

Esperado: `100.00%`, `violaciones: 0`, y un promedio de carbonos escondidos cercano a **1,85**
(el valor medido en el spec §1).

Si aparece alguna violación, **parar y reportar BLOCKED**: significaría que hay más picos ¹³C que
carbonos en la molécula, lo cual es imposible y apunta a un desalineamiento entre archivos.

- [ ] **Step 7: Commit**

```bash
git add experiments/J_carbonos_totales/prep/
git commit -m "exp J: crosspeaks con degeneracion como 5a feature"
```

---

## Task 3: Scaffold de la carpeta J + modelo con `peak_features`

**Files:**
- Create (copias **sin ningún cambio** desde `experiments/E3_dos_conjuntos/`):
  `experiments/J_carbonos_totales/oraculo.py`, `split_utils.py`, `config_utils.py`, `device_utils.py`
- Create: `experiments/J_carbonos_totales/model_j_settransformer.py`
- Test: `experiments/J_carbonos_totales/tests/test_forward_j.py`

**Interfaces:**
- Consumes: nada de las Tasks 1-2 (esta tarea es solo código, no toca datos).
- Produces:
  - `NMR_SetTransformerJ(num_classes=19, peak_features=5, d_model=64, n_heads=4, n_layers=2, n_seeds=1, fusion_hidden=(128, 64))`
  - Los cuatro módulos copiados, con las mismas APIs que en E3:
    `ajustar_conteo_hetero`, `crude_predict`, `canonicalize_smiles`, `remove_leaking_from_train`,
    `subsample_train_idx`, `pick_device`, `wants_pin_memory`, `synchronize`, `seed_everything`,
    `load_config`.

- [ ] **Step 1: Copiar los cuatro módulos sin cambios**

```bash
mkdir -p experiments/J_carbonos_totales/tests
cp experiments/E3_dos_conjuntos/oraculo.py experiments/J_carbonos_totales/oraculo.py
cp experiments/E3_dos_conjuntos/split_utils.py experiments/J_carbonos_totales/split_utils.py
cp experiments/E3_dos_conjuntos/config_utils.py experiments/J_carbonos_totales/config_utils.py
cp experiments/E3_dos_conjuntos/device_utils.py experiments/J_carbonos_totales/device_utils.py
```

Verificar que son idénticos:

```bash
for f in oraculo split_utils config_utils device_utils; do diff -q experiments/E3_dos_conjuntos/$f.py experiments/J_carbonos_totales/$f.py && echo "$f identico"; done
```

Esperado: cuatro líneas `<nombre> identico`.

`oraculo.py` se copia **sin cambios a propósito**: la lógica de forzar `sum(pred) == total` y el
cupo de CH2 es exactamente la misma; lo único que cambia es que ahora `total` vale C de la fórmula en
vez del número de señales. Es un cambio de datos, no de algoritmo.

- [ ] **Step 2: Escribir el test que falla**

Crear `experiments/J_carbonos_totales/tests/test_forward_j.py`:

```python
# coding: ascii
"""Smoke test del Set Transformer del Exp J (regla dura 5). La diferencia con
el E3 es peak_features configurable: 5 para la corrida experimental (con
degeneracion) y 4 para el control. Un solo archivo de modelo sirve a las dos."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model_j_settransformer import NMR_SetTransformerJ

N_CLASSES, MAX_CH, MAX_13C = 19, 32, 40


def _batch(n_feat, B=3):
    """Batch sintetico con mascaras MIXTAS: una molecula llena, una parcial y
    una totalmente enmascarada (el caso que produce NaN si el softmax no esta
    protegido)."""
    mask_ch = torch.ones(B, MAX_CH)
    mask_13c = torch.ones(B, MAX_13C)
    mask_ch[1, 10:] = 0.0
    mask_13c[1, 12:] = 0.0
    mask_ch[2] = 0.0
    mask_13c[2] = 0.0
    return (torch.randn(B, MAX_CH, n_feat), mask_ch,
            torch.randn(B, MAX_13C, 1), mask_13c,
            torch.randn(B, 8))


def test_forward_cinco_features():
    m = NMR_SetTransformerJ(num_classes=N_CLASSES, peak_features=5).eval()
    with torch.no_grad():
        out = m(*_batch(5))
    assert out.shape == (3, N_CLASSES), out.shape
    assert torch.isfinite(out).all(), "NaN/Inf en la salida"
    print(f"[OK] forward con 5 features (J-A) -> {tuple(out.shape)}")


def test_forward_cuatro_features():
    m = NMR_SetTransformerJ(num_classes=N_CLASSES, peak_features=4).eval()
    with torch.no_grad():
        out = m(*_batch(4))
    assert out.shape == (3, N_CLASSES), out.shape
    assert torch.isfinite(out).all(), "NaN/Inf en la salida"
    print(f"[OK] forward con 4 features (J-0 control) -> {tuple(out.shape)}")


def test_proj_ch_respeta_peak_features():
    assert NMR_SetTransformerJ(peak_features=5).proj_ch.in_features == 5
    assert NMR_SetTransformerJ(peak_features=4).proj_ch.in_features == 4
    print("[OK] proj_ch.in_features sigue a peak_features")


def test_default_es_cinco():
    """El default es la corrida experimental: el control tiene que pedir 4
    explicitamente en su config."""
    assert NMR_SetTransformerJ().proj_ch.in_features == 5
    print("[OK] peak_features default = 5")


def test_shape_equivocada_falla_fuerte():
    """Pasarle 4 features a un modelo de 5 tiene que romper, no producir
    numeros silenciosamente equivocados."""
    m = NMR_SetTransformerJ(peak_features=5).eval()
    try:
        with torch.no_grad():
            m(*_batch(4))
    except RuntimeError:
        print("[OK] entrada de 4 features a un modelo de 5 -> RuntimeError")
        return
    raise AssertionError("se esperaba RuntimeError por mismatch de dimensiones")


def test_peak_features_invalido_rechazado():
    for malo in (0, 3, -1):
        try:
            NMR_SetTransformerJ(peak_features=malo)
        except ValueError:
            continue
        raise AssertionError(f"se esperaba ValueError con peak_features={malo}")
    print("[OK] peak_features fuera de {4, 5} -> ValueError")


def test_invariante_a_permutacion_de_picos():
    """Propiedad central de un Set Transformer: el orden de los picos no
    puede cambiar la prediccion."""
    m = NMR_SetTransformerJ(peak_features=5).eval()
    pch, mch, p13, m13, cond = _batch(5, B=1)
    perm = torch.randperm(MAX_CH)
    with torch.no_grad():
        o1 = m(pch, mch, p13, m13, cond)
        o2 = m(pch[:, perm], mch[:, perm], p13, m13, cond)
    assert torch.allclose(o1, o2, atol=1e-4), (o1 - o2).abs().max()
    print("[OK] invariante a permutacion de los crosspeaks")


if __name__ == "__main__":
    test_forward_cinco_features()
    test_forward_cuatro_features()
    test_proj_ch_respeta_peak_features()
    test_default_es_cinco()
    test_shape_equivocada_falla_fuerte()
    test_peak_features_invalido_rechazado()
    test_invariante_a_permutacion_de_picos()
    print("\n>>> FORWARD J OK <<<")
```

- [ ] **Step 3: Correr el test y verificar que falla**

```bash
cd experiments/J_carbonos_totales && python tests/test_forward_j.py
```

Esperado: FALLA con `ModuleNotFoundError: No module named 'model_j_settransformer'`.

- [ ] **Step 4: Escribir el modelo**

Copiar el modelo del E3 y adaptarlo:

```bash
cp experiments/E3_dos_conjuntos/model_e3_settransformer.py experiments/J_carbonos_totales/model_j_settransformer.py
```

Después, en `experiments/J_carbonos_totales/model_j_settransformer.py`:

**(a)** Reemplazar el docstring del módulo (líneas 1-5) por:

```python
# coding: ascii
"""Set Transformer del Exp J -- identico al del Exp E Fase 3 salvo un punto:
peak_features es configurable (4 o 5), porque la corrida experimental agrega
la degeneracion como 5a feature del crosspeak y la de control no la usa.

Un solo archivo de modelo para las dos corridas es deliberado: dos archivos
casi identicos se desincronizan sin avisar, y ahi la comparacion J-A vs J-0
dejaria de medir lo que dice medir.

MAB/SAB/PMA son copia exacta del E3 (no tocar): key_padding_mask + nan_to_num
tras el softmax para que una molecula sin picos no produzca NaN."""
```

**(b)** Renombrar la clase y hacer `peak_features` configurable. Reemplazar la definición de
`class NMR_SetTransformer` y su `__init__` por:

```python
class NMR_SetTransformerJ(nn.Module):
    def __init__(self, num_classes=19, peak_features=5, d_model=64, n_heads=4,
                 n_layers=2, n_seeds=1, fusion_hidden=(128, 64)):
        super().__init__()
        # 4 = (delta_c, delta_h, amp_ch0, amp_ch1)  -- control J-0
        # 5 = las anteriores + degeneracion         -- experimental J-A
        if int(peak_features) not in (4, 5):
            raise ValueError(
                f"peak_features debe ser 4 (control) o 5 (con degeneracion), "
                f"recibido: {peak_features!r}")
        self.peak_features = int(peak_features)
        self.proj_ch = nn.Linear(self.peak_features, d_model)
        self.proj_13c = nn.Linear(1, d_model)
        self.type_emb = nn.Embedding(2, d_model)   # 0=crosspeak, 1=13C
        self.encoder = nn.ModuleList([SAB(d_model, n_heads) for _ in range(n_layers)])
        self.pma = PMA(d_model, n_heads, n_seeds)

        if len(fusion_hidden) != 2:
            raise ValueError(
                f"fusion_hidden debe tener exactamente 2 valores (ancho de las "
                f"dos capas de fusion), recibido: {fusion_hidden!r}")
        h1, h2 = int(fusion_hidden[0]), int(fusion_hidden[1])
        fusion_dim = d_model * n_seeds + 8
        self.fc_fusion1 = nn.Linear(fusion_dim, h1)
        self.fc_fusion2 = nn.Linear(h1, h2)
        self.fc_out = nn.Linear(h2, num_classes)
```

**(c)** El método `forward` queda **exactamente igual** que en el E3 (no se toca ni una línea).

- [ ] **Step 5: Correr el test y verificar que pasa**

```bash
cd experiments/J_carbonos_totales && python tests/test_forward_j.py
```

Esperado: `>>> FORWARD J OK <<<` con 7 líneas `[OK]`.

- [ ] **Step 6: Commit**

```bash
git add experiments/J_carbonos_totales/
git commit -m "exp J: scaffold de la carpeta + modelo con peak_features configurable (4 o 5)"
```

---

## Task 4: `dataset_j.py`

**Files:**
- Create: `experiments/J_carbonos_totales/dataset_j.py`
- Test: `experiments/J_carbonos_totales/tests/test_dataset_j.py`

**Interfaces:**
- Consumes: el `.npz` de 5 columnas de la Task 2 y el `.npy` de labels de la Task 1 (en los tests se
  usan archivos sintéticos, no los reales).
- Produces:
  - `NMRTwoSetsDatasetJ(peaks_ch_path, peaks_13c_path, labels_path, smiles_path, norm_cfg, peak_features=5)`
  - `__getitem__(idx) -> ((peaks_ch, mask_ch, peaks_13c, mask_13c, cond), target)` — mismos 5
    tensores que el E3, con `peaks_ch` de `(max_ch, peak_features)` y `cond` de `(8,)`.

- [ ] **Step 1: Escribir el test que falla**

Crear `experiments/J_carbonos_totales/tests/test_dataset_j.py`:

```python
# coding: ascii
"""dataset_j.py -- normaliza la 5a feature (degeneracion), recorta a
peak_features, y arma el condicionante con la semantica nueva:
cond[0] = carbonos totales (== C de la formula), cond[1] = carbonos CH2."""
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataset_j import NMRTwoSetsDatasetJ

NORM = {"c13_ppm_min": 0, "c13_ppm_max": 220, "h1_ppm_min": -1,
        "h1_ppm_max": 15, "amp_ch0_scale": 3.0, "degeneracion_scale": 4.0}

# Benceno: 1 crosspeak (delta_c 128.5, delta_h 7.26, CH -> amp0 +1, amp1 1/3),
# degeneracion 6. Label de carbonos totales: 6 en =CH/Ar (indice 13).
PEAKS_CH = np.array([[[128.5, 7.26, 1.0, 1.0 / 3.0, 6.0],
                      [0.0, 0.0, 0.0, 0.0, 0.0]]], dtype=np.float32)
MASK_CH = np.array([[True, False]])
PEAKS_13C = np.array([[[128.5], [0.0]]], dtype=np.float32)
MASK_13C = np.array([[True, False]])
LABEL = np.zeros((1, 19), dtype=np.float32)
LABEL[0, 13] = 6.0          # =CH/Ar : 6 carbonos


def _fixture(tmp):
    tmp = Path(tmp)
    np.savez(tmp / "ch.npz", peaks=PEAKS_CH, peaks_mask=MASK_CH)
    np.savez(tmp / "c13.npz", peaks_13c=PEAKS_13C, mask_13c=MASK_13C)
    np.save(tmp / "labels.npy", LABEL)
    np.save(tmp / "smiles.npy", np.array(["c1ccccc1"], dtype=object))
    return (str(tmp / "ch.npz"), str(tmp / "c13.npz"),
            str(tmp / "labels.npy"), str(tmp / "smiles.npy"))


def test_cinco_features_normaliza_la_degeneracion():
    with tempfile.TemporaryDirectory() as tmp:
        ds = NMRTwoSetsDatasetJ(*_fixture(tmp), NORM, peak_features=5)
        (pch, _, _, _, _), _ = ds[0]
    assert pch.shape == (2, 5), pch.shape
    assert abs(float(pch[0, 0]) - 128.5 / 220.0) < 1e-4      # delta_c / 220
    assert abs(float(pch[0, 1]) - (7.26 + 1.0) / 16.0) < 1e-4  # (delta_h+1)/16
    assert abs(float(pch[0, 2]) - 1.0 / 3.0) < 1e-4          # amp_ch0 / 3
    assert abs(float(pch[0, 4]) - 6.0 / 4.0) < 1e-4          # degeneracion / 4
    print("[OK] 5 features, degeneracion normalizada por degeneracion_scale")


def test_cuatro_features_recorta_la_columna():
    """El control lee el MISMO .npz y descarta la 5a columna. Que sea el mismo
    archivo es lo que garantiza que J-A y J-0 difieran solo en la feature."""
    with tempfile.TemporaryDirectory() as tmp:
        ds = NMRTwoSetsDatasetJ(*_fixture(tmp), NORM, peak_features=4)
        (pch, _, _, _, _), _ = ds[0]
    assert pch.shape == (2, 4), pch.shape
    assert abs(float(pch[0, 2]) - 1.0 / 3.0) < 1e-4
    print("[OK] 4 features: la 5a columna se recorta, las otras no cambian")


def test_cond_usa_carbonos_totales():
    """cond[0] == 6 (carbonos, no senales) y cond[2] == 6 (C de la formula del
    benceno). Que coincidan es el punto del experimento."""
    with tempfile.TemporaryDirectory() as tmp:
        ds = NMRTwoSetsDatasetJ(*_fixture(tmp), NORM, peak_features=5)
        (_, _, _, _, cond), target = ds[0]
    assert cond.shape == (8,), cond.shape
    assert float(cond[0]) == 6.0, float(cond[0])   # suma del vector = carbonos
    assert float(cond[1]) == 0.0, float(cond[1])   # el benceno no tiene CH2
    assert float(cond[2]) == 6.0, float(cond[2])   # C de la formula
    assert float(target.sum()) == 6.0
    print("[OK] cond[0] == cond[2] == 6: la suma del vector es C de la formula")


def test_cond_cuenta_carbonos_ch2():
    """cond[1] suma las 4 clases del cupo CH2 (indices 1, 5, 9, 12)."""
    with tempfile.TemporaryDirectory() as tmp:
        paths = _fixture(tmp)
        lab = np.zeros((1, 19), dtype=np.float32)
        lab[0, 1] = 2.0    # CH2
        lab[0, 5] = 1.0    # CH2-O
        lab[0, 13] = 3.0   # =CH/Ar
        np.save(paths[2], lab)
        ds = NMRTwoSetsDatasetJ(*paths, NORM, peak_features=5)
        (_, _, _, _, cond), _ = ds[0]
    assert float(cond[0]) == 6.0
    assert float(cond[1]) == 3.0, float(cond[1])
    print("[OK] cond[1] == 3 carbonos CH2 (2 CH2 + 1 CH2-O)")


def test_peak_features_cinco_con_npz_de_cuatro_falla():
    """Pedir 5 features sobre un .npz que solo tiene 4 columnas tiene que
    romper fuerte: si no, entrenaria con una columna de ceros haciendose pasar
    por la degeneracion."""
    with tempfile.TemporaryDirectory() as tmp:
        paths = list(_fixture(tmp))
        np.savez(paths[0], peaks=PEAKS_CH[:, :, :4], peaks_mask=MASK_CH)
        try:
            NMRTwoSetsDatasetJ(*paths, NORM, peak_features=5)
        except ValueError:
            print("[OK] peak_features=5 con un .npz de 4 columnas -> ValueError")
            return
    raise AssertionError("se esperaba ValueError")


def test_falta_degeneracion_scale_falla():
    """Si el config no trae degeneracion_scale y se piden 5 features, hay que
    fallar en vez de inventar un default silencioso (regla dura 3)."""
    norm_sin = {k: v for k, v in NORM.items() if k != "degeneracion_scale"}
    with tempfile.TemporaryDirectory() as tmp:
        try:
            NMRTwoSetsDatasetJ(*_fixture(tmp), norm_sin, peak_features=5)
        except KeyError:
            print("[OK] sin degeneracion_scale en el config -> KeyError")
            return
    raise AssertionError("se esperaba KeyError")


if __name__ == "__main__":
    test_cinco_features_normaliza_la_degeneracion()
    test_cuatro_features_recorta_la_columna()
    test_cond_usa_carbonos_totales()
    test_cond_cuenta_carbonos_ch2()
    test_peak_features_cinco_con_npz_de_cuatro_falla()
    test_falta_degeneracion_scale_falla()
    print("\n>>> DATASET J OK <<<")
```

- [ ] **Step 2: Correr el test y verificar que falla**

```bash
cd experiments/J_carbonos_totales && python tests/test_dataset_j.py
```

Esperado: FALLA con `ModuleNotFoundError: No module named 'dataset_j'`.

- [ ] **Step 3: Escribir el dataset**

Crear `experiments/J_carbonos_totales/dataset_j.py`:

```python
# coding: ascii
"""dataset_j.py -- Exp J: mismos dos conjuntos de picos que el E3, con dos
diferencias:

  1. Los crosspeaks traen una 5a feature, la DEGENERACION (cuantos carbonos
     comparten esa senal). peak_features=4 la recorta -> corrida de control.
  2. Los labels cuentan CARBONOS TOTALES, no senales. El codigo que arma el
     condicionante es IDENTICO al del E3 (sum del target y las 4 clases del
     cupo CH2); lo que cambia es el significado, porque cambiaron los labels:
     cond[0] pasa a ser C de la formula y cond[1] los carbonos CH2.
"""
import numpy as np
import torch
from rdkit import Chem
from torch.utils.data import Dataset

IDX_CH2 = [1, 5, 9, 12]   # CH2, CH2-O, CH2-N, =CH2 (orden de config/db.yaml)


class NMRTwoSetsDatasetJ(Dataset):
    def __init__(self, peaks_ch_path, peaks_13c_path, labels_path, smiles_path,
                 norm_cfg, peak_features=5):
        if int(peak_features) not in (4, 5):
            raise ValueError(
                f"peak_features debe ser 4 (control) o 5 (con degeneracion), "
                f"recibido: {peak_features!r}")
        self.peak_features = int(peak_features)

        self.labels = np.load(labels_path).astype(np.float32)
        self.smiles = np.load(smiles_path, allow_pickle=True)

        npz_ch = np.load(peaks_ch_path)
        peaks_ch = npz_ch["peaks"].astype(np.float32)
        self.mask_ch = npz_ch["peaks_mask"].astype(np.float32)

        n_cols = peaks_ch.shape[2]
        if n_cols < self.peak_features:
            raise ValueError(
                f"El .npz tiene {n_cols} columnas por pico pero se pidieron "
                f"peak_features={self.peak_features}. Con peak_features=5 hace "
                f"falta el archivo con degeneracion (peaks_pkl_deg_*.npz); "
                f"entrenar con una columna de ceros haciendose pasar por la "
                f"degeneracion daria un control disfrazado de experimento.")

        npz_c13 = np.load(peaks_13c_path)
        peaks_13c = npz_c13["peaks_13c"].astype(np.float32)
        self.mask_13c = npz_c13["mask_13c"].astype(np.float32)

        # --- normalizacion min-max desde el config (regla dura 3) ---
        c_min, c_max = float(norm_cfg["c13_ppm_min"]), float(norm_cfg["c13_ppm_max"])
        h_min, h_max = float(norm_cfg["h1_ppm_min"]), float(norm_cfg["h1_ppm_max"])
        amp0_scale = float(norm_cfg["amp_ch0_scale"])
        peaks_ch[:, :, 0] = (peaks_ch[:, :, 0] - c_min) / (c_max - c_min)
        peaks_ch[:, :, 1] = (peaks_ch[:, :, 1] - h_min) / (h_max - h_min)
        peaks_ch[:, :, 2] = peaks_ch[:, :, 2] / amp0_scale
        # amp_ch1 (col 3) se deja como esta, igual que en el E3.
        if self.peak_features == 5:
            # Sin default: si falta en el config tiene que romper, no elegir un
            # valor a dedo que despues nadie pueda rastrear (regla dura 3).
            deg_scale = float(norm_cfg["degeneracion_scale"])
            peaks_ch[:, :, 4] = peaks_ch[:, :, 4] / deg_scale

        self.peaks_ch = peaks_ch[:, :, :self.peak_features]
        peaks_13c[:, :, 0] = (peaks_13c[:, :, 0] - c_min) / (c_max - c_min)
        self.peaks_13c = peaks_13c

        print("[INFO] Extrayendo formulas moleculares (C,H,N,O,S,Hal)...")
        self.formula_matrix = np.zeros((len(self.smiles), 6), dtype=np.float32)
        for i, smi in enumerate(self.smiles):
            mol = Chem.MolFromSmiles(str(smi))
            if mol:
                mol = Chem.AddHs(mol)
                nums = [a.GetAtomicNum() for a in mol.GetAtoms()]
                self.formula_matrix[i] = [
                    sum(1 for z in nums if z == 6),
                    sum(1 for z in nums if z == 1),
                    sum(1 for z in nums if z == 7),
                    sum(1 for z in nums if z == 8),
                    sum(1 for z in nums if z == 16),
                    sum(1 for z in nums if z in (9, 17, 35, 53)),
                ]
        print(f"[INFO] Formulas cargadas. peak_features={self.peak_features}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        peaks_ch = torch.tensor(self.peaks_ch[idx], dtype=torch.float32)
        mask_ch = torch.tensor(self.mask_ch[idx], dtype=torch.float32)
        peaks_13c = torch.tensor(self.peaks_13c[idx], dtype=torch.float32)
        mask_13c = torch.tensor(self.mask_13c[idx], dtype=torch.float32)

        target_vec = self.labels[idx]
        # Mismo codigo que el E3. Con los labels de carbonos totales, total_c
        # vale C de la formula y total_ch2 los carbonos CH2 -- ambos siguen
        # siendo observables: C sale de la FM (exacto, sin error de lectura) y
        # los CH2 de sumar integrales de los crosspeaks de fase negativa.
        total_c = np.sum(target_vec).astype(np.float32)
        total_ch2 = sum(target_vec[i] for i in IDX_CH2)
        cond_data = [total_c, np.float32(total_ch2)] + self.formula_matrix[idx].tolist()
        cond_tensor = torch.tensor(cond_data, dtype=torch.float32)

        return (peaks_ch, mask_ch, peaks_13c, mask_13c, cond_tensor), torch.tensor(target_vec)
```

- [ ] **Step 4: Correr el test y verificar que pasa**

```bash
cd experiments/J_carbonos_totales && python tests/test_dataset_j.py
```

Esperado: `>>> DATASET J OK <<<` con 6 líneas `[OK]`.

- [ ] **Step 5: Commit**

```bash
git add experiments/J_carbonos_totales/dataset_j.py experiments/J_carbonos_totales/tests/test_dataset_j.py
git commit -m "exp J: dataset_j.py -- 5a feature normalizada, recorte por peak_features, cond de carbonos"
```

---

## Task 5: `train.py`, `evaluate.py`, los dos configs y el `.sh`

**Files:**
- Create: `experiments/J_carbonos_totales/train.py` (copia del E3, imports adaptados)
- Create: `experiments/J_carbonos_totales/evaluate.py` (copia del E3, imports adaptados)
- Create: `experiments/J_carbonos_totales/config_j_a.yaml`, `config_j_0.yaml`
- Create: `experiments/J_carbonos_totales/run_train_j.sh`
- Test: `experiments/J_carbonos_totales/tests/test_configs_j.py`

**Interfaces:**
- Consumes: `NMRTwoSetsDatasetJ` (Task 4), `NMR_SetTransformerJ` (Task 3), y los cuatro módulos
  copiados (Task 3).
- Produces:
  - `build_model(cfg, num_classes=19) -> NMR_SetTransformerJ` — lee `cfg['model']['peak_features']`.
  - Los dos `.yaml` que consumen las corridas.
  - Archivos `expJ_<jobid>.out` / `.err` en el cluster.

- [ ] **Step 1: Copiar y adaptar `train.py`**

```bash
cp experiments/E3_dos_conjuntos/train.py experiments/J_carbonos_totales/train.py
```

Después, en `experiments/J_carbonos_totales/train.py` hacer exactamente estos seis cambios (a)-(f):

**(a)** Reemplazar el docstring del módulo (líneas 2-8) por:

```python
"""
train.py -- Exp J: entrena el Set Transformer contra el vector de CARBONOS
TOTALES (no de senales). Copia de experiments/E3_dos_conjuntos/train.py con
tres cambios: importa dataset_j/model_j, pasa peak_features al dataset y al
modelo, y nada mas. Loss, scheduler, split congelado y seed son identicos --
si alguno cambiara, J dejaria de ser comparable consigo mismo entre corridas.
"""
```

**(b)** Reemplazar la línea `from dataset_e3 import NMRTwoSetsDataset` por:

```python
from dataset_j import NMRTwoSetsDatasetJ
```

**(c)** Reemplazar la función `build_model` entera por:

```python
def build_model(cfg, num_classes=19):
    arch = cfg['model']['arch']
    if arch != 'settransformer':
        raise ValueError(
            f"model.arch desconocido: {arch!r}. El Exp J solo usa "
            f"'settransformer' (la variante deepsets quedo en el Exp E).")
    from model_j_settransformer import NMR_SetTransformerJ
    m = cfg['model']
    return NMR_SetTransformerJ(
        num_classes=num_classes,
        peak_features=int(m.get('peak_features', 5)),
        d_model=int(m.get('d_model', 64)),
        n_heads=int(m.get('n_heads', 4)),
        n_layers=int(m.get('n_layers', 2)),
        n_seeds=int(m.get('n_seeds', 1)),
        fusion_hidden=tuple(m.get('fusion_hidden', (128, 64))),
    )
```

**(d)** Reemplazar la instanciación del dataset (las dos líneas que empiezan con
`full_dataset = NMRTwoSetsDataset(`) por:

```python
    full_dataset = NMRTwoSetsDatasetJ(
        str(peaks_ch), str(peaks_13c), str(labels_path), str(smiles_path),
        cfg['normalization'], peak_features=int(cfg['model'].get('peak_features', 5)))
```

**(e)** Reemplazar la línea del `print` de encabezado
(`print(f"--- ENTRENAMIENTO EXP E FASE 3 ...")`) por:

```python
    print(f"--- ENTRENAMIENTO EXP J (carbonos totales): {cfg['experiment_name']} ---")
```

**(f)** Cambiar el default del argumento `--config` al final del archivo, de
`default="config_deepsets.yaml"` a `default="config_j_a.yaml"`.

- [ ] **Step 2: Copiar y adaptar `evaluate.py`**

```bash
cp experiments/E3_dos_conjuntos/evaluate.py experiments/J_carbonos_totales/evaluate.py
```

Después, en `experiments/J_carbonos_totales/evaluate.py`:

**(a)** Reemplazar la línea `from dataset_e3 import NMRTwoSetsDataset` por:

```python
    from dataset_j import NMRTwoSetsDatasetJ
```

(mantener la indentación: está dentro de la función `evaluate`, es un import perezoso.)

**(b)** Reemplazar la instanciación `full_dataset = NMRTwoSetsDataset(...)` por:

```python
    full_dataset = NMRTwoSetsDatasetJ(
        str(peaks_ch), str(peaks_13c), str(labels_path), str(smiles_path),
        cfg["normalization"], peak_features=int(cfg["model"].get("peak_features", 5)))
```

**(c)** Reemplazar la línea del encabezado
(`print("  EVALUACION EXP E FASE 3 (dos conjuntos) - SPLIT CONGELADO")`) por:

```python
    print("  EVALUACION EXP J (carbonos totales) - SPLIT CONGELADO")
```

- [ ] **Step 3: Escribir los dos configs**

Crear `experiments/J_carbonos_totales/config_j_a.yaml`:

```yaml
# experiments/J_carbonos_totales/config_j_a.yaml
#
# Exp J-A: corrida EXPERIMENTAL. Vector de carbonos totales + degeneracion
# (derivada de la integracion de protones) como 5a feature del crosspeak.
#
# Su par de control es config_j_0.yaml, que lee EXACTAMENTE los mismos datos
# y difiere solo en peak_features: 4. Todo lo demas tiene que ser identico
# entre los dos o la comparacion no mide lo que dice medir (regla dura 8).
experiment_name: "nmr_202k_j_carbonos_deg"

paths:
  base_dir: "${NMR_DATA_DIR:-/home/lpassaglia.iquir/DB_200k}"
  peaks_ch_filename: "peaks_pkl_deg_202465.npz"
  peaks_13c_filename: "peaks_13c_202465.npz"
  labels_filename: "vectors_19v_totales_202465.npy"
  smiles_filename: "smiles_202465.npy"
  checkpoint_dir: "checkpoints_J_carbonos_deg"
  val_indices_filename: "val_indices_frozen.npy"

model:
  arch: "settransformer"
  peak_features: 5          # <-- lo unico que distingue esta corrida del control
  d_model: 64
  n_heads: 4
  n_layers: 2
  n_seeds: 1

normalization:
  c13_ppm_min: 0
  c13_ppm_max: 220
  h1_ppm_min: -1
  h1_ppm_max: 15
  amp_ch0_scale: 3.0
  # Degeneracion medida: 80.5% vale 1, 18.1% vale 2, maximo observado 12.
  # Escala 4.0 deja los valores comunes en 0.25 y 0.5.
  degeneracion_scale: 4.0

hyperparameters:
  batch_size: 64
  learning_rate: 0.001
  epochs: 100
  seed: 42
  scheduler:
    patience: 8
    factor: 0.7

system:
  device: "${NMR_DEVICE:-auto}"
  num_workers: 0
  pin_memory: true
```

Crear `experiments/J_carbonos_totales/config_j_0.yaml`:

```yaml
# experiments/J_carbonos_totales/config_j_0.yaml
#
# Exp J-0: corrida de CONTROL. Mismo vector de carbonos totales, mismos
# archivos de datos, pero SIN la degeneracion (peak_features: 4 recorta la 5a
# columna en el dataset).
#
# Sin este control, si J-A funciona no se puede afirmar que la integracion fue
# necesaria: podria ser que la restriccion de suma de la FM -- que en el
# esquema de carbonos totales es exacta -- ya alcanzara sola.
experiment_name: "nmr_202k_j_carbonos_ctrl"

paths:
  base_dir: "${NMR_DATA_DIR:-/home/lpassaglia.iquir/DB_200k}"
  peaks_ch_filename: "peaks_pkl_deg_202465.npz"
  peaks_13c_filename: "peaks_13c_202465.npz"
  labels_filename: "vectors_19v_totales_202465.npy"
  smiles_filename: "smiles_202465.npy"
  checkpoint_dir: "checkpoints_J_carbonos_ctrl"
  val_indices_filename: "val_indices_frozen.npy"

model:
  arch: "settransformer"
  peak_features: 4          # <-- control: sin degeneracion
  d_model: 64
  n_heads: 4
  n_layers: 2
  n_seeds: 1

normalization:
  c13_ppm_min: 0
  c13_ppm_max: 220
  h1_ppm_min: -1
  h1_ppm_max: 15
  amp_ch0_scale: 3.0
  degeneracion_scale: 4.0   # sin efecto con peak_features 4; se deja para que
                            # los dos configs sean diffeables linea por linea

hyperparameters:
  batch_size: 64
  learning_rate: 0.001
  epochs: 100
  seed: 42
  scheduler:
    patience: 8
    factor: 0.7

system:
  device: "${NMR_DEVICE:-auto}"
  num_workers: 0
  pin_memory: true
```

- [ ] **Step 4: Escribir el `.sh` de SLURM**

Crear `experiments/J_carbonos_totales/run_train_j.sh`:

```bash
#!/bin/bash
#SBATCH --job-name=expJ
#SBATCH --partition=gpua10_hi
#SBATCH --output=expJ_%j.out
#SBATCH --error=expJ_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:1

# Exp J -- una corrida del estudio de carbonos totales en login-1 / A10.
# Entrena Y evalua el mismo config en UN solo job: menos sbatch, y elimina la
# clase de error "evalue un checkpoint que todavia no habia terminado".
#
# Uso:
#   sbatch run_train_j.sh config_j_a.yaml     # experimental (con degeneracion)
#   sbatch run_train_j.sh config_j_0.yaml     # control (sin degeneracion)
#
# Las dos de una:
#   for c in config_j_a.yaml config_j_0.yaml; do sbatch run_train_j.sh "$c"; done
#
# ANTES de lanzar: subir por scp al base_dir del cluster los dos archivos que
# se generan localmente con prep/ --
#   vectors_19v_totales_202465.npy
#   peaks_pkl_deg_202465.npz

source /home/lpassaglia.iquir/anaconda3/etc/profile.d/conda.sh
conda activate /home/lpassaglia.iquir/anaconda3/envs/NMR_env

# Ajustar esta ruta a donde hayas clonado el repo en el cluster.
cd ~/nmr-hsqc-to-vector-/experiments/J_carbonos_totales

CONFIG="${1:?Falta el config, ej: sbatch run_train_j.sh config_j_a.yaml}"

echo "=== EXP J | CONFIG: $CONFIG ==="

echo "=== FASE 1/2: TRAIN ==="
python -u train.py --config "$CONFIG"
if [ $? -ne 0 ]; then
    echo "[ABORT] train.py fallo -- no se evalua (evitar reportar la EMA de un checkpoint viejo)"
    exit 1
fi

echo "=== FASE 2/2: EVAL ==="
# --oraculo all: cruda + asistida v1 + asistida v2, tabla de 3 vias.
python -u evaluate.py --config "$CONFIG" --oraculo all --batch-size 256
```

- [ ] **Step 5: Escribir el test de invariantes de los configs**

Crear `experiments/J_carbonos_totales/tests/test_configs_j.py`:

```python
# coding: ascii
"""J-A y J-0 tienen que diferir SOLO en peak_features (y en las claves de
identidad). Si se cuela cualquier otra diferencia -- epocas, seed, scheduler,
archivos de datos -- la comparacion entre las dos deja de medir el aporte de la
integracion y pasa a medir esa otra cosa (regla dura 8)."""
import sys
from pathlib import Path

import yaml

_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_DIR))

IDENTIDAD = {"experiment_name", "paths.checkpoint_dir"}
ESPERADA = {"model.peak_features"}


def _plano(d, pre=""):
    out = {}
    for k, v in d.items():
        clave = f"{pre}{k}"
        if isinstance(v, dict):
            out.update(_plano(v, clave + "."))
        else:
            out[clave] = v
    return out


def _cargar(nombre):
    with open(_DIR / nombre, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_difieren_solo_en_peak_features():
    a, c = _plano(_cargar("config_j_a.yaml")), _plano(_cargar("config_j_0.yaml"))
    claves = set(a) | set(c)
    distintas = {k for k in claves
                 if k not in IDENTIDAD and a.get(k, "<falta>") != c.get(k, "<falta>")}
    assert distintas == ESPERADA, f"difieren en {sorted(distintas)}"
    print("[OK] J-A y J-0 difieren solo en model.peak_features")


def test_peak_features_correctos():
    assert _cargar("config_j_a.yaml")["model"]["peak_features"] == 5
    assert _cargar("config_j_0.yaml")["model"]["peak_features"] == 4
    print("[OK] J-A = 5 features, J-0 = 4 features")


def test_leen_los_mismos_archivos_de_datos():
    """Que compartan el .npz es lo que garantiza que no puedan desincronizarse."""
    a, c = _cargar("config_j_a.yaml")["paths"], _cargar("config_j_0.yaml")["paths"]
    for k in ("peaks_ch_filename", "peaks_13c_filename", "labels_filename",
              "smiles_filename", "val_indices_filename", "base_dir"):
        assert a[k] == c[k], (k, a[k], c[k])
    print("[OK] los dos configs leen exactamente los mismos archivos")


def test_identidades_unicas():
    a, c = _cargar("config_j_a.yaml"), _cargar("config_j_0.yaml")
    assert a["experiment_name"] != c["experiment_name"]
    assert a["paths"]["checkpoint_dir"] != c["paths"]["checkpoint_dir"]
    print("[OK] experiment_name y checkpoint_dir distintos (no se pisan)")


def test_reglas_duras():
    for nombre in ("config_j_a.yaml", "config_j_0.yaml"):
        cfg = _cargar(nombre)
        assert cfg["system"]["num_workers"] == 0, nombre        # regla dura 1
        assert cfg["hyperparameters"]["scheduler"]["patience"] == 8, nombre   # regla 6
        assert cfg["hyperparameters"]["scheduler"]["factor"] == 0.7, nombre   # regla 6
        assert cfg["hyperparameters"]["epochs"] == 100, nombre  # regla dura 8
        assert cfg["hyperparameters"]["seed"] == 42, nombre     # regla dura 8
        assert cfg["paths"]["val_indices_filename"] == "val_indices_frozen.npy", nombre
    print("[OK] reglas duras 1, 6 y 8 en los dos configs")


def test_labels_nuevos_no_pisan_los_viejos():
    """El checkpoint congelado sigue usando vectors_13c_19v_202465.npy y
    peaks_pkl_202465.npz: Exp J no puede apuntar a esos nombres."""
    for nombre in ("config_j_a.yaml", "config_j_0.yaml"):
        p = _cargar(nombre)["paths"]
        assert p["labels_filename"] == "vectors_19v_totales_202465.npy", nombre
        assert p["peaks_ch_filename"] == "peaks_pkl_deg_202465.npz", nombre
    print("[OK] los configs apuntan a los archivos NUEVOS, no a los del checkpoint congelado")


def test_degeneracion_scale_presente():
    """Con peak_features 5 el dataset lo exige sin default (regla dura 3)."""
    assert _cargar("config_j_a.yaml")["normalization"]["degeneracion_scale"] == 4.0
    print("[OK] degeneracion_scale presente en J-A")


if __name__ == "__main__":
    test_difieren_solo_en_peak_features()
    test_peak_features_correctos()
    test_leen_los_mismos_archivos_de_datos()
    test_identidades_unicas()
    test_reglas_duras()
    test_labels_nuevos_no_pisan_los_viejos()
    test_degeneracion_scale_presente()
    print("\n>>> CONFIGS J OK <<<")
```

- [ ] **Step 6: Correr el test de configs**

```bash
cd experiments/J_carbonos_totales && python tests/test_configs_j.py
```

Esperado: `>>> CONFIGS J OK <<<` con 7 líneas `[OK]`.

- [ ] **Step 7: Verificar el `.sh` y que `train.py`/`evaluate.py` compilan**

```bash
bash -n experiments/J_carbonos_totales/run_train_j.sh && echo "SINTAXIS SH OK"
```

Esperado: `SINTAXIS SH OK`.

```bash
grep -n "gres=gpu:1" experiments/J_carbonos_totales/run_train_j.sh; grep -c "gpus=" experiments/J_carbonos_totales/run_train_j.sh
```

Esperado: la primera muestra `#SBATCH --gres=gpu:1`; el `grep -c` devuelve `0` (regla dura 2).

```bash
cd experiments/J_carbonos_totales && python -m py_compile train.py evaluate.py && echo "COMPILAN OK"
```

Esperado: `COMPILAN OK`.

- [ ] **Step 8: Verificar que `build_model` respeta `peak_features`**

```bash
cd experiments/J_carbonos_totales && python -c "
import yaml
from train import build_model
for f, esperado in [('config_j_a.yaml', 5), ('config_j_0.yaml', 4)]:
    cfg = yaml.safe_load(open(f, encoding='utf-8'))
    m = build_model(cfg, num_classes=19)
    assert m.proj_ch.in_features == esperado, (f, m.proj_ch.in_features)
    n = sum(p.numel() for p in m.parameters())
    print(f'[OK] {f}: proj_ch.in_features={m.proj_ch.in_features} params={n:,}')
"
```

Esperado: dos líneas `[OK]`, con `in_features=5` para J-A y `4` para J-0.

- [ ] **Step 9: Commit**

```bash
git add experiments/J_carbonos_totales/
git commit -m "exp J: train/evaluate adaptados, los dos configs (J-A y J-0) y run_train_j.sh"
```

---

## Task 6: README, RATIONALE y sección placeholder en `RESULTS.md`

**Files:**
- Create: `experiments/J_carbonos_totales/README.md`
- Create: `experiments/J_carbonos_totales/RATIONALE.md`
- Modify: `docs/Runs/RESULTS.md` (fila en la tabla resumen + sección nueva al final)

**Interfaces:**
- Consumes: todo lo anterior. No produce nada que consuman tareas posteriores.

- [ ] **Step 1: Escribir el README**

Crear `experiments/J_carbonos_totales/README.md`:

````markdown
# Exp J — Vector de carbonos totales

**Qué es:** el vector de 19 clases pasa a contar **carbonos**, no señales. El benceno da
`=CH/Ar = 6` en vez de 1, y `sum(vector) == C` de la fórmula molecular. La degeneración de cada
señal —derivada de la integración de protones— entra como 5ª feature de cada crosspeak.

**Qué NO es:** un reemplazo del checkpoint congelado. Exp J entrena un modelo para un target
distinto; los dos coexisten y sus EMAs **no son comparables**.

Spec: `docs/superpowers/specs/2026-08-12-vector-carbonos-totales-design.md`.

## Por qué

El vector viejo choca con la FM: dice 1 donde la fórmula dice C6. Para el generador de estructuras
aguas abajo hace falta el conteo real de carbonos. Afecta al **62 %** del dataset, con 3 carbonos
escondidos en promedio.

La integración lo resuelve: en el tolueno se ven 3 señales aromáticas CH, pero integran 2H / 2H / 1H
— de ahí salen los 5 carbonos aromáticos protonados.

## Las dos corridas

| Corrida | Config | `peak_features` | Qué mide |
|---|---|---|---|
| **J-A** | `config_j_a.yaml` | 5 | La propuesta completa (con degeneración) |
| **J-0** | `config_j_0.yaml` | 4 | Control: cuánto se logra **sin** integración |

**La diferencia entre las dos es el resultado.** Los dos configs leen los **mismos** archivos de
datos; J-0 solo recorta la 5ª columna. Si J-A ≈ J-0, la restricción de suma de la FM ya alcanzaba y
la integración es redundante — resultado igual de válido, y te ahorra pedirla en el laboratorio.

## Techo del experimento

Medido sobre 20 000 moléculas: **97,7 %**. Se descompone en 90,3 % de moléculas sin cuaternarios
escondidos (la integración resuelve toda la simetría) + 7,4 % donde los hay pero caen en una sola
clase Cq y la suma de la FM los ubica igual. Queda un 2,3 % genuinamente ambiguo.

Los cuaternarios no tienen integración: en un ¹³C real las intensidades no son cuantitativas.

## Cómo se corre

**1. Generar los datos (local, en la PC de Lucas — no en el cluster):**

```bash
python prep/make_labels_totales.py --config prep/config_prep.yaml
```

```bash
python prep/make_peaks_degeneracion.py --config prep/config_prep.yaml
```

El primero corre un **gate**: si el clasificador no reproduce exactamente los labels viejos, aborta
sin escribir nada. Genera `vectors_19v_totales_202465.npy` y `peaks_pkl_deg_202465.npz` en
`E:/Proyectos/SciTrix/ScitrixDB/DB_nmr_to_vector/202K_suma/`.

**2. Subir los dos archivos al cluster:**

```bash
scp vectors_19v_totales_202465.npy peaks_pkl_deg_202465.npz lpassaglia.iquir@login-1:/home/lpassaglia.iquir/DB_200k/
```

**3. Smoke tests antes de cualquier `sbatch` (regla dura 5):**

```bash
python tests/test_forward_j.py
```

```bash
python tests/test_dataset_j.py
```

```bash
python tests/test_configs_j.py
```

**4. Lanzar las dos corridas (desde `experiments/J_carbonos_totales/` en login-1):**

```bash
for c in config_j_a.yaml config_j_0.yaml; do sbatch run_train_j.sh "$c"; done
```

Cada job entrena Y evalúa, y deja todo en un `expJ_<jobid>.out`.

## Cluster

**login-1 / A10 ("capitán")**, env `NMR_env`, partición `gpua10_hi`, `base_dir`
`/home/lpassaglia.iquir/DB_200k`. Las dos corridas van al **mismo** cluster (regla dura 8) — mezclar
hardware metería la varianza A10-vs-XPU (~0,8 pp) dentro de la comparación.

No hay script para Clementina: el cupo QOS del grupo está trabado ahí, y Exp I ya ocupa esa cola.

## Métrica

**Primaria: EMA asistida v2** sobre el vector de carbonos, con `evaluate.py --oraculo all`.

> **Estos EMA no son comparables con el 92,14 % del checkpoint congelado.** Es otro target, más
> difícil por construcción: la suma promedio pasa de 11,4 a 13,3 y hay que acertar la degeneración
> además de la clase. El punto de comparación es **J-0**, no el modelo viejo.

## Tests

| Archivo | Qué verifica | Requiere |
|---|---|---|
| `prep/tests/test_labels_totales.py` | Benceno=6, tolueno, suma == C | rdkit, numpy |
| `prep/tests/test_peaks_degeneracion.py` | La degeneración cuenta en vez de descartar | rdkit, numpy |
| `tests/test_forward_j.py` | Forward con 4 y con 5 features (regla dura 5) | torch |
| `tests/test_dataset_j.py` | Normalización, recorte y `cond` | torch, rdkit |
| `tests/test_configs_j.py` | J-A y J-0 difieren solo en `peak_features` | PyYAML |
````

- [ ] **Step 2: Escribir el RATIONALE**

Crear `experiments/J_carbonos_totales/RATIONALE.md`:

```markdown
# Exp J — decisiones de diseño

## Por qué una carpeta nueva y no una variante dentro del E3

El Exp I (sweep de hiperparámetros) vive **dentro** de `E3_dos_conjuntos/` justamente porque su
validez depende de usar el `train.py` bit-a-bit idéntico al del checkpoint congelado.

Exp J es el caso opuesto: los labels, el dataset y la semántica del condicionante divergen de
verdad. Copiar es lo correcto acá — es la convención estándar del proyecto (carpeta autocontenida)
y evita que un cambio pensado para J rompa el E3 en producción.

## Por qué degeneración y no la integral cruda

El químico carga **H** (2, 3, 6…), que es lo que lee del ¹H integrado. El pipeline divide por los
H-por-carbono —que ya conoce de la multiplicidad— y le pasa al modelo el **número de carbonos
equivalentes**.

Hacer la división afuera es inyectar conocimiento químico explícito como condicionante, que es la
palanca que más rindió en toda la serie histórica (CH2, Fórmula Molecular). Una división no es una
operación natural para una red; no hay razón para hacérsela aprender.

## Por qué los cuaternarios no llevan degeneración

En un ¹³C real las intensidades no son cuantitativas (NOE, tiempos de relajación distintos). Darle
al modelo la degeneración de un carbono sin H sería entrenarlo con información que no va a existir
en el espectro experimental — el modelo aprendería a depender de algo que en inferencia no está.

Lo único que acota los cuaternarios es la suma total de la FM, y eso es exactamente lo que produce
el 2,3 % de ambigüedad residual del techo.

## Por qué un solo archivo de modelo con `peak_features` configurable

Dos archivos casi idénticos se desincronizan sin avisar. Si J-A y J-0 corrieran sobre modelos
distintos que divergieron en algo, la comparación entre ambas dejaría de medir el aporte de la
integración y pasaría a medir esa divergencia — sin que nada tire error.

Por el mismo motivo los dos configs leen el **mismo** `.npz`: el control recorta la columna en el
dataset, no usa otro archivo.

## Por qué el gate de verificación del clasificador

`make_labels_totales.py` regenera los labels **viejos** y los compara contra
`vectors_13c_19v_202465.npy` antes de escribir nada. Si no coinciden al 100 %, aborta.

Sin ese gate, un clasificador sutilmente distinto produciría un ground truth nuevo corrupto sin
tirar ningún error — el modo de falla exacto que la regla dura 7 existe para prevenir. Verificado:
100,000 % sobre 5 000 moléculas, 0 discrepancias.

## Por qué `oraculo.py` se copia sin cambios

La lógica de forzar `sum(pred) == total` y el cupo de CH2 es idéntica. Lo único que cambia es que
`total` ahora vale C de la fórmula en vez del número de señales — un cambio de datos, no de
algoritmo.

De hecho el esquema nuevo es **más robusto**: antes `total` se contaba del espectro y dos señales
solapadas lo arruinaban; ahora sale de la FM, exacto.
```

- [ ] **Step 3: Agregar la fila a la tabla resumen de `RESULTS.md`**

En `docs/Runs/RESULTS.md`, después de la fila del Exp I (la última de la tabla de arriba), agregar:

```markdown
| Exp J — vector de carbonos totales (Set Transformer) | n/a (sin imagen) | 19 | 202k | none | PENDIENTE | PENDIENTE | PENDIENTE | **TARGET NUEVO, EMA NO comparable con las filas de arriba**: el vector cuenta carbonos, no senales (benceno = 6, no 1). 2 corridas: J-A (con degeneracion de la integracion) y J-0 (control). Techo medido 97.7%. **Jobs NO lanzados todavia.** Ver seccion |
```

- [ ] **Step 4: Agregar la sección al final de `RESULTS.md`**

En `docs/Runs/RESULTS.md`, justo **antes** de la línea `<!-- Template for next entries`, insertar:

```markdown
## Exp J — vector de carbonos totales (simetría resuelta por integración)

- **Fecha:** 2026-08-12 (preparación) · **SLURM:** PENDIENTE ·
  **Configs:** `experiments/J_carbonos_totales/config_j_{a,0}.yaml` ·
  **Cluster:** login-1 / A10.
- **Estado: los jobs NO se lanzaron todavía.** Los números se completan cuando corran.

### ⚠️ Estos EMA no son comparables con el resto de la tabla

El target cambió. Hasta el Exp I, el vector de 19 clases contaba **señales** (carbonos equivalentes
por simetría colapsados: benceno = 1 en `=CH/Ar`). Exp J cuenta **carbonos totales** (benceno = 6),
así que `sum(vector) == C` de la fórmula molecular. Es un problema más difícil por construcción: la
suma promedio pasa de 11,4 a 13,3 y hay que acertar la degeneración además de la clase.

**El punto de comparación de J-A es J-0, no el 92,14 % del checkpoint congelado.**

### Por qué

El vector de señales choca con la FM: dice 1 donde la fórmula dice C6, y el generador de estructuras
aguas abajo necesita el conteo real de carbonos. Medido sobre 20 000 moléculas: **62 % del dataset
tiene simetría**, con 3 carbonos escondidos en promedio (máximo 26).

La integración de protones revela la degeneración — en el tolueno, 3 señales aromáticas CH que
integran 2H / 2H / 1H son 5 carbonos.

### Techo del experimento: 97,7 %

| Caso | Fracción | Por qué |
|---|---|---|
| Sin cuaternarios escondidos | 90,3 % | La integración resuelve toda la simetría |
| Cuaternarios escondidos, una sola clase Cq | 7,4 % | La suma de la FM los ubica igual |
| Cuaternarios escondidos en varias clases Cq | 2,3 % | Ambigüedad real |

De los carbonos escondidos, el **91,4 %** está en carbonos protonados (alcanzables por integración).
Los cuaternarios no tienen integración: en un ¹³C real las intensidades no son cuantitativas.

### Las dos corridas

| Corrida | `peak_features` | Best Val Loss | EMA cruda | EMA asist v2 | min |
|---|---|---|---|---|---|
| J-A (con degeneración) | 5 | PENDIENTE | PENDIENTE | PENDIENTE | PENDIENTE |
| J-0 (control, sin) | 4 | PENDIENTE | PENDIENTE | PENDIENTE | PENDIENTE |

Los dos configs leen los mismos archivos de datos y difieren **solo** en `peak_features` (verificado
por `tests/test_configs_j.py`). Invariantes: val congelado, 100 épocas, `ConstrainedMSELoss`,
scheduler `patience=8/factor=0.7`, `num_workers=0`, seed 42, mismo cluster.

### Lecturas posibles (fijadas antes de ver los resultados)

| Resultado | Conclusión |
|---|---|
| J-A ≫ J-0 | La integración es la que hace el trabajo. Diseño validado. |
| J-A ≈ J-0 | La suma exacta de la FM ya alcanzaba; la integración es redundante. Ahorra pedirla en el laboratorio. |
| Ambas bajas | El vector de carbonos es genuinamente más difícil. Se documenta y se para. |

- **Verificado antes de entrenar:** el clasificador portado reproduce los labels históricos al
  **100,000 %** sobre 5 000 moléculas (0 discrepancias), y el vector sin colapso suma exactamente C
  también al 100 %.
- **Takeaway:** PENDIENTE. Spec:
  `docs/superpowers/specs/2026-08-12-vector-carbonos-totales-design.md`. Plan:
  `docs/superpowers/plans/2026-08-12-vector-carbonos-totales.md`. Cómo correrlo:
  `experiments/J_carbonos_totales/README.md`.

---
```

- [ ] **Step 5: Correr toda la batería de tests una última vez**

```bash
cd experiments/J_carbonos_totales && python prep/tests/test_labels_totales.py && python prep/tests/test_peaks_degeneracion.py && python tests/test_forward_j.py && python tests/test_dataset_j.py && python tests/test_configs_j.py
```

Esperado: los cinco imprimen su `>>> ... OK <<<`.

- [ ] **Step 6: Verificar que el E3 sigue intacto**

Exp J no puede haber tocado nada del experimento en producción:

```bash
git status --short experiments/E3_dos_conjuntos/ && git diff --stat HEAD -- experiments/E3_dos_conjuntos/
```

Esperado: **sin salida** en los dos comandos (ni archivos modificados ni diff).

- [ ] **Step 7: Commit**

```bash
git add experiments/J_carbonos_totales/README.md experiments/J_carbonos_totales/RATIONALE.md docs/Runs/RESULTS.md
git commit -m "exp J: README, RATIONALE y seccion placeholder en RESULTS.md"
```

---

## Checklist final para Lucas (lo que queda a mano)

Nada de esto lo puede hacer Claude Code — requiere SSH al cluster.

1. Correr los dos scripts de `prep/` localmente (~5 min) y confirmar que el gate da 100 %.
2. `scp` de `vectors_19v_totales_202465.npy` y `peaks_pkl_deg_202465.npz` a
   `/home/lpassaglia.iquir/DB_200k/` en login-1.
3. `git pull` en el cluster y ajustar el `cd` de `run_train_j.sh` si la ruta del repo difiere.
4. Correr los smoke tests en el cluster con `NMR_env` activado.
5. Lanzar: `for c in config_j_a.yaml config_j_0.yaml; do sbatch run_train_j.sh "$c"; done`
6. Bajar los `.out`, completar la tabla del Exp J en `RESULTS.md` y escribir el takeaway con lo que
   salga — incluida la lectura "J-A ≈ J-0" si es la que toca.
