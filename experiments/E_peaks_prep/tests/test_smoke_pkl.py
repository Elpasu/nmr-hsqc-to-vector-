# coding: ascii
"""
Smoke test OFFLINE de Exp E Fase 1b (rule 5 de CLAUDE.md) -- construye un
mini pkl + mini arrays de smiles/mol_ids/labels en un directorio temporal
(3 moleculas), corre extract_peaks_pkl.main() sobre ellos, y confirma que
el .npz de salida y el reporte de validacion tienen las formas y valores
esperados. No toca ningun dato real del cluster.

Requiere rdkit + pyyaml -- correr local antes de procesar los pkl reales:
    python tests/test_smoke_pkl.py
"""
import pickle
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import yaml

from extract_peaks_pkl import main

CLASS_NAMES = [
    "CH3", "CH2", "CH", "Cq", "CH3-O", "CH2-O", "CH-O", "Cq-O",
    "CH3-N", "CH2-N", "CH-N", "Cq-N", "=CH2", "=CH/Ar", "Cqsp2",
    "Aldeh", "Imina", "C-2X", "C-3X",
]


def test_pipeline_end_to_end_with_synthetic_files():
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)

        # 2 moleculas "144k" (etanol, metano) + 1 molecula "58k" (etanol de nuevo)
        smiles_144 = np.array(["CCO", "C"], dtype=object)
        mol_ids_144 = np.array(["mol0", "mol1"], dtype=object)
        smiles_58 = np.array(["CCO"], dtype=object)
        mol_ids_58 = np.array(["mol2"], dtype=object)
        smiles_real = np.concatenate([smiles_144, smiles_58])   # mismo orden -> alineacion OK

        pkl_144 = {
            "mol0": {0: 18.0, 1: 58.0, 3: 1.2, 4: 1.2, 5: 1.2, 6: 3.5, 7: 3.7},  # etanol
            "mol1": {0: -2.0, 1: 0.2, 2: 0.2, 3: 0.2, 4: 0.2},  # metano (CH4)
        }
        pkl_58 = {
            "mol2": {0: 18.0, 1: 58.0, 3: 1.2, 4: 1.2, 5: 1.2, 6: 3.5, 7: 3.7},  # etanol otra vez
        }

        labels = np.zeros((3, 19), dtype=int)
        labels[0, CLASS_NAMES.index("CH3")] = 1
        labels[0, CLASS_NAMES.index("CH2")] = 1
        labels[1, CLASS_NAMES.index("CH3")] = 1   # metano cuenta como 1 entorno visible (aprox para el test)
        labels[2, CLASS_NAMES.index("CH3")] = 1
        labels[2, CLASS_NAMES.index("CH2")] = 1

        np.save(base / "smiles_144.npy", smiles_144, allow_pickle=True)
        np.save(base / "mol_ids_144.npy", mol_ids_144, allow_pickle=True)
        np.save(base / "smiles_58.npy", smiles_58, allow_pickle=True)
        np.save(base / "mol_ids_58.npy", mol_ids_58, allow_pickle=True)
        np.save(base / "smiles_real.npy", smiles_real, allow_pickle=True)
        np.save(base / "labels.npy", labels)
        with open(base / "pkl_144.pkl", "wb") as f:
            pickle.dump(pkl_144, f)
        with open(base / "pkl_58.pkl", "wb") as f:
            pickle.dump(pkl_58, f)

        config = {
            "paths": {
                "base_dir": str(base),
                "pkl_144k": "pkl_144.pkl",
                "mol_ids_144k": "mol_ids_144.npy",
                "smiles_144k": "smiles_144.npy",
                "pkl_58k": "pkl_58.pkl",
                "mol_ids_58k": "mol_ids_58.npy",
                "smiles_58k": "smiles_58.npy",
                "smiles_202465": "smiles_real.npy",
                "labels_202465": "labels.npy",
                "peaks_output_filename": "peaks_out.npz",
            },
            "classes_19v": CLASS_NAMES,
        }
        config_path = base / "config_pkl.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(config, f)

        main(str(config_path))

        with np.load(base / "peaks_out.npz") as out:
            assert out["peaks"].shape[0] == 3
            assert out["peaks_mask"].shape[0] == 3
            # etanol (mol0 y mol2) -> 2 picos cada uno; metano -> 1 pico (un solo C)
            counts = out["peaks_mask"].sum(axis=1)
        assert counts.tolist() == [2, 1, 2], counts.tolist()
        print(f"[OK] test_pipeline_end_to_end_with_synthetic_files -> peaks_mask counts={counts.tolist()}")


if __name__ == "__main__":
    test_pipeline_end_to_end_with_synthetic_files()
    print("\n>>> SMOKE EXP E FASE 1b OK - listo para correr con los pkl reales <<<")
