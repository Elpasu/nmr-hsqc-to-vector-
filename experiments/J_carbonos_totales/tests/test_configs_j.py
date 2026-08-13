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
