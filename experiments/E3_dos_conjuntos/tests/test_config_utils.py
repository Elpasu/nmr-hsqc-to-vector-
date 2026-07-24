# coding: ascii
"""Tests de la expansion de variables de entorno en los configs (Fase 3 de la
migracion a Intel XPU, ver docs/MIGRACION_XPU_Clementina_XXI.md).

Motivo: `base_dir` estaba hardcodeado a la ruta de login-1. Para correr el
mismo config en Clementina XXI sin duplicar archivos (decision D7 + regla dura
3 del proyecto), los paths admiten ${VAR:-default}: el .sh de cada cluster
exporta lo suyo y el default preserva el comportamiento historico.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config_utils import expand_env, load_config

VAR = "NMR_TEST_DATA_DIR_XYZ"


def _clear():
    os.environ.pop(VAR, None)


def test_var_seteada_se_expande():
    os.environ[VAR] = "/data/nuevo"
    assert expand_env("${" + VAR + "}") == "/data/nuevo"
    _clear()
    print("[OK] ${VAR} seteada -> se expande")


def test_default_se_usa_si_no_esta_seteada():
    _clear()
    assert expand_env("${" + VAR + ":-/home/viejo}") == "/home/viejo"
    print("[OK] ${VAR:-default} sin setear -> default (preserva login-1)")


def test_var_seteada_gana_al_default():
    os.environ[VAR] = "/data/nuevo"
    assert expand_env("${" + VAR + ":-/home/viejo}") == "/data/nuevo"
    _clear()
    print("[OK] ${VAR:-default} seteada -> gana la variable")


def test_var_sin_default_ni_valor_es_error():
    # No dejarla literal: un path "${VAR}/x.npz" daria un FileNotFoundError
    # confuso mucho despues, en vez de fallar claro al arrancar.
    _clear()
    try:
        expand_env("${" + VAR + "}/peaks.npz")
    except RuntimeError as e:
        assert VAR in str(e), str(e)
        print("[OK] ${VAR} sin valor ni default -> RuntimeError nombrando la var")
    else:
        raise AssertionError("debio fallar en vez de dejar el literal")


def test_string_sin_variables_queda_intacto():
    s = "/home/lpassaglia.iquir/DB_200k"
    assert expand_env(s) == s
    print("[OK] string sin variables -> intacto")


def test_expande_recursivo_en_dict_y_lista():
    _clear()
    cfg = {
        "paths": {"base_dir": "${" + VAR + ":-/viejo}"},
        "lista": ["${" + VAR + ":-/a}", "literal"],
        "num": 42, "flag": True, "nada": None,
    }
    out = expand_env(cfg)
    assert out["paths"]["base_dir"] == "/viejo", out
    assert out["lista"] == ["/a", "literal"], out
    assert out["num"] == 42 and out["flag"] is True and out["nada"] is None, out
    print("[OK] expande recursivo en dict/lista y no toca int/bool/None")


def test_no_muta_la_entrada():
    _clear()
    cfg = {"paths": {"base_dir": "${" + VAR + ":-/viejo}"}}
    original = cfg["paths"]["base_dir"]
    expand_env(cfg)
    assert cfg["paths"]["base_dir"] == original, "expand_env muto la entrada"
    print("[OK] no muta el dict de entrada")


def test_load_config_expande_base_dir_real():
    # El config real de E3 debe seguir apuntando a login-1 si no exportas nada.
    _clear()
    os.environ.pop("NMR_DATA_DIR", None)
    cfg_path = Path(__file__).resolve().parent.parent / "config_settransformer.yaml"
    cfg = load_config(cfg_path)
    base = cfg["paths"]["base_dir"]
    assert "$" not in base, base
    assert base == "/home/lpassaglia.iquir/DB_200k", base
    print(f"[OK] config real sin NMR_DATA_DIR -> {base} (login-1 intacto)")

    os.environ["NMR_DATA_DIR"] = "/data/contrib/pci_78/Lucas/DB_202K"
    cfg = load_config(cfg_path)
    assert cfg["paths"]["base_dir"] == "/data/contrib/pci_78/Lucas/DB_202K", cfg["paths"]
    os.environ.pop("NMR_DATA_DIR", None)
    print("[OK] config real con NMR_DATA_DIR -> ruta de Clementina")


if __name__ == "__main__":
    test_var_seteada_se_expande()
    test_default_se_usa_si_no_esta_seteada()
    test_var_seteada_gana_al_default()
    test_var_sin_default_ni_valor_es_error()
    test_string_sin_variables_queda_intacto()
    test_expande_recursivo_en_dict_y_lista()
    test_no_muta_la_entrada()
    test_load_config_expande_base_dir_real()
    print("\n>>> CONFIG UTILS OK <<<")
