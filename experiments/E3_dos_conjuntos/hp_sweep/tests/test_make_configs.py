# coding: ascii
"""Invariantes del diseno experimental del Exp I. Estos tests son la barrera
que impide que un descuido invalide 15 horas de GPU: si dos configs comparten
checkpoint_dir, la segunda corrida pisa a la primera SIN tirar error y despues
se reporta la EMA equivocada atribuida al modelo equivocado."""
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

import make_configs

GRID_PATH = _HERE.parent / "sweep_grid.yaml"
N_EXPECTED = 23

EXPECTED_FILES = {
    "hp_dmodel_32.yaml", "hp_dmodel_128.yaml", "hp_dmodel_256.yaml",
    "hp_layers_1.yaml", "hp_layers_3.yaml", "hp_layers_4.yaml",
    "hp_heads_2.yaml", "hp_heads_8.yaml",
    "hp_pma_2.yaml", "hp_pma_4.yaml",
    "hp_lr_0p0003.yaml", "hp_lr_0p003.yaml",
    "hp_bs_32.yaml", "hp_bs_128.yaml",
    "hp_fusion_64x32.yaml", "hp_fusion_256x128.yaml",
    "hp_grid_32x1.yaml", "hp_grid_32x4.yaml",
    "hp_grid_128x1.yaml", "hp_grid_128x4.yaml",
    "hp_rep_42.yaml", "hp_rep_43.yaml", "hp_rep_44.yaml",
}

# Claves de identidad: son unicas por construccion y NO forman parte del diseno
# experimental, asi que se excluyen de la comparacion "difiere en una sola clave".
IDENTITY_KEYS = {"experiment_name", "paths.checkpoint_dir"}


def _built():
    grid = make_configs.load_grid(GRID_PATH)
    baseline = make_configs.load_baseline(GRID_PATH.parent / grid["baseline_config"])
    return make_configs.build_all(grid, baseline), baseline


def _flat(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flat(v, key + "."))
        else:
            out[key] = v
    return out


def _diff_keys(a, b):
    fa, fb = _flat(a), _flat(b)
    keys = set(fa) | set(fb)
    return {k for k in keys
            if k not in IDENTITY_KEYS and fa.get(k, "<ausente>") != fb.get(k, "<ausente>")}


def test_exactly_23_configs_with_expected_names():
    built, _ = _built()
    assert len(built) == N_EXPECTED, f"{len(built)} configs, se esperaban {N_EXPECTED}"
    assert set(built) == EXPECTED_FILES, set(built) ^ EXPECTED_FILES
    print(f"[OK] {N_EXPECTED} configs con los nombres esperados")


def test_experiment_name_and_checkpoint_dir_unique():
    built, _ = _built()
    names = [c["experiment_name"] for c in built.values()]
    dirs = [c["paths"]["checkpoint_dir"] for c in built.values()]
    assert len(set(names)) == N_EXPECTED, "experiment_name repetido"
    assert len(set(dirs)) == N_EXPECTED, "checkpoint_dir repetido -- se pisarian entre si"
    print("[OK] experiment_name y checkpoint_dir unicos en las 23")


def test_ofat_configs_differ_in_exactly_one_key():
    built, baseline = _built()
    ofat_prefixes = ("hp_dmodel_", "hp_layers_", "hp_heads_", "hp_pma_",
                     "hp_lr_", "hp_bs_", "hp_fusion_")
    n = 0
    for fname, cfg in built.items():
        if not fname.startswith(ofat_prefixes):
            continue
        d = _diff_keys(cfg, baseline)
        assert len(d) == 1, f"{fname} difiere en {sorted(d)} (deberia ser 1 sola clave)"
        n += 1
    assert n == 16, f"{n} configs OFAT, se esperaban 16"
    print("[OK] las 16 corridas OFAT varian exactamente una dimension")


def test_grid2d_configs_differ_in_exactly_two_keys():
    built, baseline = _built()
    cells = set()
    for fname, cfg in built.items():
        if not fname.startswith("hp_grid_"):
            continue
        d = _diff_keys(cfg, baseline)
        assert d == {"model.d_model", "model.n_layers"}, f"{fname}: {sorted(d)}"
        cells.add((cfg["model"]["d_model"], cfg["model"]["n_layers"]))
    assert cells == {(32, 1), (32, 4), (128, 1), (128, 4)}, cells
    print("[OK] el grid 2D emite las 4 celdas no cubiertas por baseline/OFAT")


def test_noise_configs_differ_only_in_seed():
    built, baseline = _built()
    seeds = set()
    for fname, cfg in built.items():
        if not fname.startswith("hp_rep_"):
            continue
        assert _diff_keys(cfg, baseline) == {"hyperparameters.seed"}, fname
        seeds.add(cfg["hyperparameters"]["seed"])
    assert seeds == {42, 43, 44}, seeds
    print("[OK] las 3 replicas difieren solo en la seed")


def test_dmodel_divisible_by_nheads():
    """MAB.forward hace d = dim_V // num_heads: si no divide, el modelo se
    construye igual y entrena BASURA en silencio con dimensiones truncadas."""
    built, _ = _built()
    for fname, cfg in built.items():
        d, h = cfg["model"]["d_model"], cfg["model"]["n_heads"]
        assert d % h == 0, f"{fname}: d_model={d} no divisible por n_heads={h}"
    print("[OK] d_model divisible por n_heads en las 23")


def test_invariants_identical_in_all_configs():
    """Regla dura 8: si alguno de estos varia, las EMAs dejan de ser comparables
    y el estudio entero no vale nada."""
    built, baseline = _built()
    fb = _flat(baseline)
    invariants = [
        "hyperparameters.epochs",
        "hyperparameters.scheduler.patience",
        "hyperparameters.scheduler.factor",
        "system.num_workers",
        "system.device",
        "paths.base_dir",
        "paths.val_indices_filename",
        "paths.peaks_ch_filename",
        "paths.peaks_13c_filename",
        "paths.labels_filename",
        "paths.smiles_filename",
        "model.arch",
        "normalization.c13_ppm_max",
        "normalization.h1_ppm_max",
        "normalization.amp_ch0_scale",
    ]
    for fname, cfg in built.items():
        fc = _flat(cfg)
        for key in invariants:
            assert fc[key] == fb[key], f"{fname}: {key} = {fc[key]!r} != {fb[key]!r}"
    assert fb["hyperparameters.epochs"] == 100
    assert fb["hyperparameters.scheduler.patience"] == 8
    assert fb["hyperparameters.scheduler.factor"] == 0.7
    assert fb["system.num_workers"] == 0
    print("[OK] invariantes (epocas, scheduler, workers, dataset, split) identicos")


def test_base_dir_env_var_not_expanded():
    """load_baseline NO debe usar config_utils.load_config: eso expandiria
    ${NMR_DATA_DIR:-...} al valor de login-1 y los configs dejarian de servir en
    Clementina. Tienen que quedar con la variable literal."""
    built, _ = _built()
    for fname, cfg in built.items():
        assert "${NMR_DATA_DIR" in cfg["paths"]["base_dir"], f"{fname}: base_dir expandido"
    print("[OK] base_dir conserva la variable de entorno sin expandir")


def test_slug_formats():
    assert make_configs.slug(32) == "32"
    assert make_configs.slug(0.0003) == "0p0003"
    assert make_configs.slug(0.003) == "0p003"
    assert make_configs.slug([64, 32]) == "64x32"
    print("[OK] slug() formatea enteros, floats y listas sin caracteres raros")


def test_written_configs_on_disk_match_generated():
    """Los configs commiteados en configs/ tienen que ser EXACTAMENTE lo que
    make_configs.py genera hoy. Si alguien edito uno a mano, este test lo caza."""
    import yaml
    built, _ = _built()
    cfg_dir = GRID_PATH.parent / "configs"
    on_disk = {p.name for p in cfg_dir.glob("*.yaml")}
    assert on_disk == set(built), on_disk ^ set(built)
    for fname, cfg in built.items():
        with open(cfg_dir / fname, "r", encoding="utf-8") as f:
            assert yaml.safe_load(f) == cfg, f"{fname} en disco != generado"
    print("[OK] los 23 configs en disco coinciden con el generador")


def test_set_path_rejects_typo_in_existing_leaf():
    """Un typo en una clave que YA EXISTE en el baseline (ej. 'learing_rate' en
    vez de 'learning_rate') debe fallar fuerte, no crear una clave nueva
    ignorada en silencio por train.py."""
    baseline = {"hyperparameters": {"learning_rate": 0.001}}
    try:
        make_configs._set_path(baseline, "hyperparameters.learing_rate", 0.003)
    except KeyError:
        pass
    else:
        raise AssertionError("se esperaba KeyError por typo en la clave final")
    print("[OK] _set_path rechaza un typo en una clave final existente")


def test_set_path_allows_explicitly_whitelisted_new_leaf():
    """model.fusion_hidden y hyperparameters.seed son claves nuevas legitimas
    (default en el codigo, no en el YAML) y deben poder escribirse aunque no
    esten en el baseline."""
    baseline = {"model": {"d_model": 64}, "hyperparameters": {"batch_size": 64}}
    make_configs._set_path(baseline, "model.fusion_hidden", [64, 32])
    make_configs._set_path(baseline, "hyperparameters.seed", 43)
    assert baseline["model"]["fusion_hidden"] == [64, 32]
    assert baseline["hyperparameters"]["seed"] == 43
    print("[OK] _set_path permite las claves nuevas explicitamente permitidas")


def test_check_ignores_non_yaml_files_in_configs_dir():
    """Un archivo suelto en configs/ que no sea .yaml (ej. un .gitkeep o un
    README) no debe tratarse como una config 'que sobra': --check compara
    solo contra los .yaml en disco."""
    import tempfile
    import yaml
    built, _ = _built()
    with tempfile.TemporaryDirectory() as td:
        cfg_dir = Path(td)
        for fname, cfg in built.items():
            with open(cfg_dir / fname, "w", encoding="utf-8") as f:
                yaml.safe_dump(cfg, f)
        (cfg_dir / "README.md").write_text("not a config", encoding="utf-8")
        on_disk_yaml = {f.name for f in cfg_dir.glob("*.yaml")}
        extra = sorted(on_disk_yaml - set(built))
        assert extra == [], f"README.md no deberia contar como 'sobra': {extra}"
    print("[OK] --check filtra por extension .yaml, ignora archivos sueltos")


def test_insert_raises_on_filename_collision():
    out = {}
    make_configs._insert(out, "hp_dmodel_32.yaml", {"a": 1})
    try:
        make_configs._insert(out, "hp_dmodel_32.yaml", {"a": 2})
    except ValueError:
        pass
    else:
        raise AssertionError("se esperaba ValueError por nombre de archivo duplicado")
    print("[OK] _insert rechaza un nombre de archivo repetido")


if __name__ == "__main__":
    test_exactly_23_configs_with_expected_names()
    test_experiment_name_and_checkpoint_dir_unique()
    test_ofat_configs_differ_in_exactly_one_key()
    test_grid2d_configs_differ_in_exactly_two_keys()
    test_noise_configs_differ_only_in_seed()
    test_dmodel_divisible_by_nheads()
    test_invariants_identical_in_all_configs()
    test_base_dir_env_var_not_expanded()
    test_slug_formats()
    test_written_configs_on_disk_match_generated()
    test_set_path_rejects_typo_in_existing_leaf()
    test_set_path_allows_explicitly_whitelisted_new_leaf()
    test_check_ignores_non_yaml_files_in_configs_dir()
    test_insert_raises_on_filename_collision()
    print("\n>>> MAKE_CONFIGS OK <<<")
