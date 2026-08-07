# coding: ascii
"""make_configs.py -- expande sweep_grid.yaml a los 23 configs del Exp I.

Uso:  python make_configs.py           (escribe/actualiza configs/)
      python make_configs.py --check   (falla si configs/ esta desactualizado)

Los configs generados se commitean: un revisor tiene que poder leer exactamente
que se entreno sin ejecutar nada.
"""
import argparse
import copy
import os
import sys

import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
GRID_PATH = os.path.join(_HERE, "sweep_grid.yaml")
OUT_DIR = os.path.join(_HERE, "configs")

_BANNER = (
    "# ARCHIVO GENERADO por hp_sweep/make_configs.py -- NO editar a mano.\n"
    "# Para cambiar el estudio: editar hp_sweep/sweep_grid.yaml y regenerar.\n"
    "# Exp I -- estudio de hiperparametros del Set Transformer (E3).\n"
)

# Ejes que legitimamente introducen una clave AUSENTE del baseline (tienen
# default en el codigo, no en el YAML): model.fusion_hidden y
# hyperparameters.seed, agregados en la Task 1. Cualquier otra ruta de
# sweep_grid.yaml debe apuntar a una clave que YA EXISTE en el baseline, o
# es un typo silencioso.
_ALLOWED_NEW_LEAF_KEYS = {"model.fusion_hidden", "hyperparameters.seed"}


def slug(value):
    """Convierte un valor a un fragmento de nombre de archivo seguro:
    0.0003 -> '0p0003' (el punto rompe la lectura de la extension),
    [64, 32] -> '64x32'."""
    if isinstance(value, (list, tuple)):
        return "x".join(slug(v) for v in value)
    if isinstance(value, float):
        return ("%g" % value).replace(".", "p").replace("-", "m")
    return str(value)


def load_grid(path=GRID_PATH):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_baseline(path):
    """Carga el config baseline con yaml.safe_load CRUDO, a proposito.

    NO usar config_utils.load_config(): expande ${NMR_DATA_DIR:-...} al valor de
    login-1 y los configs generados dejarian de funcionar en Clementina. La
    variable tiene que sobrevivir literal hasta el YAML de salida.
    """
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _set_path(cfg, dotted, value):
    """Escribe cfg['a']['b'] = value a partir de 'a.b'. Falla si el tramo
    intermedio no existe, o si la clave final no existe en el baseline y no
    esta en _ALLOWED_NEW_LEAF_KEYS: un typo en sweep_grid.yaml debe romper
    fuerte, no crear una clave nueva que train.py ignoraria en silencio."""
    parts = dotted.split(".")
    node = cfg
    for p in parts[:-1]:
        if p not in node or not isinstance(node[p], dict):
            raise KeyError(f"Ruta invalida en sweep_grid.yaml: {dotted!r} "
                           f"(no existe la seccion {p!r} en el baseline)")
        node = node[p]
    leaf = parts[-1]
    if leaf not in node and dotted not in _ALLOWED_NEW_LEAF_KEYS:
        raise KeyError(
            f"Ruta invalida en sweep_grid.yaml: {dotted!r} -- la clave "
            f"{leaf!r} no existe en el baseline y no esta en "
            f"_ALLOWED_NEW_LEAF_KEYS (typo probable). Si es una clave nueva "
            f"legitima (con default en el codigo), agregarla a "
            f"_ALLOWED_NEW_LEAF_KEYS explicitamente.")
    node[leaf] = value


def _get_path(cfg, dotted):
    node = cfg
    for p in dotted.split("."):
        node = node[p]
    return node


def _make_variant(baseline, prefix, name, tag, overrides):
    """Copia del baseline con los overrides aplicados y una identidad unica."""
    cfg = copy.deepcopy(baseline)
    for dotted, value in overrides.items():
        _set_path(cfg, dotted, value)
    ident = f"{name}_{tag}"
    cfg["experiment_name"] = f"{prefix}_{ident}"
    cfg["paths"]["checkpoint_dir"] = f"checkpoints_E3_hp_{ident}"
    return f"hp_{ident}.yaml", cfg


def build_all(grid, baseline):
    """Devuelve {nombre_de_archivo: config}. 16 OFAT + 4 grid 2D + 3 replicas."""
    prefix = grid["prefix"]
    out = {}

    # --- OFAT ---------------------------------------------------------------
    ofat_values = {}          # path -> valores probados, para validar el grid
    for axis in grid["ofat"]:
        ofat_values[axis["path"]] = list(axis["values"])
        for value in axis["values"]:
            fname, cfg = _make_variant(baseline, prefix, axis["name"],
                                       slug(value), {axis["path"]: value})
            out[fname] = cfg

    # --- Grid 2D ------------------------------------------------------------
    g = grid["grid2d"]
    paths = list(g["axes"].keys())
    if len(paths) != 2:
        raise ValueError("grid2d espera exactamente 2 ejes")
    pa, pb = paths
    for va in g["axes"][pa]:
        for vb in g["axes"][pb]:
            changed = [p for p, v in ((pa, va), (pb, vb)) if _get_path(baseline, p) != v]
            if len(changed) == 0:
                continue                      # es el baseline mismo
            if len(changed) == 1:
                # Ya cubierta por el OFAT. Validar que efectivamente lo este:
                # si no, la celda se perderia en silencio.
                p = changed[0]
                v = va if p == pa else vb
                if v not in ofat_values.get(p, []):
                    raise ValueError(
                        f"La celda del grid ({pa}={va}, {pb}={vb}) difiere del "
                        f"baseline solo en {p}={v}, pero ese valor NO esta en el "
                        f"eje OFAT correspondiente: quedaria sin correr.")
                continue
            fname, cfg = _make_variant(baseline, prefix, g["name"],
                                       f"{slug(va)}x{slug(vb)}", {pa: va, pb: vb})
            out[fname] = cfg

    # --- Replicas (piso de ruido) -------------------------------------------
    n = grid["noise"]
    for value in n["values"]:
        fname, cfg = _make_variant(baseline, prefix, n["name"], slug(value),
                                   {n["path"]: value})
        out[fname] = cfg

    # --- Validaciones globales ----------------------------------------------
    for fname, cfg in out.items():
        d, h = cfg["model"]["d_model"], cfg["model"]["n_heads"]
        if d % h != 0:
            raise ValueError(f"{fname}: d_model={d} no es divisible por n_heads={h}")
    if len({c["paths"]["checkpoint_dir"] for c in out.values()}) != len(out):
        raise ValueError("checkpoint_dir repetido: las corridas se pisarian entre si")
    return out


def _dump(cfg):
    return _BANNER + yaml.safe_dump(cfg, sort_keys=False, default_flow_style=False,
                                    allow_unicode=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="no escribe; falla si configs/ esta desactualizado")
    args = ap.parse_args()

    grid = load_grid()
    baseline = load_baseline(os.path.join(_HERE, grid["baseline_config"]))
    built = build_all(grid, baseline)

    if args.check:
        stale = []
        for fname, cfg in built.items():
            path = os.path.join(OUT_DIR, fname)
            if not os.path.exists(path):
                stale.append(f"{fname} (falta)")
                continue
            with open(path, "r", encoding="utf-8") as f:
                if yaml.safe_load(f) != cfg:
                    stale.append(f"{fname} (distinto)")
        extra = sorted(set(os.listdir(OUT_DIR)) - set(built)) if os.path.isdir(OUT_DIR) else []
        if stale or extra:
            print("[FAIL] configs/ desactualizado:")
            for s in stale + [f"{e} (sobra)" for e in extra]:
                print(f"  - {s}")
            sys.exit(1)
        print(f"[OK] los {len(built)} configs en disco estan al dia")
        return

    os.makedirs(OUT_DIR, exist_ok=True)
    for fname, cfg in sorted(built.items()):
        with open(os.path.join(OUT_DIR, fname), "w", encoding="utf-8") as f:
            f.write(_dump(cfg))
    print(f"[OK] {len(built)} configs escritos en {OUT_DIR}")


if __name__ == "__main__":
    main()
