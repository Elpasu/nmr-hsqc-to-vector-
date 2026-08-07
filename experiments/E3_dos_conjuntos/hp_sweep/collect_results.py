# coding: ascii
"""collect_results.py -- junta los .out de SLURM del Exp I en un CSV y una tabla
markdown lista para pegar en docs/Runs/RESULTS.md.

Corre en la PC local sobre los .out que Lucas baje del cluster; no necesita torch
ni acceso al cluster.

Uso:  python collect_results.py --out-dir ruta/a/los/out
      python collect_results.py --out-dir ruta/a/los/out --config-dir hp_sweep/configs
      (--config-dir es el default: completa PENDIENTE las corridas de las 23 que
      todavia no tienen .out, en vez de que desaparezcan de la tabla)
"""
import argparse
import csv
import os
import re
import statistics
from pathlib import Path

import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(_HERE, "results.csv")
DEFAULT_CONFIG_DIR = os.path.join(_HERE, "configs")

FIELDS = ["run", "experiment_name", "n_params", "best_val_loss",
          "ema_crude", "ema_assist_v1", "ema_assist_v2", "minutes"]

# Prefijo de los 23 experiment_name del sweep (ver make_configs.py: prefix +
# "_" + ident). Un .out cuyo experiment_name no arranca asi es de OTRO
# experimento (V10, Exp F, etc.) y se ignora -- si no, ensuciaria la tabla con
# una fila que no pertenece al sweep.
_PREFIX = "nmr_202k_e3_hp_"

# train.py:111 y evaluate.py:212 imprimen el nombre; sirve cualquiera de los dos.
_RE_NAME = re.compile(r"ENTRENAMIENTO EXP E FASE 3 \([^)]*\): (\S+)")
_RE_NAME_EVAL = re.compile(r"Experimento \(checkpoint\): (\S+)")
# train.py:139
_RE_PARAMS = re.compile(r"Parametros totales del modelo \([^)]*\): ([\d,]+)")
# train.py:180
_RE_DONE = re.compile(r"\[DONE\] ([\d.]+) min\. Mejor Val: ([\d.]+)")
# evaluate.py:175 (tabla de 3 vias que emite --oraculo all)
_RE_EMA3 = re.compile(r"EMA GLOBAL\s+([\d.]+)%\s+([\d.]+)%\s+([\d.]+)%")


def _first(regex, text, group=1):
    m = regex.search(text)
    return m.group(group) if m else None


def parse_out(text):
    """Extrae las metricas de un .out. Todo lo que no aparezca queda en None:
    una corrida truncada NO debe aparecer como si tuviera resultados."""
    row = {k: None for k in FIELDS}

    row["experiment_name"] = _first(_RE_NAME, text) or _first(_RE_NAME_EVAL, text)

    params = _first(_RE_PARAMS, text)
    if params:
        row["n_params"] = int(params.replace(",", ""))

    m = _RE_DONE.search(text)
    if m:
        row["minutes"] = float(m.group(1))
        row["best_val_loss"] = float(m.group(2))

    m = _RE_EMA3.search(text)
    if m:
        row["ema_crude"] = float(m.group(1))
        row["ema_assist_v1"] = float(m.group(2))
        row["ema_assist_v2"] = float(m.group(3))

    return row


def run_label(experiment_name, prefix=_PREFIX):
    """'nmr_202k_e3_hp_dmodel_128' -> 'dmodel_128'."""
    if not experiment_name:
        return None
    return experiment_name[len(prefix):] if experiment_name.startswith(prefix) else experiment_name


def expected_runs(config_dir):
    """Lee los experiment_name de configs/*.yaml y devuelve sus run labels,
    ordenados. Es el universo completo de corridas esperadas (23 hoy): una
    corrida sin .out todavia (no lanzada, o no bajada) tiene que aparecer
    como PENDIENTE en la tabla, no desaparecer de ella."""
    labels = []
    for path in sorted(Path(config_dir).glob("*.yaml")):
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        labels.append(run_label(cfg["experiment_name"]))
    return labels


def collect(out_dir, config_dir=None):
    """Junta los .out de out_dir en una fila por corrida.

    Un .out cuyo experiment_name no tiene el prefijo del sweep (_PREFIX) es de
    otro experimento y se ignora. Dos .out para la MISMA corrida (ej.
    relanzaste un job truncado por TIME LIMIT sin borrar el .out viejo) es un
    error fuerte: promediar ambos en silencio duplicaria una replica en la
    banda de ruido (ver noise_band), inflando artificialmente la confianza del
    estudio. Si se pasa config_dir, cualquier corrida esperada (segun
    configs/*.yaml) que no tenga .out aparece igual, como fila PENDIENTE, en
    vez de desaparecer de la tabla.
    """
    by_run = {}
    by_run_source = {}
    for name in sorted(os.listdir(out_dir)):
        if not name.endswith(".out"):
            continue
        with open(os.path.join(out_dir, name), "r", encoding="utf-8", errors="replace") as f:
            row = parse_out(f.read())
        exp_name = row["experiment_name"]
        if exp_name is None or not exp_name.startswith(_PREFIX):
            continue                       # .out sin nombre, o de otro experimento
        run = run_label(exp_name)
        if run in by_run:
            raise ValueError(
                f"Dos .out distintos para la corrida {run!r}: "
                f"{by_run_source[run]!r} y {name!r}. Seguramente relanzaste un "
                f"job truncado (TIME LIMIT u otro fallo) sin borrar el .out "
                f"viejo -- promediar ambos duplicaria una replica en la banda "
                f"de ruido. Borra el .out del intento fallido y volve a correr.")
        row["run"] = run
        by_run[run] = row
        by_run_source[run] = name

    if config_dir is not None:
        for run in expected_runs(config_dir):
            if run not in by_run:
                pending = {k: None for k in FIELDS}
                pending["run"] = run
                by_run[run] = pending

    rows = list(by_run.values())
    rows.sort(key=lambda r: r["run"] or "")
    return rows


def noise_band(rows):
    """(media, desvio muestral) de la EMA v2 sobre las replicas rep_*. None si
    hay menos de 2: con una sola replica no hay banda que reportar."""
    vals = [r["ema_assist_v2"] for r in rows
            if (r.get("run") or "").startswith("rep_") and r.get("ema_assist_v2") is not None]
    if len(vals) < 2:
        return None
    return statistics.mean(vals), statistics.stdev(vals)


def _fmt(v, spec):
    return "PENDIENTE" if v is None else format(v, spec)


def to_markdown(rows):
    lines = [
        "| Corrida | Params | Best Val Loss | EMA cruda | EMA asist v2 | min |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r.get('run') or '?'} "
            f"| {_fmt(r.get('n_params'), ',d')} "
            f"| {_fmt(r.get('best_val_loss'), '.4f')} "
            f"| {_fmt(r.get('ema_crude'), '.2f')} "
            f"| {_fmt(r.get('ema_assist_v2'), '.2f')} "
            f"| {_fmt(r.get('minutes'), '.1f')} |"
        )
    band = noise_band(rows)
    if band:
        lines.append("")
        lines.append(f"**Piso de ruido (replicas rep_*):** EMA asistida v2 = "
                     f"{band[0]:.2f} +- {band[1]:.2f} pp. Las diferencias por debajo de "
                     f"{band[1]:.2f} pp NO son significativas.")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True, help="carpeta con los expE3_hp_*.out")
    ap.add_argument("--config-dir", default=DEFAULT_CONFIG_DIR,
                    help="carpeta con los configs del sweep (default: hp_sweep/configs). "
                         "Se usa para completar PENDIENTE las corridas que todavia no "
                         "tienen .out. Pasar --config-dir '' para desactivar.")
    args = ap.parse_args()

    config_dir = args.config_dir or None
    rows = collect(args.out_dir, config_dir=config_dir)
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in FIELDS})

    done = sum(1 for r in rows if r["ema_assist_v2"] is not None)
    print(f"[OK] {done}/{len(rows)} corridas con EMA -> {CSV_PATH}\n")
    print(to_markdown(rows))


if __name__ == "__main__":
    main()
