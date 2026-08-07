# coding: ascii
"""make_plot.py -- figura del Exp I: EMA asistida v2 vs cada eje OFAT, con la
banda de ruido de las replicas sombreada.

Es la figura que contesta visualmente "por que esta combinacion y no otra":
si todos los puntos caen dentro de la banda, la respuesta es "porque da igual, y
esta es la mas chica".

Uso:  python make_plot.py [--csv results.csv] [--out ../plots/hp_sweep_ofat.png]
"""
import argparse
import csv
import os
import statistics

import matplotlib
matplotlib.use("Agg")            # sin display: corre igual en el cluster
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(_HERE, "results.csv")
OUT_PATH = os.path.join(_HERE, "..", "plots", "hp_sweep_ofat.png")

# Titulo legible por eje OFAT (las claves son el 'name' de sweep_grid.yaml).
AXIS_TITLES = {
    "dmodel": "d_model (ancho)",
    "layers": "n_layers (profundidad)",
    "heads": "n_heads (cabezas de atencion)",
    "pma": "n_seeds (semillas del PMA)",
    "lr": "learning rate",
    "bs": "batch size",
    "fusion": "cabeza de fusion",
}


def _num(s):
    if s is None or str(s).strip() == "":
        return None
    return float(s)


def read_rows(csv_path=CSV_PATH):
    """Lee el CSV de collect_results.py. Celdas vacias -> None (NO 0.0: un cero
    se dibujaria como un resultado catastrofico inexistente)."""
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append({
                "run": r.get("run"),
                "n_params": _num(r.get("n_params")),
                "best_val_loss": _num(r.get("best_val_loss")),
                "ema_assist_v2": _num(r.get("ema_assist_v2")),
            })
    return rows


def _tag_sort_key(tag):
    """Convierte un tag slug-eado (ver make_configs.slug) de vuelta a un valor
    ordenable: '0p0003' -> 0.0003, '64x32' -> (64.0, 32.0), '128' -> (128.0,).
    El eje x de cada panel tiene que reflejar el VALOR del hiperparametro
    barrido, nunca el ranking de desempeno (EMA) -- ordenar por EMA baria un
    patron artificial en el grafico."""
    nums = []
    for part in tag.split("x"):
        try:
            nums.append(float(part.replace("p", ".").replace("m", "-")))
        except ValueError:
            return (float("inf"), tag)   # tag no numerico: al final, estable
    return tuple(nums)


def group_by_axis(rows):
    """{'dmodel': [('32', 90.1), ('128', 91.5)], ...} solo con los ejes OFAT.
    Excluye rep_* (son la banda de ruido) y grid_* (van en su propia tabla).
    Las corridas sin EMA todavia (pendientes) se omiten del panel. Las entradas
    se ordenan por el VALOR del hiperparametro barrido (usando _tag_sort_key),
    no por el EMA, para reflejar la relacion real entre parametro y desempeno."""
    groups = {}
    for r in rows:
        run = r.get("run") or ""
        if "_" not in run:
            continue
        axis, tag = run.split("_", 1)
        if axis in ("rep", "grid") or axis not in AXIS_TITLES:
            continue
        if r["ema_assist_v2"] is None:
            continue
        groups.setdefault(axis, []).append((tag, r["ema_assist_v2"]))
    for axis in groups:
        groups[axis].sort(key=lambda t: _tag_sort_key(t[0]))
    return groups


def _noise_band(rows):
    vals = [r["ema_assist_v2"] for r in rows
            if (r.get("run") or "").startswith("rep_") and r["ema_assist_v2"] is not None]
    if len(vals) < 2:
        return None
    return statistics.mean(vals), statistics.stdev(vals)


def make_figure(rows, out_path=OUT_PATH):
    groups = group_by_axis(rows)
    band = _noise_band(rows)
    axes_names = [a for a in AXIS_TITLES if a in groups]

    n = max(len(axes_names), 1)
    ncols = min(n, 4)
    nrows = (n + ncols - 1) // ncols
    fig, axarr = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 3.4 * nrows),
                              squeeze=False)

    for i in range(nrows * ncols):
        ax = axarr[i // ncols][i % ncols]
        if i >= len(axes_names):
            ax.axis("off")
            continue
        name = axes_names[i]
        labels = [t for t, _ in groups[name]]
        values = [v for _, v in groups[name]]
        ax.bar(range(len(values)), values, color="#4878a8")
        ax.set_xticks(range(len(values)))
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.set_title(AXIS_TITLES[name], fontsize=10)
        ax.set_ylabel("EMA asistida v2 (%)", fontsize=8)
        if band:
            mean, std = band
            ax.axhspan(mean - std, mean + std, color="#c8a020", alpha=0.25,
                       label="baseline +- 1 sigma")
            ax.axhline(mean, color="#c8a020", lw=1.2)
            lo = min(values + [mean - std]) - 0.5
            hi = max(values + [mean + std]) + 0.5
            ax.set_ylim(lo, hi)
            if i == 0:
                ax.legend(fontsize=7)

    title = "Exp I -- EMA asistida v2 por eje de hiperparametro"
    if band:
        title += f"  (baseline {band[0]:.2f} +- {band[1]:.2f} pp, n=3 replicas)"
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    out_path = os.path.abspath(out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=CSV_PATH)
    ap.add_argument("--out", default=OUT_PATH)
    args = ap.parse_args()
    path = make_figure(read_rows(args.csv), args.out)
    print(f"[OK] figura escrita en {path}")


if __name__ == "__main__":
    main()
