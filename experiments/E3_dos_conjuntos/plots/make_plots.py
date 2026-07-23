# coding: ascii
"""
make_plots.py -- genera todas las figuras de Exp E Fase 3 (DeepSets + Set
Transformer) y del estudio de escalado, a partir de los .out de train/eval
(sin depender de torch: solo parsea texto + el parquet de predicciones).

Auto-detecta los .out por su contenido (experiment_name del header), no por
jobid, asi que sirve aunque cambien los numeros de job de SLURM.

Figuras generadas (en esta misma carpeta):
  1. train_curves_fase3.png     -- val/train loss vs epoca, DeepSets vs Set Transformer
  2. train_curves_scaling.png   -- val loss vs epoca, 5 fracciones de train
  3. ema_fase3.png              -- EMA cruda/asistida, DeepSets vs Set Transformer (+ baselines)
  4. ema_por_entorno.png        -- EMA asistida por entorno quimico, DeepSets vs Set Transformer
  5. confusion_topk_fase3.png   -- confusiones cruzadas (top-3 del .out), DeepSets | Set Transformer
  6. confusion_full_settransformer.png -- matriz de confusion cruzada completa (desde el parquet)
  7. scaling_curve_ema.png      -- EMA asistida y val loss vs tamano de train (curva de escalado)

Uso:
  python make_plots.py
  python make_plots.py --runs-dir <carpeta con los .out> --parquet <ruta parquet>
"""
import argparse
import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- Clases 19v (orden EXACTO de config/db.yaml) ----------------------------
GROUP_NAMES = [
    "CH3", "CH2", "CH", "Cq",
    "CH3-O", "CH2-O", "CH-O", "Cq-O",
    "CH3-N", "CH2-N", "CH-N", "Cq-N",
    "=CH2", "=CH/Ar", "Cqsp2", "Aldeh", "Imina",
    "C-2X", "C-3X",
]
N_CLASSES = 19

# Baselines historicos (de docs/Runs/RESULTS.md), EMA asistida limpia.
BASELINES_ASSIST = {"V10": 74.92, "Exp C": 70.02, "E2": 70.90}

ENTORNOS = ["Alifaticos (sp3)", "Heteroatomicos O/N (sp3)",
            "Carbonos sp2 (Olef/Arom/C=O)", "Sistemas X-Multiples (C-2X/3X)"]

# --- Parsers ----------------------------------------------------------------

RE_TRAIN_HEADER = re.compile(r"ENTRENAMIENTO EXP E FASE 3 \((\w+)\):\s*(\S+)")
RE_EPOCH = re.compile(
    r"\[EPOCH (\d+)\]\s+Train:\s+([\d.]+)\s+\|\s+Val:\s+([\d.]+)\s+\|\s+LR:\s+([\d.]+)")
RE_TRAINSIZE = re.compile(r"train=(\d+)")
RE_FRACTION = re.compile(r"train_fraction=([\d.]+)")
RE_EVAL_HEADER = re.compile(r"Experimento \(checkpoint\):\s*(\S+)\s*\|\s*arch:\s*(\w+)")
RE_EMA = re.compile(r"MODO (CRUDO|ASISTIDO).*EXACT MATCH ACCURACY:\s+([\d.]+)%")
RE_CONF_SRC = re.compile(r"^  (\S+)\s+\(falla en ([\d.]+)% mol\)")
RE_CONF_DST = re.compile(r"^      (\S+)\s+:\s+(\d+) senales")
RE_ENT_NAME = re.compile(r"^  (\S.*\S)\s*$")
RE_ENT_EMA = re.compile(r"EMA del entorno:\s+([\d.]+)%")
RE_SCALING_PCT = re.compile(r"scaling_(\d+)pct")


def parse_train_out(text):
    m = RE_TRAIN_HEADER.search(text)
    if not m:
        return None
    arch, name = m.group(1), m.group(2)
    epochs, tr, va, lr = [], [], [], []
    for em in RE_EPOCH.finditer(text):
        epochs.append(int(em.group(1)))
        tr.append(float(em.group(2)))
        va.append(float(em.group(3)))
        lr.append(float(em.group(4)))
    ms = RE_TRAINSIZE.search(text)
    train_size = int(ms.group(1)) if ms else None
    return {"arch": arch, "name": name, "epoch": np.array(epochs),
            "train": np.array(tr), "val": np.array(va), "lr": np.array(lr),
            "train_size": train_size}


def parse_eval_out(text):
    m = RE_EVAL_HEADER.search(text)
    if not m:
        return None
    name, arch = m.group(1), m.group(2)
    ema = {}
    for em in RE_EMA.finditer(text):
        ema["crude" if em.group(1) == "CRUDO" else "assist"] = float(em.group(2))
    # confusiones cruzadas (top-3 por clase origen)
    conf = {}
    cur = None
    for line in text.splitlines():
        ms = RE_CONF_SRC.match(line)
        if ms and ms.group(1) in GROUP_NAMES:
            cur = ms.group(1)
            conf[cur] = []
            continue
        md = RE_CONF_DST.match(line)
        if md and cur is not None and md.group(1) in GROUP_NAMES:
            conf[cur].append((md.group(1), int(md.group(2))))
    # EMA por entorno
    ent_ema = {}
    lines = text.splitlines()
    for i, line in enumerate(lines):
        mm = RE_ENT_EMA.search(line)
        if mm:
            # el nombre del entorno esta 2 lineas arriba
            for j in range(i - 1, max(i - 4, -1), -1):
                nm = lines[j].strip()
                if nm in ENTORNOS:
                    ent_ema[nm] = float(mm.group(1))
                    break
    return {"name": name, "arch": arch, "ema": ema, "conf": conf, "ent_ema": ent_ema}


def classify(runs_dir):
    trains, evals = {}, {}
    for f in sorted(Path(runs_dir).glob("*.out")):
        text = f.read_text(encoding="utf-8", errors="replace")
        if "ENTRENAMIENTO EXP E FASE 3" in text:
            d = parse_train_out(text)
            if d:
                trains[d["name"]] = d
        elif "EVALUACION EXP E FASE 3" in text:
            d = parse_eval_out(text)
            if d:
                evals[d["name"]] = d
    return trains, evals


def conf_matrix_from_topk(conf):
    """Matriz 19x19 con los conteos top-3 parseados del .out (celdas no top-3 = 0)."""
    M = np.zeros((N_CLASSES, N_CLASSES))
    idx = {n: i for i, n in enumerate(GROUP_NAMES)}
    for src, dsts in conf.items():
        for dst, cnt in dsts:
            M[idx[src], idx[dst]] = cnt
    return M


def cross_confusion_full(y_true, y_pred):
    """Matriz cruzada completa (mismo algoritmo que evaluate.analizar_confusiones)."""
    M = np.zeros((N_CLASSES, N_CLASSES))
    for i in range(len(y_true)):
        diff = y_pred[i] - y_true[i]
        under = [(g, -diff[g]) for g in range(N_CLASSES) if diff[g] < 0]
        over = [(g, diff[g]) for g in range(N_CLASSES) if diff[g] > 0]
        for gu, cu in under:
            for go, co in over:
                M[gu, go] += min(cu, co)
    return M


# --- Plots ------------------------------------------------------------------

def plot_train_curves_fase3(trains, outdir):
    ds = trains.get("nmr_202k_e3_deepsets_2sets_19v")
    st = trains.get("nmr_202k_e3_settransformer_2sets_19v")
    fig, ax = plt.subplots(figsize=(8, 5))
    for d, color, label in [(ds, "#1f77b4", "DeepSets"), (st, "#d62728", "Set Transformer")]:
        if d is None:
            continue
        ax.plot(d["epoch"], d["val"], color=color, lw=2, label=f"{label} (val)")
        ax.plot(d["epoch"], d["train"], color=color, lw=1, ls="--", alpha=0.5,
                label=f"{label} (train)")
    ax.set_xlabel("Epoca")
    ax.set_ylabel("Loss (ConstrainedMSE)")
    ax.set_yscale("log")
    ax.set_title("Exp E Fase 3 - curvas de entrenamiento")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "train_curves_fase3.png", dpi=150)
    plt.close(fig)


def plot_train_curves_scaling(trains, outdir):
    runs = []
    for name, d in trains.items():
        mp = RE_SCALING_PCT.search(name)
        if mp:
            runs.append((int(mp.group(1)), d))
    runs.sort()
    if not runs:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    cmap = plt.get_cmap("viridis")
    for k, (pct, d) in enumerate(runs):
        color = cmap(k / max(len(runs) - 1, 1))
        ax.plot(d["epoch"], d["val"], color=color, lw=2,
                label=f"{pct}% (N={d['train_size']:,})")
    ax.set_xlabel("Epoca")
    ax.set_ylabel("Val Loss (ConstrainedMSE)")
    ax.set_yscale("log")
    ax.set_title("Estudio de escalado - val loss por fraccion de train")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(title="Fraccion de train")
    fig.tight_layout()
    fig.savefig(outdir / "train_curves_scaling.png", dpi=150)
    plt.close(fig)


def plot_ema_fase3(evals, outdir):
    ds = evals.get("nmr_202k_e3_deepsets_2sets_19v")
    st = evals.get("nmr_202k_e3_settransformer_2sets_19v")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # --- asistida (con baselines historicos) ---
    ax = axes[0]
    labels = list(BASELINES_ASSIST.keys()) + ["DeepSets F3", "Set Transf. F3"]
    vals = list(BASELINES_ASSIST.values())
    vals += [ds["ema"]["assist"] if ds else 0, st["ema"]["assist"] if st else 0]
    colors = ["#999999"] * len(BASELINES_ASSIST) + ["#1f77b4", "#d62728"]
    bars = ax.bar(labels, vals, color=colors)
    ax.axhline(90, color="green", ls="--", alpha=0.6, label="objetivo ~90%")
    ax.set_ylabel("EMA asistida (%)")
    ax.set_ylim(0, 100)
    ax.set_title("EMA asistida (oraculo) vs historico")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.1f}", ha="center", fontsize=9)
    ax.tick_params(axis="x", rotation=20)
    ax.legend()

    # --- cruda vs asistida (solo F3) ---
    ax = axes[1]
    x = np.arange(2)
    w = 0.35
    crude = [ds["ema"]["crude"] if ds else 0, st["ema"]["crude"] if st else 0]
    assist = [ds["ema"]["assist"] if ds else 0, st["ema"]["assist"] if st else 0]
    ax.bar(x - w / 2, crude, w, label="cruda", color="#ff7f0e")
    ax.bar(x + w / 2, assist, w, label="asistida", color="#2ca02c")
    ax.set_xticks(x)
    ax.set_xticklabels(["DeepSets", "Set Transformer"])
    ax.set_ylabel("EMA (%)")
    ax.set_title("EMA cruda vs asistida (Fase 3)")
    for xi, c, a in zip(x, crude, assist):
        ax.text(xi - w / 2, c + 1, f"{c:.1f}", ha="center", fontsize=9)
        ax.text(xi + w / 2, a + 1, f"{a:.1f}", ha="center", fontsize=9)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "ema_fase3.png", dpi=150)
    plt.close(fig)


def plot_ema_por_entorno(evals, outdir):
    ds = evals.get("nmr_202k_e3_deepsets_2sets_19v")
    st = evals.get("nmr_202k_e3_settransformer_2sets_19v")
    if not (ds and st and ds["ent_ema"] and st["ent_ema"]):
        return
    ents = [e for e in ENTORNOS if e in ds["ent_ema"] and e in st["ent_ema"]]
    x = np.arange(len(ents))
    w = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w / 2, [ds["ent_ema"][e] for e in ents], w, label="DeepSets", color="#1f77b4")
    ax.bar(x + w / 2, [st["ent_ema"][e] for e in ents], w, label="Set Transformer", color="#d62728")
    ax.set_xticks(x)
    ax.set_xticklabels([e.replace(" ", "\n", 1) for e in ents], fontsize=9)
    ax.set_ylabel("EMA asistida del entorno (%)")
    ax.set_ylim(0, 100)
    ax.set_title("EMA asistida por entorno quimico (Fase 3)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "ema_por_entorno.png", dpi=150)
    plt.close(fig)


def _draw_heatmap(ax, M, title, annotate=True, vmax=None):
    Mp = M.copy()
    np.fill_diagonal(Mp, 0)
    im = ax.imshow(Mp, cmap="magma_r", vmax=vmax)
    ax.set_xticks(range(N_CLASSES)); ax.set_yticks(range(N_CLASSES))
    ax.set_xticklabels(GROUP_NAMES, rotation=90, fontsize=7)
    ax.set_yticklabels(GROUP_NAMES, fontsize=7)
    ax.set_xlabel("confunde con (over)")
    ax.set_ylabel("clase real (under)")
    ax.set_title(title)
    if annotate:
        thr = Mp.max() * 0.12 if Mp.max() > 0 else 1
        for i in range(N_CLASSES):
            for j in range(N_CLASSES):
                if Mp[i, j] >= thr:
                    ax.text(j, i, int(Mp[i, j]), ha="center", va="center",
                            fontsize=6, color="white" if Mp[i, j] > Mp.max() * 0.5 else "black")
    return im


def plot_confusion_topk_fase3(evals, outdir):
    ds = evals.get("nmr_202k_e3_deepsets_2sets_19v")
    st = evals.get("nmr_202k_e3_settransformer_2sets_19v")
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    for ax, d, title in [(axes[0], ds, "DeepSets (82.96%)"),
                         (axes[1], st, "Set Transformer (91.35%)")]:
        if d is None:
            continue
        M = conf_matrix_from_topk(d["conf"])
        im = _draw_heatmap(ax, M, f"Confusiones cruzadas (top-3) - {title}")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="senales confundidas")
    fig.suptitle("Confusiones cruzadas modo asistido (top-3 por clase, del .out)", y=1.02)
    fig.tight_layout()
    fig.savefig(outdir / "confusion_topk_fase3.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_full_settransformer(parquet_path, outdir):
    if not Path(parquet_path).exists():
        print(f"[WARN] no se encontro el parquet: {parquet_path} -- salteo matriz completa")
        return
    import pandas as pd
    df = pd.read_parquet(parquet_path)
    yt = np.array([np.asarray(r) for r in df["y_true"].to_list()])
    yp = np.array([np.asarray(r) for r in df["y_pred_assisted"].to_list()])
    M = cross_confusion_full(yt, yp)
    fig, ax = plt.subplots(figsize=(9, 8))
    im = _draw_heatmap(ax, M, "Set Transformer - matriz de confusion cruzada COMPLETA (asistido)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="senales confundidas")
    fig.tight_layout()
    fig.savefig(outdir / "confusion_full_settransformer.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_scaling_curve(trains, evals, outdir):
    rows = []
    for name, td in trains.items():
        mp = RE_SCALING_PCT.search(name)
        if not mp:
            continue
        ev = evals.get(name)
        if ev is None:
            continue
        rows.append((int(mp.group(1)), td["train_size"], td["val"].min(),
                     ev["ema"]["assist"], ev["ema"]["crude"]))
    rows.sort()
    if not rows:
        return
    pct = [r[0] for r in rows]
    N = [r[1] for r in rows]
    valloss = [r[2] for r in rows]
    assist = [r[3] for r in rows]
    crude = [r[4] for r in rows]

    fig, ax1 = plt.subplots(figsize=(9, 5.5))
    ax1.plot(N, assist, "o-", color="#d62728", lw=2, ms=8, label="EMA asistida")
    for xi, yi, p in zip(N, assist, pct):
        ax1.annotate(f"{yi:.1f}%\n({p}%)", (xi, yi), textcoords="offset points",
                     xytext=(0, 10), ha="center", fontsize=8)
    ax1.set_xscale("log")
    ax1.set_xlabel("Tamano del train set (moleculas, escala log)")
    ax1.set_ylabel("EMA asistida (%)", color="#d62728")
    ax1.tick_params(axis="y", labelcolor="#d62728")
    ax1.set_ylim(min(assist) - 3, 100)
    ax1.grid(True, which="both", alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(N, valloss, "s--", color="#1f77b4", lw=1.5, ms=6, alpha=0.7, label="Val loss (min)")
    ax2.set_ylabel("Val loss (min, MSE)", color="#1f77b4")
    ax2.tick_params(axis="y", labelcolor="#1f77b4")

    ax1.set_title("Curva de escalado - EMA asistida y val loss vs tamano de train")
    lines1, lab1 = ax1.get_legend_handles_labels()
    lines2, lab2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, lab1 + lab2, loc="lower right")
    fig.tight_layout()
    fig.savefig(outdir / "scaling_curve_ema.png", dpi=150)
    plt.close(fig)


def main():
    here = Path(__file__).resolve().parent
    repo_root = here.parents[2]   # experiments/E3_dos_conjuntos/plots -> repo root
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", type=str,
                    default=str(repo_root / "docs" / "runs" / "VE3_deepset+settransformers_HSQC+C"))
    ap.add_argument("--parquet", type=str,
                    default=str(repo_root / "docs" / "Runs" / "E3_settransformer" /
                                "predictions_nmr_202k_e3_settransformer_2sets_19v.parquet"))
    args = ap.parse_args()

    outdir = here
    trains, evals = classify(args.runs_dir)
    print(f"[INFO] runs-dir: {args.runs_dir}")
    print(f"[INFO] trains parseados: {sorted(trains.keys())}")
    print(f"[INFO] evals parseados:  {sorted(evals.keys())}")

    plot_train_curves_fase3(trains, outdir)
    plot_train_curves_scaling(trains, outdir)
    plot_ema_fase3(evals, outdir)
    plot_ema_por_entorno(evals, outdir)
    plot_confusion_topk_fase3(evals, outdir)
    plot_confusion_full_settransformer(args.parquet, outdir)
    plot_scaling_curve(trains, evals, outdir)

    pngs = sorted(p.name for p in outdir.glob("*.png"))
    print(f"[OK] figuras generadas en {outdir}:")
    for p in pngs:
        print(f"   - {p}")


if __name__ == "__main__":
    main()
