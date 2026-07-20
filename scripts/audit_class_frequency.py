# coding: ascii
"""
scripts/audit_class_frequency.py -- auditoria de la distribucion de las 19
clases en el dataset completo, para separar "clase dificil por ser
invisible en HSQC" de "clase dificil porque casi no hay ejemplos".

No requiere GPU ni torch: corre en el login node con numpy + pyyaml.

Uso:
    python scripts/audit_class_frequency.py --config config/db.yaml
"""
import argparse
from pathlib import Path

import numpy as np
import yaml


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def compute_class_stats(labels, class_names):
    """labels: (N, 19) int array de conteos. Devuelve una lista de dicts,
    uno por clase, con total_count, presence_rate (moleculas con >=1
    ocurrencia) y mean_count (promedio sobre TODAS las moleculas, no solo
    las que tienen esa clase)."""
    n = labels.shape[0]
    stats = []
    for i, name in enumerate(class_names):
        col = labels[:, i]
        total_count = int(col.sum())
        n_present = int((col > 0).sum())
        stats.append({
            "name": name,
            "idx": i,
            "total_count": total_count,
            "presence_rate": n_present / n * 100.0,
            "mean_count": float(col.mean()),
        })
    return stats


def print_report(title, stats, n):
    print(f"\n{'=' * 70}")
    print(f"  {title}  (N={n})")
    print(f"{'=' * 70}")
    print(f"\n{'CLASE':<10} | {'% moleculas c/ >=1':>18} | {'conteo total':>12} | {'promedio/mol':>12}")
    print("-" * 70)
    for s in sorted(stats, key=lambda s: s["presence_rate"]):
        print(f"{s['name']:<10} | {s['presence_rate']:>17.2f}% | {s['total_count']:>12,} | {s['mean_count']:>12.3f}")


def main(config_path):
    cfg = load_config(config_path)
    base_dir = Path(cfg["data"]["base_dir"])
    labels_path = base_dir / cfg["data"]["labels_19v"]
    class_names = cfg["classes_19v"]
    n_144k = int(cfg["data"]["N_144k"])

    print(f"-> Cargando labels desde: {labels_path}")
    labels = np.load(labels_path).astype(int)
    n_total = labels.shape[0]
    print(f"-> Moleculas totales: {n_total}  |  clases: {len(class_names)}")

    stats_all = compute_class_stats(labels, class_names)
    print_report("DISTRIBUCION COMPLETA (202465 moleculas)", stats_all, n_total)

    # Split original (144k) vs nuevas (58k) -- mismo supuesto de orden
    # preservado que ya valido Exp D (el val congelado dio exactamente
    # 14428, confirmando que los primeros N_144k indices son las
    # originales sin reordenar).
    labels_orig = labels[:n_144k]
    labels_new = labels[n_144k:]
    stats_orig = compute_class_stats(labels_orig, class_names)
    stats_new = compute_class_stats(labels_new, class_names)

    print_report(f"SOLO ORIGINALES (144k, indices [0,{n_144k}))", stats_orig, len(labels_orig))
    print_report(f"SOLO NUEVAS (58k scaffolds diversos, indices [{n_144k},{n_total}))", stats_new, len(labels_new))

    print(f"\n{'=' * 70}")
    print("  DELTA DE PRESENCIA: nuevas - originales (pp)")
    print(f"{'=' * 70}")
    print(f"\n{'CLASE':<10} | {'orig %':>8} | {'nuevas %':>8} | {'delta pp':>9}")
    print("-" * 70)
    by_name_orig = {s["name"]: s for s in stats_orig}
    by_name_new = {s["name"]: s for s in stats_new}
    deltas = []
    for name in class_names:
        o = by_name_orig[name]["presence_rate"]
        nnew = by_name_new[name]["presence_rate"]
        deltas.append((name, o, nnew, nnew - o))
    for name, o, nnew, d in sorted(deltas, key=lambda x: x[3]):
        print(f"{name:<10} | {o:>7.2f}% | {nnew:>7.2f}% | {d:>+8.2f}")

    print("\n>>> AUDIT OK <<<")
    print("Lectura: las clases arriba de todo en 'DISTRIBUCION COMPLETA' son las")
    print("mas raras del dataset -- si Cqsp2 / =CH/Ar aparecen ahi, parte de su MAE")
    print("alto puede ser escasez de ejemplos, no solo ser invisibles en HSQC.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auditoria de distribucion de clases (19v)")
    parser.add_argument("--config", type=str, default="config/db.yaml")
    args = parser.parse_args()
    main(args.config)
