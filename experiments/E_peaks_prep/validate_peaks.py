# coding: ascii
"""
validate_peaks.py -- Exp E Fase 1: valida que la extraccion de picos
(extract_peaks.py) no pierda informacion respecto al label de 19 clases.

Compara, por molecula, el numero de picos detectados (peaks_mask.sum) contra
el conteo VISIBLE del label (excluyendo Cq/Cq-O/Cq-N/Cqsp2 -- mismo criterio
que scripts/audit_data_pipeline.py). Un deficit positivo (visible > blobs)
es una colision real: dos o mas carbonos distintos cuyas gaussianas se
fusionaron en un solo componente conexo.

Uso (en el cluster, login node, sin GPU):
    python validate_peaks.py --config config.yaml
"""
import argparse
from pathlib import Path

import numpy as np

INVISIBLE_CLASSES = ["Cq", "Cq-O", "Cq-N", "Cqsp2"]


def visible_label_counts(labels, class_names):
    idx_invisible = [class_names.index(c) for c in INVISIBLE_CLASSES]
    idx_visible = [i for i in range(len(class_names)) if i not in idx_invisible]
    return labels[:, idx_visible].sum(axis=1).astype(int)


def blob_counts_from_mask(peaks_mask):
    return peaks_mask.sum(axis=1).astype(int)


def validation_report(blob_counts, visible_counts):
    deficit = visible_counts.astype(int) - blob_counts.astype(int)
    n = len(deficit)
    n_exact = int((deficit == 0).sum())
    n_collision = int((deficit > 0).sum())
    mean_deficit_positive = float(deficit[deficit > 0].mean()) if n_collision > 0 else 0.0
    return {
        "n": n,
        "pct_exact_match": n_exact / n * 100.0,
        "n_collision": n_collision,
        "pct_collision": n_collision / n * 100.0,
        "mean_deficit_positive": mean_deficit_positive,
        "deficit": deficit,
    }


def main(config_path):
    import h5py
    import yaml

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    base_dir = Path(cfg["paths"]["base_dir"])
    labels_path = base_dir / cfg["paths"]["labels_filename"]
    peaks_path = base_dir / cfg["paths"]["peaks_output_filename"]
    class_names = cfg["classes_19v"]

    print("=" * 60)
    print("  EXP E FASE 1: validacion de picos vs label")
    print("=" * 60)

    labels = np.load(labels_path).astype(int)
    with h5py.File(peaks_path, "r") as f:
        peaks_mask = f["peaks_mask"][:]

    visible_counts = visible_label_counts(labels, class_names)
    blob_counts = blob_counts_from_mask(peaks_mask)
    report = validation_report(blob_counts, visible_counts)

    print(f"\nMoleculas evaluadas: {report['n']}")
    print(f"Match exacto (blobs == visible): {report['pct_exact_match']:.2f}%")
    print(f"Con colision (visible > blobs): {report['n_collision']} "
          f"({report['pct_collision']:.2f}%)")
    print(f"Deficit promedio en las que tienen colision: "
          f"{report['mean_deficit_positive']:.2f}")

    deficit = report["deficit"]
    worst_idx = np.argsort(deficit)[::-1][:3]
    print("\nEjemplos con mayor colision (para inspeccion manual):")
    for idx in worst_idx:
        if deficit[idx] <= 0:
            break
        print(f"  molecula {idx}: blobs={blob_counts[idx]} "
              f"visible_label={visible_counts[idx]} deficit={deficit[idx]}")

    print("\n>>> EXP E FASE 1 validate_peaks.py OK <<<")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exp E Fase 1: validacion de picos")
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    main(args.config)
