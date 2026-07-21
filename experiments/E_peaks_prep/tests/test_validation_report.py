# coding: ascii
"""Tests de validate_peaks.py -- corre localmente, solo numpy."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from validate_peaks import (
    blob_counts_from_mask,
    validation_report,
    visible_label_counts,
)

CLASS_NAMES = [
    "CH3", "CH2", "CH", "Cq", "CH3-O", "CH2-O", "CH-O", "Cq-O",
    "CH3-N", "CH2-N", "CH-N", "Cq-N", "=CH2", "=CH/Ar", "Cqsp2",
    "Aldeh", "Imina", "C-2X", "C-3X",
]


def test_visible_label_counts_excludes_invisible_classes():
    labels = np.zeros((2, 19), dtype=int)
    labels[0, CLASS_NAMES.index("CH3")] = 3       # visible
    labels[0, CLASS_NAMES.index("Cq")] = 5         # invisible, no debe contar
    labels[1, CLASS_NAMES.index("Cqsp2")] = 4      # invisible, no debe contar
    labels[1, CLASS_NAMES.index("CH")] = 2         # visible

    counts = visible_label_counts(labels, CLASS_NAMES)
    assert counts.tolist() == [3, 2], counts.tolist()
    print("[OK] test_visible_label_counts_excludes_invisible_classes")


def test_blob_counts_from_mask():
    mask = np.array([
        [True, True, False],
        [True, False, False],
        [False, False, False],
    ])
    counts = blob_counts_from_mask(mask)
    assert counts.tolist() == [2, 1, 0], counts.tolist()
    print("[OK] test_blob_counts_from_mask")


def test_validation_report_exact_match_and_collision():
    blob_counts = np.array([3, 1, 5])
    visible_counts = np.array([3, 2, 5])  # molecula 1: deficit=1 (colision real)

    report = validation_report(blob_counts, visible_counts)
    assert report["n"] == 3
    assert abs(report["pct_exact_match"] - (2 / 3 * 100.0)) < 1e-6
    assert report["n_collision"] == 1
    assert abs(report["mean_deficit_positive"] - 1.0) < 1e-6
    print(f"[OK] test_validation_report_exact_match_and_collision -> {report}")


if __name__ == "__main__":
    test_visible_label_counts_excludes_invisible_classes()
    test_blob_counts_from_mask()
    test_validation_report_exact_match_and_collision()
    print("\n>>> test_validation_report.py OK <<<")
