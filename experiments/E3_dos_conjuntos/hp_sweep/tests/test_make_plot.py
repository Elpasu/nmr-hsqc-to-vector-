# coding: ascii
"""La figura tiene que poder generarse ANTES de que existan los resultados
reales (con corridas a medias) y no romperse. Se testea con un CSV sintetico en
un directorio temporal: no toca plots/ del repo."""
import csv
import sys
import tempfile
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

import make_plot

ROWS = [
    {"run": "rep_42", "ema_assist_v2": "91.35", "n_params": "70163", "best_val_loss": "0.0097"},
    {"run": "rep_43", "ema_assist_v2": "91.95", "n_params": "70163", "best_val_loss": "0.0094"},
    {"run": "rep_44", "ema_assist_v2": "92.14", "n_params": "70163", "best_val_loss": "0.0091"},
    {"run": "dmodel_32", "ema_assist_v2": "90.10", "n_params": "22019", "best_val_loss": "0.0121"},
    {"run": "dmodel_128", "ema_assist_v2": "91.50", "n_params": "241555", "best_val_loss": "0.0093"},
    {"run": "dmodel_256", "ema_assist_v2": "", "n_params": "", "best_val_loss": ""},   # pendiente
    {"run": "layers_1", "ema_assist_v2": "90.80", "n_params": "45000", "best_val_loss": "0.0110"},
    {"run": "layers_4", "ema_assist_v2": "91.60", "n_params": "120000", "best_val_loss": "0.0092"},
    {"run": "grid_32x1", "ema_assist_v2": "89.90", "n_params": "18000", "best_val_loss": "0.0130"},
]


def _write_csv(path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["run", "n_params", "best_val_loss", "ema_assist_v2"])
        w.writeheader()
        for r in ROWS:
            w.writerow(r)


def test_read_rows_parses_and_skips_empty():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "results.csv"
        _write_csv(p)
        rows = make_plot.read_rows(p)
    assert len(rows) == len(ROWS)
    got = {r["run"]: r["ema_assist_v2"] for r in rows}
    assert got["dmodel_32"] == 90.10
    assert got["dmodel_256"] is None, "una celda vacia debe ser None, no 0.0"
    print("[OK] read_rows convierte numeros y deja None las celdas vacias")


def test_group_by_axis_splits_ofat_axes():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "results.csv"
        _write_csv(p)
        groups = make_plot.group_by_axis(make_plot.read_rows(p))
    assert "dmodel" in groups and "layers" in groups
    assert "rep" not in groups, "las replicas son la banda de ruido, no un eje OFAT"
    assert "grid" not in groups, "el grid 2D no va en los paneles OFAT"
    assert [v for _, v in groups["dmodel"]] == [90.10, 91.50], groups["dmodel"]
    print("[OK] group_by_axis separa los ejes OFAT y excluye rep_/grid_")


def test_make_figure_writes_png():
    with tempfile.TemporaryDirectory() as td:
        csv_path = Path(td) / "results.csv"
        png_path = Path(td) / "hp_sweep_ofat.png"
        _write_csv(csv_path)
        make_plot.make_figure(make_plot.read_rows(csv_path), png_path)
        assert png_path.exists() and png_path.stat().st_size > 5000, png_path.stat().st_size
    print("[OK] make_figure escribe un PNG no trivial")


def test_make_figure_survives_empty_csv():
    """Antes de correr los jobs el CSV esta vacio; el script no debe explotar."""
    with tempfile.TemporaryDirectory() as td:
        csv_path = Path(td) / "results.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=["run", "n_params", "best_val_loss",
                                          "ema_assist_v2"]).writeheader()
        png_path = Path(td) / "vacio.png"
        make_plot.make_figure(make_plot.read_rows(csv_path), png_path)
        assert png_path.exists()
    print("[OK] CSV vacio -> figura vacia, sin excepcion")


def test_group_by_axis_sorts_by_hyperparameter_value_not_by_score():
    """d_model=128 anotada a proposito con la PEOR EMA de las tres, para que un
    sort por score (el bug original) la deje primera en vez de en el medio."""
    rows = [
        {"run": "dmodel_32", "ema_assist_v2": "91.00", "n_params": "20000", "best_val_loss": "0.012"},
        {"run": "dmodel_128", "ema_assist_v2": "85.00", "n_params": "240000", "best_val_loss": "0.015"},
        {"run": "dmodel_256", "ema_assist_v2": "92.00", "n_params": "800000", "best_val_loss": "0.009"},
    ]
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "results.csv"
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["run", "n_params", "best_val_loss", "ema_assist_v2"])
            w.writeheader()
            for r in rows:
                w.writerow(r)
        groups = make_plot.group_by_axis(make_plot.read_rows(p))
    tags_in_order = [tag for tag, _ in groups["dmodel"]]
    assert tags_in_order == ["32", "128", "256"], f"esperaba ['32', '128', '256'] pero got {tags_in_order}"
    print("[OK] group_by_axis ordena por el VALOR del hiperparametro, no por el score")


def test_group_by_axis_warns_on_unknown_axis():
    import io
    import contextlib
    rows = [
        {"run": "newaxis_5", "ema_assist_v2": "90.0", "n_params": "10000", "best_val_loss": "0.01"},
    ]
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "results.csv"
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["run", "n_params", "best_val_loss", "ema_assist_v2"])
            w.writeheader()
            for r in rows:
                w.writerow(r)
        buf = io.StringIO()
        with contextlib.redirect_stderr(buf):
            groups = make_plot.group_by_axis(make_plot.read_rows(p))
    assert "newaxis" not in groups
    assert "newaxis" in buf.getvalue(), buf.getvalue()
    print("[OK] eje desconocido genera warning en vez de desaparecer en silencio")


if __name__ == "__main__":
    test_read_rows_parses_and_skips_empty()
    test_group_by_axis_splits_ofat_axes()
    test_make_figure_writes_png()
    test_make_figure_survives_empty_csv()
    test_group_by_axis_sorts_by_hyperparameter_value_not_by_score()
    test_group_by_axis_warns_on_unknown_axis()
    print("\n>>> MAKE_PLOT OK <<<")
