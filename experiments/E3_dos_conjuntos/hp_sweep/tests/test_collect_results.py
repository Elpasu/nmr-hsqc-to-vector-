# coding: ascii
"""Parsing de los .out de SLURM. El .out sintetico de abajo copia LITERALMENTE
el formato que imprimen train.py y evaluate.py --oraculo all (verificado contra
train.py:111/139/180 y evaluate.py:175/212)."""
import os
import sys
import tempfile
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

import yaml

import collect_results

FAKE_OUT = """=== EXP I | CONFIG: hp_sweep/configs/hp_dmodel_128.yaml ===
=== FASE 1/2: TRAIN ===
--- ENTRENAMIENTO EXP E FASE 3 (settransformer): nmr_202k_e3_hp_dmodel_128 ---
[INFO] Seed: 42
[INFO] Dispositivo: cuda
[INFO] Split congelado: SMILES invalidos=0 | train=187314 (leak removido=12, train_fraction=1.0) | val=14428
[INFO] Parametros totales del modelo (settransformer): 241,555 (chico por diseno; V10 ~8,603,299)
[INFO] Scheduler: patience=8, factor=0.7

[START] 100 epochs...
[EPOCH 1] Train: 0.9012 | Val: 0.4211 | LR: 0.001000 | Time: 23.4s
[EPOCH 100] Train: 0.0081 | Val: 0.0093 | LR: 0.001000 | Time: 23.1s
[SAVE] Nuevo mejor modelo!

[DONE] 41.7 min. Mejor Val: 0.0093
=== FASE 2/2: EVAL ===
============================================================
  EVALUACION EXP E FASE 3 (dos conjuntos) - SPLIT CONGELADO
============================================================
-> Experimento (checkpoint): nmr_202k_e3_hp_dmodel_128  | arch: settransformer
-> Modos: ['off', 'on', 'v2']   | idx_ch2: [1, 5, 9, 12]

============================================================
  MODO CRUDO  ->  EXACT MATCH ACCURACY: 2.11%
============================================================

====================================================================
  TABLA COMPARATIVA: CRUDA vs ASISTIDA v1 (doble) vs ASISTIDA v2 (hetero)
====================================================================

                                  CRUDA  ASIST v1  ASIST v2    v2-v1
  ------------------------------------------------------------------
  EMA GLOBAL                       2.11%    91.48%    91.50%    +0.02
  Alifaticos                      96.50%    96.55%    96.55%    +0.00
"""

TRUNCATED_OUT = """--- ENTRENAMIENTO EXP E FASE 3 (settransformer): nmr_202k_e3_hp_bs_32 ---
[INFO] Parametros totales del modelo (settransformer): 70,163 (chico por diseno; V10 ~8,603,299)
[START] 100 epochs...
[EPOCH 3] Train: 0.4001 | Val: 0.2010 | LR: 0.001000 | Time: 40.2s
slurmstepd: error: *** JOB 2377999 CANCELLED AT 2026-08-08T04:00:00 DUE TO TIME LIMIT ***
"""

UNRELATED_OUT = ("--- ENTRENAMIENTO EXP E FASE 3 (settransformer): "
                  "nmr_202k_v10_2ch_fm_19v ---\n")


def test_parses_complete_run():
    r = collect_results.parse_out(FAKE_OUT)
    assert r["experiment_name"] == "nmr_202k_e3_hp_dmodel_128", r["experiment_name"]
    assert r["n_params"] == 241555, r["n_params"]
    assert r["minutes"] == 41.7, r["minutes"]
    assert r["best_val_loss"] == 0.0093, r["best_val_loss"]
    assert r["ema_crude"] == 2.11, r["ema_crude"]
    assert r["ema_assist_v1"] == 91.48, r["ema_assist_v1"]
    assert r["ema_assist_v2"] == 91.50, r["ema_assist_v2"]
    print("[OK] parsea una corrida completa (train + eval)")


def test_truncated_run_gives_none_not_crash():
    """Un job cortado por TIME LIMIT no debe romper la recoleccion ni, peor,
    aparecer como si tuviera resultados."""
    r = collect_results.parse_out(TRUNCATED_OUT)
    assert r["experiment_name"] == "nmr_202k_e3_hp_bs_32"
    assert r["n_params"] == 70163
    assert r["best_val_loss"] is None and r["minutes"] is None
    assert r["ema_crude"] is None and r["ema_assist_v2"] is None
    print("[OK] corrida truncada -> None en vez de excepcion o dato inventado")


def test_empty_text_gives_all_none():
    r = collect_results.parse_out("")
    assert all(v is None for v in r.values()), r
    print("[OK] texto vacio -> todo None")


def test_markdown_marks_missing_runs_as_pending():
    rows = [
        collect_results.parse_out(FAKE_OUT),
        collect_results.parse_out(TRUNCATED_OUT),
    ]
    rows[0]["run"] = "dmodel_128"
    rows[1]["run"] = "bs_32"
    md = collect_results.to_markdown(rows)
    assert "| dmodel_128 |" in md
    assert "91.50" in md
    assert "PENDIENTE" in md, "una corrida sin EMA debe marcarse, no dejarse en blanco"
    print("[OK] la tabla markdown marca las corridas incompletas")


def test_noise_band_from_replicas():
    """Las filas hp_rep_* dan la banda de ruido: media y desvio de la EMA v2."""
    rows = [
        {"run": "rep_42", "ema_assist_v2": 91.35},
        {"run": "rep_43", "ema_assist_v2": 91.95},
        {"run": "rep_44", "ema_assist_v2": 92.14},
        {"run": "dmodel_128", "ema_assist_v2": 91.50},
    ]
    mean, std = collect_results.noise_band(rows)
    assert abs(mean - 91.8133) < 1e-3, mean
    assert abs(std - 0.4125) < 1e-3, std
    print(f"[OK] banda de ruido = {mean:.2f} +- {std:.2f} pp")


def test_noise_band_none_without_replicas():
    assert collect_results.noise_band([{"run": "dmodel_128", "ema_assist_v2": 91.5}]) is None
    print("[OK] sin replicas -> None (no se inventa una banda)")


def test_expected_runs_reads_configs_dir():
    with tempfile.TemporaryDirectory() as td:
        for name, exp in [("a.yaml", "nmr_202k_e3_hp_dmodel_32"),
                           ("b.yaml", "nmr_202k_e3_hp_rep_43")]:
            with open(os.path.join(td, name), "w", encoding="utf-8") as f:
                yaml.safe_dump({"experiment_name": exp}, f)
        runs = collect_results.expected_runs(td)
    assert sorted(runs) == ["dmodel_32", "rep_43"], runs
    print("[OK] expected_runs lee los experiment_name de configs/*.yaml")


def test_collect_fills_pending_for_missing_run():
    with tempfile.TemporaryDirectory() as out_dir, tempfile.TemporaryDirectory() as cfg_dir:
        with open(os.path.join(out_dir, "expE3_hp_1.out"), "w", encoding="utf-8") as f:
            f.write(FAKE_OUT)   # experiment_name = nmr_202k_e3_hp_dmodel_128
        for name, exp in [("a.yaml", "nmr_202k_e3_hp_dmodel_128"),
                           ("b.yaml", "nmr_202k_e3_hp_dmodel_256")]:
            with open(os.path.join(cfg_dir, name), "w", encoding="utf-8") as f:
                yaml.safe_dump({"experiment_name": exp}, f)
        rows = collect_results.collect(out_dir, config_dir=cfg_dir)
    by_run = {r["run"]: r for r in rows}
    assert set(by_run) == {"dmodel_128", "dmodel_256"}, by_run
    assert by_run["dmodel_128"]["ema_assist_v2"] == 91.50
    assert by_run["dmodel_256"]["ema_assist_v2"] is None   # nunca corrio -> PENDIENTE
    print("[OK] collect() completa PENDIENTE una corrida esperada sin .out")


def test_collect_raises_on_duplicate_run():
    with tempfile.TemporaryDirectory() as out_dir:
        with open(os.path.join(out_dir, "expE3_hp_1.out"), "w", encoding="utf-8") as f:
            f.write(FAKE_OUT)
        with open(os.path.join(out_dir, "expE3_hp_2.out"), "w", encoding="utf-8") as f:
            f.write(FAKE_OUT)   # mismo experiment_name -> relanzamiento sin limpiar
        try:
            collect_results.collect(out_dir)
        except ValueError:
            pass
        else:
            raise AssertionError("se esperaba ValueError por .out duplicados")
    print("[OK] collect() rechaza dos .out para la misma corrida (relanzamiento)")


def test_collect_ignores_unrelated_out():
    with tempfile.TemporaryDirectory() as out_dir:
        with open(os.path.join(out_dir, "otro.out"), "w", encoding="utf-8") as f:
            f.write(UNRELATED_OUT)
        rows = collect_results.collect(out_dir)
    assert rows == [], rows
    print("[OK] collect() ignora un .out de otro experimento (prefijo distinto)")


if __name__ == "__main__":
    test_parses_complete_run()
    test_truncated_run_gives_none_not_crash()
    test_empty_text_gives_all_none()
    test_markdown_marks_missing_runs_as_pending()
    test_noise_band_from_replicas()
    test_noise_band_none_without_replicas()
    test_expected_runs_reads_configs_dir()
    test_collect_fills_pending_for_missing_run()
    test_collect_raises_on_duplicate_run()
    test_collect_ignores_unrelated_out()
    print("\n>>> COLLECT_RESULTS OK <<<")
