# coding: ascii
"""
Evaluacion Exp E Fase 3 (dos conjuntos de picos: crosspeaks C-H + 13C, split
CONGELADO) sobre el checkpoint de este experimento (entrenado por train.py).
Mismo patron que experiments/E2_deepsets/evaluate.py: Subset sobre
val_indices_frozen.npy. El modelo se instancia segun model.arch del config
(deepsets | settransformer).

  --oraculo on   -> ajustar_conteo_doble_exacto (EMA ASISTIDA).
  --oraculo off  -> np.clip(np.floor(pred_raw), 0, None) (EMA CRUDA).
  --oraculo both -> corre ambos e imprime la tabla comparativa (DEFAULT).

NO ejecutar hasta tener el checkpoint _best.pth de este experimento.
"""
import os
import argparse
import yaml
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader, Subset


# Oraculo / post-procesamiento: fuente unica (numpy puro, sin torch).
from oraculo import (
    N_CLASSES, IDX_CH2,
    crude_predict, ajustar_conteo_doble_exacto, ajustar_conteo_hetero,
)
from device_utils import pick_device, wants_pin_memory
from config_utils import load_config as _load_config

# --- Clases 19v (orden EXACTO de config/db.yaml, no reordenar) --------------
GROUP_NAMES = [
    "CH3", "CH2", "CH", "Cq",
    "CH3-O", "CH2-O", "CH-O", "Cq-O",
    "CH3-N", "CH2-N", "CH-N", "Cq-N",
    "=CH2", "=CH/Ar", "Cqsp2", "Aldeh", "Imina",
    "C-2X", "C-3X",
]

ENTORNOS = {
    "Alifaticos (sp3)":               ["CH3", "CH2", "CH", "Cq"],
    "Heteroatomicos O/N (sp3)":       ["CH3-O", "CH2-O", "CH-O", "Cq-O",
                                       "CH3-N", "CH2-N", "CH-N", "Cq-N"],
    "Carbonos sp2 (Olef/Arom/C=O)":   ["=CH2", "=CH/Ar", "Cqsp2", "Aldeh", "Imina"],
    "Sistemas X-Multiples (C-2X/3X)": ["C-2X", "C-3X"],
}
ENTORNOS_IDX = {ent: [GROUP_NAMES.index(c) for c in cls]
                for ent, cls in ENTORNOS.items()}


# --- Config ------------------------------------------------------------------
def load_config(path):
    """Alias historico; la logica vive en config_utils (expande ${VAR})."""
    return _load_config(path)


# --- Metricas ----------------------------------------------------------------
def compute_ema(preds, targets):
    return float(np.mean(np.all(preds == targets, axis=1)) * 100)


def compute_mae(preds, targets):
    return np.mean(np.abs(targets - preds), axis=0)


def ema_entorno(preds, targets, indices):
    return float(np.mean(np.all(preds[:, indices] == targets[:, indices], axis=1)) * 100)


def estado(mae):
    if mae < 0.010:
        return "[PERFECTO]"
    if mae < 0.025:
        return "[EXCELENTE]"
    if mae < 0.040:
        return "[BUENO]"
    if mae < 0.060:
        return "[ACEPTABLE]"
    return "[MEJORABLE]"


def analizar_confusiones_cruzadas(all_preds, all_targets):
    error_matrix = np.zeros((N_CLASSES, N_CLASSES), dtype=int)
    for i in range(len(all_targets)):
        diff = all_preds[i] - all_targets[i]
        under = [(g, -diff[g]) for g in range(N_CLASSES) if diff[g] < 0]
        over = [(g, diff[g]) for g in range(N_CLASSES) if diff[g] > 0]
        for gu, cu in under:
            for go, co in over:
                error_matrix[gu][go] += min(cu, co)
    return error_matrix


# --- Reporte por modo --------------------------------------------------------
def report_mode(label, preds, targets):
    n = len(targets)
    ema = compute_ema(preds, targets)
    mae = compute_mae(preds, targets)

    print("\n" + "=" * 60)
    print(f"  MODO {label}  ->  EXACT MATCH ACCURACY: {ema:.2f}%")
    print("=" * 60)

    print(f"\n{'GRUPO':<10} | {'MAE':>6}  Estado")
    print("-" * 40)
    for i, name in enumerate(GROUP_NAMES):
        print(f"{name:<10} | {mae[i]:.4f}  {estado(mae[i])}")

    print("\n  ERRORES POR ENTORNO")
    print("  " + "-" * 40)
    for entorno, indices in ENTORNOS_IDX.items():
        err_mask = np.any(preds[:, indices] != targets[:, indices], axis=1)
        n_err = int(err_mask.sum())
        mae_ent = float(np.mean(np.abs(targets[:, indices] - preds[:, indices])))
        ema_ent = ema_entorno(preds, targets, indices)
        print(f"\n  {entorno}")
        print(f"    Moleculas con error: {n_err} / {n}  ({n_err / n * 100:.1f}%)")
        print(f"    EMA del entorno:     {ema_ent:.2f}%   |   MAE promedio: {mae_ent:.4f}")
        for i in indices:
            err_g = int((preds[:, i] != targets[:, i]).sum())
            print(f"      {GROUP_NAMES[i]:<10}: MAE={mae[i]:.4f} | "
                  f"Mol. con error: {err_g} ({err_g / n * 100:.1f}%)")

    return ema, mae


def print_confusiones(preds, targets):
    n = len(targets)
    error_matrix = analizar_confusiones_cruzadas(preds, targets)
    print("\n" + "=" * 60)
    print("  MAPA DE CONFUSIONES CRUZADAS (solo modo asistido)")
    print("=" * 60)
    for i, name in enumerate(GROUP_NAMES):
        fila = error_matrix[i].copy()
        fila[i] = 0
        total = int(fila.sum())
        if total == 0:
            continue
        top3 = [(GROUP_NAMES[j], fila[j])
                for j in np.argsort(fila)[::-1][:3] if fila[j] > 0]
        pct_g = (preds[:, i] != targets[:, i]).sum() / n * 100
        print(f"  {name:<10} (falla en {pct_g:.1f}% mol) -> confunde con:")
        for dest, cnt in top3:
            print(f"      {dest:<10}: {cnt:>5} senales  ({cnt / total * 100:.1f}%)")


def print_tabla_comparativa(preds_on, preds_off, targets):
    ema_on = compute_ema(preds_on, targets)
    ema_off = compute_ema(preds_off, targets)
    print("\n" + "=" * 60)
    print("  TABLA COMPARATIVA: EMA CRUDA vs ASISTIDA")
    print("=" * 60)
    print(f"\n  {'':<32}{'CRUDA':>10}{'ASISTIDA':>12}{'Delta':>10}")
    print("  " + "-" * 62)
    print(f"  {'EMA GLOBAL':<32}{ema_off:>9.2f}%{ema_on:>11.2f}%{ema_on - ema_off:>+9.2f}")
    for entorno, indices in ENTORNOS_IDX.items():
        e_off = ema_entorno(preds_off, targets, indices)
        e_on = ema_entorno(preds_on, targets, indices)
        print(f"  {entorno:<32}{e_off:>9.2f}%{e_on:>11.2f}%{e_on - e_off:>+9.2f}")
    print("\n  Delta = ASISTIDA - CRUDA. Un Delta grande (>10 pp) indica que la")
    print("  EMA reportada depende fuertemente del oraculo de doble restriccion.")


def print_tabla_comparativa_3(preds_off, preds_on, preds_v2, targets):
    """Cruda vs Asistida v1 (doble) vs Asistida v2 (hetero)."""
    e_off = compute_ema(preds_off, targets)
    e_on = compute_ema(preds_on, targets)
    e_v2 = compute_ema(preds_v2, targets)
    print("\n" + "=" * 68)
    print("  TABLA COMPARATIVA: CRUDA vs ASISTIDA v1 (doble) vs ASISTIDA v2 (hetero)")
    print("=" * 68)
    print(f"\n  {'':<28}{'CRUDA':>9}{'ASIST v1':>10}{'ASIST v2':>10}{'v2-v1':>9}")
    print("  " + "-" * 66)
    print(f"  {'EMA GLOBAL':<28}{e_off:>8.2f}%{e_on:>9.2f}%{e_v2:>9.2f}%{e_v2 - e_on:>+8.2f}")
    for entorno, indices in ENTORNOS_IDX.items():
        a = ema_entorno(preds_off, targets, indices)
        b = ema_entorno(preds_on, targets, indices)
        c = ema_entorno(preds_v2, targets, indices)
        print(f"  {entorno:<28}{a:>8.2f}%{b:>9.2f}%{c:>9.2f}%{c - b:>+8.2f}")
    print("\n  v2-v1 = ganancia del zeroing por heteroatomos ausentes (N==0/O==0/N+O<2/3).")
    print("  Si v2-v1 ~ 0, la confusion vive en moleculas CON el heteroatomo (fuera de")
    print("  alcance del oraculo por conteos: ver el spec, seccion 6).")


# --- Main --------------------------------------------------------------------
def evaluate(config_path, oraculo, eval_batch_size):
    # Import perezoso: dataset trae rdkit; el test de oraculo no lo necesita.
    from dataset_e3 import NMRTwoSetsDataset
    from train import build_model

    cfg = load_config(config_path)

    base_dir = Path(cfg["paths"]["base_dir"])
    peaks_ch = base_dir / cfg["paths"]["peaks_ch_filename"]
    peaks_13c = base_dir / cfg["paths"]["peaks_13c_filename"]
    labels_path = base_dir / cfg["paths"]["labels_filename"]
    smiles_path = base_dir / cfg["paths"]["smiles_filename"]
    ckpt_path = base_dir / cfg["paths"]["checkpoint_dir"] / f"{cfg['experiment_name']}_best.pth"
    val_indices_path = base_dir / cfg["paths"]["val_indices_filename"]

    num_workers = int(cfg["system"]["num_workers"])   # 0 (rule 1)

    run_off = oraculo in ("off", "both", "all")
    run_on = oraculo in ("on", "both", "all")
    run_v2 = oraculo in ("v2", "all")
    modes = [m for m, f in (("off", run_off), ("on", run_on), ("v2", run_v2)) if f]

    print("=" * 60)
    print("  EVALUACION EXP E FASE 3 (dos conjuntos) - SPLIT CONGELADO")
    print("=" * 60)
    print(f"-> Experimento (checkpoint): {cfg['experiment_name']}  | arch: {cfg['model']['arch']}")
    print(f"-> Modos: {modes}   | idx_ch2: {IDX_CH2}")
    print(f"-> num_workers={num_workers} (rule 1)  batch_size={eval_batch_size}")

    device = pick_device(cfg["system"].get("device", "auto"))
    print(f"-> Dispositivo: {device.type.upper()}")

    if not os.path.exists(ckpt_path):
        print(f"\n[ERROR] No se encontro el checkpoint en:\n  {ckpt_path}")
        print("        Corri primero train.py.")
        return
    if not os.path.exists(val_indices_path):
        print(f"\n[ERROR] No se encontro el split congelado en:\n  {val_indices_path}")
        print("        Corri primero experiments/D_val_congelado/split.py (Exp D).")
        return

    full_dataset = NMRTwoSetsDataset(str(peaks_ch), str(peaks_13c), str(labels_path),
                                     str(smiles_path), cfg["normalization"])
    val_indices = np.load(val_indices_path)
    val_ds = Subset(full_dataset, val_indices.tolist())

    use_pin = bool(cfg["system"].get("pin_memory", False)) and wants_pin_memory(device)
    val_loader = DataLoader(val_ds, batch_size=eval_batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=use_pin)

    model = build_model(cfg, num_classes=N_CLASSES).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    all_targets, all_pred_on, all_pred_off, all_pred_v2 = [], [], [], []
    with torch.no_grad():
        for inputs, targets in val_loader:
            pch = inputs[0].to(device); mch = inputs[1].to(device)
            p13 = inputs[2].to(device); m13 = inputs[3].to(device)
            cond = inputs[4].to(device)
            pred_raw = model(pch, mch, p13, m13, cond).cpu().numpy()
            targs = targets.cpu().numpy().astype(int)
            cond_np = cond.cpu().numpy()
            all_targets.append(targs)

            if run_off:
                all_pred_off.append(crude_predict(pred_raw))
            if run_on:
                batch_on = np.empty_like(targs)
                for k in range(len(targs)):
                    batch_on[k] = ajustar_conteo_doble_exacto(
                        pred_raw[k], int(cond_np[k, 0]), int(cond_np[k, 1]))
                all_pred_on.append(batch_on)
            if run_v2:
                batch_v2 = np.empty_like(targs)
                for k in range(len(targs)):
                    # cond: [total, ch2, C, H, N, O, S, Hal] -> N=idx4, O=idx5
                    batch_v2[k] = ajustar_conteo_hetero(
                        pred_raw[k], int(cond_np[k, 0]), int(cond_np[k, 1]),
                        int(cond_np[k, 4]), int(cond_np[k, 5]))
                all_pred_v2.append(batch_v2)

    all_targets = np.vstack(all_targets)
    print(f"\n-> Set de validacion (congelado): {len(all_targets)} moleculas")

    preds_on = np.vstack(all_pred_on) if run_on else None
    preds_off = np.vstack(all_pred_off) if run_off else None
    preds_v2 = np.vstack(all_pred_v2) if run_v2 else None

    if run_off:
        report_mode("CRUDO (--oraculo off)", preds_off, all_targets)
    if run_on:
        report_mode("ASISTIDO v1 (--oraculo on)", preds_on, all_targets)
    if run_v2:
        report_mode("ASISTIDO v2 hetero (--oraculo v2)", preds_v2, all_targets)
    # Mapa de confusiones sobre el mejor asistido disponible.
    if run_v2:
        print_confusiones(preds_v2, all_targets)
    elif run_on:
        print_confusiones(preds_on, all_targets)

    if run_off and run_on and run_v2:
        print_tabla_comparativa_3(preds_off, preds_on, preds_v2, all_targets)
    elif run_on and run_off:
        print_tabla_comparativa(preds_on, preds_off, all_targets)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Eval Exp E Fase 3 (split congelado)")
    parser.add_argument("--config", type=str, default="config_deepsets.yaml",
                        help="Config del experimento (deepsets o settransformer).")
    parser.add_argument("--oraculo", choices=["on", "off", "both", "v2", "all"],
                        default="all",
                        help="off=cruda, on=asistida v1, v2=asistida hetero, "
                             "both=cruda+v1, all=cruda+v1+v2 con tabla 3-vias (default).")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Batch size de evaluacion.")
    args = parser.parse_args()
    evaluate(args.config, args.oraculo, args.batch_size)
