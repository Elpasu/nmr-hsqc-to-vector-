# coding: ascii
"""
Evaluacion V10 (HSQC 2 canales + Formula Molecular + 19 clases) - EXP A.

Objetivo del Exp A: separar cuanto del rendimiento viene del modelo y cuanto
del post-procesamiento que fuerza los conteos. Se evaluan DOS modos sobre el
MISMO val set en una sola corrida:

  --oraculo on   -> ajustar_conteo_doble_exacto: obliga a la prediccion a sumar
                    exactamente total_senales y total_CH2 (ambos derivados del
                    target via el condicionante) = EMA ASISTIDA.
  --oraculo off  -> np.clip(np.floor(pred_raw), 0, None): sin forzar sumas
                    = EMA CRUDA.
  --oraculo both -> corre ambos e imprime la tabla comparativa (DEFAULT).

Reglas (CLAUDE.md):
  - Nada hardcodeado: paths, val_split, seed, num_workers y nombres de clase
    salen de config/db.yaml. El checkpoint y los modos salen del config de eval.
  - num_workers=0 (h5py no es fork-safe; rule 1).
  - num_classes=19 y el orden de clases se leen tal cual de db.yaml (rule 7).
  - El split se reproduce IGUAL que train_v10.py (mismo seed y val_split) para
    que la EMA corresponda al val real del V10.

NO ejecutar hasta tener el checkpoint _best.pth del V10.
"""
import os
import argparse
import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from model_v10 import NMR_Net


# --- Agrupaciones de analisis (por NOMBRE de clase, no por indice) -----------
# Se resuelven a indices contra classes_19v de db.yaml, asi nunca se desalinean.
ENTORNOS = {
    "Alifaticos (sp3)":               ["CH3", "CH2", "CH", "Cq"],
    "Heteroatomicos O/N (sp3)":       ["CH3-O", "CH2-O", "CH-O", "Cq-O",
                                       "CH3-N", "CH2-N", "CH-N", "Cq-N"],
    "Carbonos sp2 (Olef/Arom/C=O)":   ["=CH2", "=CH/Ar", "Cqsp2", "Aldeh", "Imina"],
    "Sistemas X-Multiples (C-2X/3X)": ["C-2X", "C-3X"],
}

# Clases CH2 (para el condicionante total_CH2). Se derivan por nombre.
CH2_CLASS_NAMES = {"CH2", "CH2-O", "CH2-N", "=CH2"}


# --- Config ------------------------------------------------------------------
def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# --- Post-procesamiento ------------------------------------------------------
def crude_predict(pred_cruda):
    """Modo CRUDO: floor con clip a >=0. Ignora el condicionante por completo."""
    return np.clip(np.floor(pred_cruda), 0, None).astype(int)


def ajustar_conteo_doble_exacto(pred_cruda, total_real, ch2_real, idx_ch2, n_classes):
    """Modo ASISTIDO (oraculo): fuerza sum(pred)==total_real y
    sum(pred[idx_ch2])==ch2_real, repartiendo por el resto decimal.
    Misma logica que evaluate_v9.py, parametrizada por idx_ch2 / n_classes."""
    pred_int = np.floor(pred_cruda).astype(int)
    restos = pred_cruda - pred_int
    idx_resto = [i for i in range(n_classes) if i not in idx_ch2]

    # 1) Restriccion sobre los CH2
    ch2_asignados = sum(pred_int[i] for i in idx_ch2)
    ch2_faltantes = int(ch2_real - ch2_asignados)
    if ch2_faltantes > 0:
        for i in sorted(idx_ch2, key=lambda i: restos[i])[-ch2_faltantes:]:
            pred_int[i] += 1
    elif ch2_faltantes < 0:
        sobran = abs(ch2_faltantes)
        for i in sorted(idx_ch2, key=lambda i: restos[i]):
            if pred_int[i] > 0:
                pred_int[i] -= 1
                sobran -= 1
                if sobran == 0:
                    break

    # 2) Restriccion sobre el resto (total - CH2)
    resto_real = total_real - ch2_real
    resto_asignados = sum(pred_int[i] for i in idx_resto)
    resto_faltantes = int(resto_real - resto_asignados)
    if resto_faltantes > 0:
        for i in sorted(idx_resto, key=lambda i: restos[i])[-resto_faltantes:]:
            pred_int[i] += 1
    elif resto_faltantes < 0:
        sobran = abs(resto_faltantes)
        for i in sorted(idx_resto, key=lambda i: restos[i]):
            if pred_int[i] > 0:
                pred_int[i] -= 1
                sobran -= 1
                if sobran == 0:
                    break

    return pred_int


# --- Metricas ----------------------------------------------------------------
def compute_ema(preds, targets):
    return float(np.mean(np.all(preds == targets, axis=1)) * 100)


def compute_mae(preds, targets):
    return np.mean(np.abs(targets - preds), axis=0)


def ema_entorno(preds, targets, indices):
    return float(np.mean(np.all(preds[:, indices] == targets[:, indices], axis=1)) * 100)


def estado(mae):
    # Escala de rigor NUEVA (reporte 20/04/2026).
    if mae < 0.010:
        return "[PERFECTO]"
    if mae < 0.025:
        return "[EXCELENTE]"
    if mae < 0.040:
        return "[BUENO]"
    if mae < 0.060:
        return "[ACEPTABLE]"
    return "[MEJORABLE]"


def analizar_confusiones_cruzadas(all_preds, all_targets, n_classes):
    error_matrix = np.zeros((n_classes, n_classes), dtype=int)
    for i in range(len(all_targets)):
        diff = all_preds[i] - all_targets[i]
        under = [(g, -diff[g]) for g in range(n_classes) if diff[g] < 0]
        over = [(g, diff[g]) for g in range(n_classes) if diff[g] > 0]
        for gu, cu in under:
            for go, co in over:
                error_matrix[gu][go] += min(cu, co)
    return error_matrix


# --- Reporte por modo --------------------------------------------------------
def report_mode(label, preds, targets, group_names, entornos_idx):
    n = len(targets)
    ema = compute_ema(preds, targets)
    mae = compute_mae(preds, targets)

    print("\n" + "=" * 60)
    print(f"  MODO {label}  ->  EXACT MATCH ACCURACY: {ema:.2f}%")
    print("=" * 60)

    print(f"\n{'GRUPO':<10} | {'MAE':>6}  Estado")
    print("-" * 40)
    for i, name in enumerate(group_names):
        print(f"{name:<10} | {mae[i]:.4f}  {estado(mae[i])}")

    print("\n  ERRORES POR ENTORNO")
    print("  " + "-" * 40)
    for entorno, indices in entornos_idx.items():
        err_mask = np.any(preds[:, indices] != targets[:, indices], axis=1)
        n_err = int(err_mask.sum())
        mae_ent = float(np.mean(np.abs(targets[:, indices] - preds[:, indices])))
        ema_ent = ema_entorno(preds, targets, indices)
        print(f"\n  {entorno}")
        print(f"    Moleculas con error: {n_err} / {n}  ({n_err / n * 100:.1f}%)")
        print(f"    EMA del entorno:     {ema_ent:.2f}%   |   MAE promedio: {mae_ent:.4f}")
        for i in indices:
            err_g = int((preds[:, i] != targets[:, i]).sum())
            print(f"      {group_names[i]:<10}: MAE={mae[i]:.4f} | "
                  f"Mol. con error: {err_g} ({err_g / n * 100:.1f}%)")

    return ema, mae


def print_confusiones(preds, targets, group_names):
    n = len(targets)
    n_classes = len(group_names)
    error_matrix = analizar_confusiones_cruzadas(preds, targets, n_classes)
    print("\n" + "=" * 60)
    print("  MAPA DE CONFUSIONES CRUZADAS (solo modo asistido)")
    print("=" * 60)
    for i, name in enumerate(group_names):
        fila = error_matrix[i].copy()
        fila[i] = 0
        total = int(fila.sum())
        if total == 0:
            continue
        top3 = [(group_names[j], fila[j])
                for j in np.argsort(fila)[::-1][:3] if fila[j] > 0]
        pct_g = (preds[:, i] != targets[:, i]).sum() / n * 100
        print(f"  {name:<10} (falla en {pct_g:.1f}% mol) -> confunde con:")
        for dest, cnt in top3:
            print(f"      {dest:<10}: {cnt:>5} senales  ({cnt / total * 100:.1f}%)")


def print_tabla_comparativa(preds_on, preds_off, targets, group_names, entornos_idx):
    ema_on = compute_ema(preds_on, targets)
    ema_off = compute_ema(preds_off, targets)
    print("\n" + "=" * 60)
    print("  TABLA COMPARATIVA: EMA CRUDA vs ASISTIDA")
    print("=" * 60)
    print(f"\n  {'':<32}{'CRUDA':>10}{'ASISTIDA':>12}{'Delta':>10}")
    print("  " + "-" * 62)
    print(f"  {'EMA GLOBAL':<32}{ema_off:>9.2f}%{ema_on:>11.2f}%{ema_on - ema_off:>+9.2f}")
    for entorno, indices in entornos_idx.items():
        e_off = ema_entorno(preds_off, targets, indices)
        e_on = ema_entorno(preds_on, targets, indices)
        print(f"  {entorno:<32}{e_off:>9.2f}%{e_on:>11.2f}%{e_on - e_off:>+9.2f}")
    print("\n  Delta = ASISTIDA - CRUDA. Un Delta grande (>10 pp) indica que la")
    print("  EMA reportada depende fuertemente del oraculo de doble restriccion.")


# --- Main --------------------------------------------------------------------
def evaluate(db_config_path, eval_config_path, oraculo):
    # Import perezoso: dataset_v10 trae rdkit/h5py; el smoke test no lo necesita.
    from dataset_v10 import NMRDataset

    db = load_yaml(db_config_path)
    ev = load_yaml(eval_config_path)

    group_names = list(db["classes_19v"])
    n_classes = int(db["model"]["num_classes"])
    assert len(group_names) == n_classes, \
        f"classes_19v ({len(group_names)}) != num_classes ({n_classes})"

    idx_ch2 = [i for i, name in enumerate(group_names) if name in CH2_CLASS_NAMES]
    entornos_idx = {ent: [group_names.index(c) for c in cls]
                    for ent, cls in ENTORNOS.items()}

    base_dir = db["data"]["base_dir"]
    h5_path = os.path.join(base_dir, db["data"]["h5_v3"])
    labels_path = os.path.join(base_dir, db["data"]["labels_19v"])
    smiles_path = os.path.join(base_dir, db["data"]["smiles"])
    ckpt_path = os.path.join(base_dir, ev["checkpoint"]["dir"],
                             ev["checkpoint"]["filename"])

    val_split = float(db["hyperparameters"]["val_split"])
    seed = int(db["hyperparameters"]["seed"])
    num_workers = int(db["system"]["num_workers"])   # 0 (rule 1)
    batch_size = int(ev["evaluation"]["eval_batch_size"])

    modes = ["on", "off"] if oraculo == "both" else [oraculo]
    run_on = "on" in modes
    run_off = "off" in modes

    print("=" * 60)
    print("  EVALUACION V10 (2CH + FM + 19v) - EXP A: CRUDA vs ASISTIDA")
    print("=" * 60)
    print(f"-> Modos: {modes}   | idx_ch2 derivados: {idx_ch2}")
    print(f"-> num_workers={num_workers} (rule 1)  batch_size={batch_size}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"-> Dispositivo: {device.type.upper()}")

    if not os.path.exists(ckpt_path):
        print(f"\n[ERROR] No se encontro el checkpoint en:\n  {ckpt_path}")
        print("        El V10 todavia esta entrenando? Correr cuando exista _best.pth.")
        return

    # Split IGUAL que train_v10.py (mismo seed y val_split -> mismo val set).
    full_dataset = NMRDataset(h5_path, labels_path, smiles_path)
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    generator = torch.Generator().manual_seed(seed)
    _, val_ds = random_split(full_dataset, [train_size, val_size], generator=generator)

    use_pin = bool(db["system"].get("pin_memory", False)) and device.type == "cuda"
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=use_pin)

    model = NMR_Net(num_classes=n_classes).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    all_targets, all_pred_on, all_pred_off = [], [], []
    with torch.no_grad():
        for inputs, targets in val_loader:
            hsqc = inputs[0].to(device)
            proj = inputs[1].to(device)
            cond = inputs[2].to(device)
            pred_raw = model(hsqc, proj, cond).cpu().numpy()
            targs = targets.cpu().numpy().astype(int)
            cond_np = cond.cpu().numpy()
            all_targets.append(targs)

            if run_off:
                all_pred_off.append(crude_predict(pred_raw))
            if run_on:
                batch_on = np.empty_like(targs)
                for k in range(len(targs)):
                    batch_on[k] = ajustar_conteo_doble_exacto(
                        pred_raw[k], int(cond_np[k, 0]), int(cond_np[k, 1]),
                        idx_ch2, n_classes)
                all_pred_on.append(batch_on)

    all_targets = np.vstack(all_targets)
    print(f"\n-> Set de validacion: {len(all_targets)} moleculas")

    preds_on = np.vstack(all_pred_on) if run_on else None
    preds_off = np.vstack(all_pred_off) if run_off else None

    if run_off:
        report_mode("CRUDO (--oraculo off)", preds_off, all_targets,
                    group_names, entornos_idx)
    if run_on:
        report_mode("ASISTIDO (--oraculo on)", preds_on, all_targets,
                    group_names, entornos_idx)
        print_confusiones(preds_on, all_targets, group_names)

    if run_on and run_off:
        print_tabla_comparativa(preds_on, preds_off, all_targets,
                                group_names, entornos_idx)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Eval V10 - EXP A (cruda vs asistida)")
    parser.add_argument("--db-config", type=str, default="config/db.yaml",
                        help="Fuente unica de verdad (paths, seed, clases).")
    parser.add_argument("--config", type=str, default="configs/config_V11a_eval.yaml",
                        help="Config del eval (checkpoint + modos).")
    parser.add_argument("--oraculo", choices=["on", "off", "both"], default="both",
                        help="on=asistida, off=cruda, both=ambas + tabla (default).")
    args = parser.parse_args()
    evaluate(args.db_config, args.config, args.oraculo)
