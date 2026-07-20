# coding: ascii
"""
Evaluacion Exp C (V10 + GAP, sin regularizacion, split CONGELADO) sobre
el checkpoint de este experimento (entrenado por train.py, Task 6).
Mismo patron que experiments/D_val_congelado/evaluate.py y
experiments/B_regularizacion/evaluate.py: Subset sobre
val_indices_frozen.npy en vez de random_split.

  --oraculo on   -> ajustar_conteo_doble_exacto (EMA ASISTIDA).
  --oraculo off  -> np.clip(np.floor(pred_raw), 0, None) (EMA CRUDA).
  --oraculo both -> corre ambos e imprime la tabla comparativa (DEFAULT).

Config: UN SOLO config.yaml (ver Task 5). El checkpoint se deriva IGUAL
que el guardado del training: {base_dir}/{paths.checkpoint_dir}/{experiment_name}_best.pth.

Reglas (CLAUDE.md):
  - num_workers=0 (h5py no es fork-safe; rule 1).
  - val_indices_frozen.npy debe existir (Exp D ya lo genero).

NO ejecutar hasta tener el checkpoint _best.pth de este experimento (Task 6).
"""
import os
import argparse
import yaml
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader, Subset


# --- Clases 19v (orden EXACTO de config/db.yaml, no reordenar) --------------
GROUP_NAMES = [
    "CH3", "CH2", "CH", "Cq",
    "CH3-O", "CH2-O", "CH-O", "Cq-O",
    "CH3-N", "CH2-N", "CH-N", "Cq-N",
    "=CH2", "=CH/Ar", "Cqsp2", "Aldeh", "Imina",
    "C-2X", "C-3X",
]
N_CLASSES = 19
IDX_CH2 = [1, 5, 9, 12]   # CH2, CH2-O, CH2-N, =CH2

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
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# --- Post-procesamiento ------------------------------------------------------
def crude_predict(pred_cruda):
    """Modo CRUDO: floor con clip a >=0. Ignora el condicionante por completo."""
    return np.clip(np.floor(pred_cruda), 0, None).astype(int)


def ajustar_conteo_doble_exacto(pred_cruda, total_real, ch2_real):
    """Modo ASISTIDO (oraculo): fuerza sum(pred)==total_real y
    sum(pred[IDX_CH2])==ch2_real, repartiendo por el resto decimal."""
    pred_int = np.floor(pred_cruda).astype(int)
    restos = pred_cruda - pred_int
    idx_resto = [i for i in range(N_CLASSES) if i not in IDX_CH2]

    ch2_asignados = sum(pred_int[i] for i in IDX_CH2)
    ch2_faltantes = int(ch2_real - ch2_asignados)
    if ch2_faltantes > 0:
        for i in sorted(IDX_CH2, key=lambda i: restos[i])[-ch2_faltantes:]:
            pred_int[i] += 1
    elif ch2_faltantes < 0:
        sobran = abs(ch2_faltantes)
        for i in sorted(IDX_CH2, key=lambda i: restos[i]):
            if pred_int[i] > 0:
                pred_int[i] -= 1
                sobran -= 1
                if sobran == 0:
                    break

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


# --- Main --------------------------------------------------------------------
def evaluate(config_path, oraculo, eval_batch_size):
    # Import perezoso: dataset_v10 trae rdkit/h5py; el smoke test no lo necesita.
    from dataset_v10 import NMRDataset
    from model_c import NMR_Net

    cfg = load_config(config_path)

    base_dir = Path(cfg["paths"]["base_dir"])
    h5_path = base_dir / cfg["paths"]["h5_filename"]
    labels_path = base_dir / cfg["paths"]["labels_filename"]
    smiles_path = base_dir / cfg["paths"]["smiles_filename"]
    ckpt_path = base_dir / cfg["paths"]["checkpoint_dir"] / f"{cfg['experiment_name']}_best.pth"
    val_indices_path = base_dir / cfg["paths"]["val_indices_filename"]

    num_workers = int(cfg["system"]["num_workers"])   # 0 (rule 1)

    modes = ["on", "off"] if oraculo == "both" else [oraculo]
    run_on = "on" in modes
    run_off = "off" in modes

    print("=" * 60)
    print("  EVALUACION EXP C (V10 + GAP, sin regularizacion) - SPLIT CONGELADO")
    print("=" * 60)
    print(f"-> Experimento (checkpoint): {cfg['experiment_name']}")
    print(f"-> Modos: {modes}   | idx_ch2: {IDX_CH2}")
    print(f"-> num_workers={num_workers} (rule 1)  batch_size={eval_batch_size}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"-> Dispositivo: {device.type.upper()}")

    if not os.path.exists(ckpt_path):
        print(f"\n[ERROR] No se encontro el checkpoint en:\n  {ckpt_path}")
        print("        Corri primero train.py (Task 6).")
        return
    if not os.path.exists(val_indices_path):
        print(f"\n[ERROR] No se encontro el split congelado en:\n  {val_indices_path}")
        print("        Corri primero experiments/D_val_congelado/split.py (Exp D).")
        return

    full_dataset = NMRDataset(str(h5_path), str(labels_path), str(smiles_path))
    val_indices = np.load(val_indices_path)
    val_ds = Subset(full_dataset, val_indices.tolist())

    use_pin = bool(cfg["system"].get("pin_memory", False)) and device.type == "cuda"
    val_loader = DataLoader(val_ds, batch_size=eval_batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=use_pin)

    model = NMR_Net(num_classes=N_CLASSES).to(device)
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
                        pred_raw[k], int(cond_np[k, 0]), int(cond_np[k, 1]))
                all_pred_on.append(batch_on)

    all_targets = np.vstack(all_targets)
    print(f"\n-> Set de validacion (congelado): {len(all_targets)} moleculas")

    preds_on = np.vstack(all_pred_on) if run_on else None
    preds_off = np.vstack(all_pred_off) if run_off else None

    if run_off:
        report_mode("CRUDO (--oraculo off)", preds_off, all_targets)
    if run_on:
        report_mode("ASISTIDO (--oraculo on)", preds_on, all_targets)
        print_confusiones(preds_on, all_targets)

    if run_on and run_off:
        print_tabla_comparativa(preds_on, preds_off, all_targets)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Eval Exp C (split congelado)")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Config unico (ver Task 5 del plan).")
    parser.add_argument("--oraculo", choices=["on", "off", "both"], default="both",
                        help="on=asistida, off=cruda, both=ambas + tabla (default).")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Batch size de evaluacion.")
    args = parser.parse_args()
    evaluate(args.config, args.oraculo, args.batch_size)
