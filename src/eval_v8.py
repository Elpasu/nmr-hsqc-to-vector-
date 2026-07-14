# coding: ascii
"""
Evaluacion V8 (17 CLASES, HSQC 2 canales, sin formula molecular)
"""
import torch
import numpy as np
import os
from torch.utils.data import DataLoader, random_split
from dataset_v8 import NMRDataset
from model_v8 import NMR_Net

# --- CONFIGURACION ---
DATA_DIR       = "/home/lpassaglia.iquir/DB_144K"
H5_FILE        = os.path.join(DATA_DIR, "nmr_dataset_v3_144280.h5")
LABEL_FILE     = os.path.join(DATA_DIR, "vectors_13c_17v_144280.npy")

CHECKPOINT_DIR = "/home/lpassaglia.iquir/DB_144K/checkpoints_V8"
MODEL_NAME     = "nmr_144k_v8_best.pth"
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, MODEL_NAME)

BATCH_SIZE = 256
VAL_SPLIT  = 0.1
SEED       = 42

GROUP_NAMES = [
    "CH3",     # 0   alifatico
    "CH2",     # 1   alifatico
    "CH",      # 2   alifatico
    "Cq",      # 3   alifatico
    "CH3-O",   # 4   con oxigeno
    "CH2-O",   # 5   con oxigeno
    "CH-O",    # 6   con oxigeno (incluye hemiacetales)
    "Cq-O",    # 7   con oxigeno
    "CH3-N",   # 8   con nitrogeno
    "CH2-N",   # 9   con nitrogeno
    "CH-N",    # 10  con nitrogeno
    "Cq-N",    # 11  con nitrogeno
    "=CH2",    # 12  sp2
    "=CH/Ar",  # 13  sp2
    "Cqsp2",   # 14  sp2 cuaternario
    "Aldeh",   # 15  aldehido
    "Imina",   # 16  imina
]
N_CLASSES = 17

ENTORNOS = {
    "Alifaticos (sp3)":      [0, 1, 2, 3],
    "Con oxigeno (sp3-O)":   [4, 5, 6, 7],
    "Con nitrogeno (sp3-N)": [8, 9, 10, 11],
    "sp2":                   [12, 13, 14, 15, 16],
}

MULT_GROUPS = {
    "Alifatico": ([0, 1, 2],    ["CH3",   "CH2",   "CH"  ]),
    "Con-O":     ([4, 5, 6],    ["CH3-O", "CH2-O", "CH-O"]),
    "Con-N":     ([8, 9, 10],   ["CH3-N", "CH2-N", "CH-N"]),
}


def ajustar_conteo_doble_exacto(pred_cruda, total_real, ch2_real):
    pred_int = np.floor(pred_cruda).astype(int)
    restos   = pred_cruda - pred_int
    idx_ch2  = [1, 5, 9, 12]
    idx_rest = [i for i in range(N_CLASSES) if i not in idx_ch2]

    ch2_asign = sum(pred_int[i] for i in idx_ch2)
    falt_ch2  = int(ch2_real - ch2_asign)
    if falt_ch2 > 0:
        for i in sorted(idx_ch2, key=lambda i: restos[i])[-falt_ch2:]:
            pred_int[i] += 1
    elif falt_ch2 < 0:
        sobran = abs(falt_ch2)
        for i in sorted(idx_ch2, key=lambda i: restos[i]):
            if pred_int[i] > 0:
                pred_int[i] -= 1
                sobran -= 1
                if sobran == 0:
                    break

    resto_real  = total_real - ch2_real
    resto_asign = sum(pred_int[i] for i in idx_rest)
    falt_rest   = int(resto_real - resto_asign)
    if falt_rest > 0:
        for i in sorted(idx_rest, key=lambda i: restos[i])[-falt_rest:]:
            pred_int[i] += 1
    elif falt_rest < 0:
        sobrans = abs(falt_rest)
        for i in sorted(idx_rest, key=lambda i: restos[i]):
            if pred_int[i] > 0:
                pred_int[i] -= 1
                sobrans -= 1
                if sobrans == 0:
                    break
    return pred_int


def analizar_confusion_multiplicidad(all_preds, all_targets):
    n = len(all_targets)
    matrices = {}

    for entorno, (indices, names) in MULT_GROUPS.items():
        conf = np.zeros((3, 3), dtype=int)
        for i in range(n):
            t = all_targets[i]
            p = all_preds[i]
            for ri, ridx in enumerate(indices):
                for pi, pidx in enumerate(indices):
                    if ri != pi:
                        transf = min(max(0, t[ridx]-p[ridx]),
                                     max(0, p[pidx]-t[pidx]))
                        conf[ri][pi] += transf
                    else:
                        conf[ri][pi] += t[ridx]
        matrices[entorno] = (conf, names)

    mols_error_real = sum(
        1 for i in range(n)
        if any(
            min(max(0, all_targets[i][ridx] - all_preds[i][ridx]),
                max(0, all_preds[i][pidx] - all_targets[i][pidx])) > 0
            for _, (indices, _) in MULT_GROUPS.items()
            for ri, ridx in enumerate(indices)
            for pi, pidx in enumerate(indices)
            if ri != pi
        )
    )
    return matrices, mols_error_real, (mols_error_real / n * 100)


def analizar_confusiones_cruzadas(all_preds, all_targets):
    error_matrix = np.zeros((N_CLASSES, N_CLASSES), dtype=int)
    for i in range(len(all_targets)):
        t    = all_targets[i]
        p    = all_preds[i]
        diff = p - t
        under = [(g, -diff[g]) for g in range(N_CLASSES) if diff[g] < 0]
        over  = [(g,  diff[g]) for g in range(N_CLASSES) if diff[g] > 0]
        for (gu, cu) in under:
            for (go, co) in over:
                error_matrix[gu][go] += min(cu, co)
    return error_matrix


def evaluate():
    print("=" * 60)
    print("  EVALUACION V8 (17 CLASES, HSQC 2CH, SIN FM)")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"-> Dispositivo: {device.type.upper()}")

    if not os.path.exists(CHECKPOINT_PATH):
        print(f"[ERROR] No se encontro: {CHECKPOINT_PATH}")
        return

    full_dataset = NMRDataset(H5_FILE, LABEL_FILE)
    val_size   = int(len(full_dataset) * VAL_SPLIT)
    train_size = len(full_dataset) - val_size
    generator  = torch.Generator().manual_seed(SEED)
    _, val_ds  = random_split(full_dataset, [train_size, val_size],
                               generator=generator)

    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=4,
                            pin_memory=torch.cuda.is_available())

    model = NMR_Net(num_classes=17).to(device)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.eval()

    all_preds   = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in val_loader:
            hsqc      = inputs[0].to(device)
            proj      = inputs[1].to(device)
            cond      = inputs[2].to(device)
            outputs   = model(hsqc, proj, cond)

            pred_raw  = outputs.cpu().numpy()
            targs     = targets.cpu().numpy().astype(int)
            cond_np   = cond.cpu().numpy()
            total_sig = cond_np[:, 0]
            total_ch2 = cond_np[:, 1]

            for k in range(len(targs)):
                p = ajustar_conteo_doble_exacto(
                    pred_raw[k], int(total_sig[k]), int(total_ch2[k]))
                all_preds.append(p)
                all_targets.append(targs[k])

    all_preds   = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)
    n_mols      = len(all_targets)

    print(f"-> Set de validacion: {n_mols} moleculas\n")

    # --- EMA y MAE ---
    perfect = np.all(all_preds == all_targets, axis=1)
    ema     = perfect.mean() * 100
    mae     = np.mean(np.abs(all_targets - all_preds), axis=0)

    print("=" * 60)
    print(f"  EXACT MATCH ACCURACY: {ema:.2f}%")
    print("=" * 60)
    print(f"\n{'GRUPO':<10} | {'MAE':>6}  Estado")
    print("-" * 40)
    for i, name in enumerate(GROUP_NAMES):
        m = mae[i]
        estado = "[EXCELENTE]" if m < 0.05 else "[ACEPTABLE]" if m < 0.1 else "[MEJORABLE]"
        print(f"{name:<10} | {m:.4f}  {estado}")

    # --- Errores por entorno ---
    print("\n" + "=" * 60)
    print("  ERRORES POR ENTORNO")
    print("=" * 60)
    for entorno, indices in ENTORNOS.items():
        err_mask = np.any(all_preds[:, indices] != all_targets[:, indices], axis=1)
        n_err    = err_mask.sum()
        mae_ent  = np.mean(np.abs(all_targets[:, indices] - all_preds[:, indices]))
        print(f"\n  {entorno}")
        print(f"    Moleculas con error: {n_err} / {n_mols}  ({n_err/n_mols*100:.1f}%)")
        print(f"    MAE promedio:        {mae_ent:.4f}")
        for i in indices:
            err_g = (all_preds[:, i] != all_targets[:, i]).sum()
            print(f"      {GROUP_NAMES[i]:<10}: MAE={mae[i]:.4f} | "
                  f"Mol. con error: {err_g} ({err_g/n_mols*100:.1f}%)")

    # --- Confusion de multiplicidad ---
    print("\n" + "=" * 60)
    print("  CONFUSION DE MULTIPLICIDAD")
    print("=" * 60)
    matrices, n_err_mult, pct_mult = analizar_confusion_multiplicidad(
        all_preds, all_targets)
    print(f"\n  Moleculas con al menos un error de multiplicidad: "
          f"{n_err_mult} ({pct_mult:.1f}%)")

    for entorno, (conf, names) in matrices.items():
        w = max(len(n) for n in names) + 1
        total_real = conf.sum(axis=1)
        print(f"\n  Entorno {entorno}:")
        header = f"  {'Real/Pred':<{w}}" + "".join(f"{n:>{w}}" for n in names) + f"  {'Precision':>10}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for i, name in enumerate(names):
            row = f"  {name:<{w}}" + "".join(f"{conf[i][j]:>{w}}" for j in range(3))
            prec = conf[i][i] / total_real[i] * 100 if total_real[i] > 0 else 0.0
            print(row + f"  {prec:>9.1f}%")

    # --- Confusiones cruzadas ---
    print("\n" + "=" * 60)
    print("  MAPA DE CONFUSIONES CRUZADAS")
    print("=" * 60)
    error_matrix = analizar_confusiones_cruzadas(all_preds, all_targets)
    print()
    for i, name in enumerate(GROUP_NAMES):
        fila = error_matrix[i].copy()
        fila[i] = 0
        total = fila.sum()
        if total == 0:
            continue
        top3 = [(GROUP_NAMES[j], fila[j])
                for j in np.argsort(fila)[::-1][:3] if fila[j] > 0]
        pct_g = (all_preds[:, i] != all_targets[:, i]).sum() / n_mols * 100
        print(f"  {name:<10} (falla en {pct_g:.1f}% mol) -> confunde con:")
        for dest, cnt in top3:
            print(f"      {dest:<10}: {cnt:>5} senales  ({cnt/total*100:.1f}%)")

    # --- Resumen ---
    print("\n" + "=" * 60)
    print("  RESUMEN EJECUTIVO")
    print("=" * 60)
    grupos_mal = [(GROUP_NAMES[i], mae[i]) for i in range(N_CLASSES) if mae[i] >= 0.1]
    grupos_mal.sort(key=lambda x: -x[1])
    print(f"\n  EMA: {ema:.2f}%")
    if grupos_mal:
        print("  Grupos con MAE >= 0.1:")
        for g, m in grupos_mal:
            print(f"    {g:<10}: {m:.4f}")
    else:
        print("  TODOS LOS GRUPOS POR DEBAJO DE 0.1 MAE!")
    print(f"  Errores de multiplicidad: {n_err_mult} mol. ({pct_mult:.1f}%)")


if __name__ == "__main__":
    evaluate()