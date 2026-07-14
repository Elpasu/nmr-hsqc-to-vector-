# -*- coding: utf-8 -*-
"""
Evaluacion V9 (19 VARIABLES) - ORACULO ESTRICTO (DOBLE RESTRICCION)
"""
import torch
import numpy as np
import os
from torch.utils.data import DataLoader, random_split
from dataset_v9 import NMRDataset
from model_v9 import NMR_Net

# --- CONFIGURACION ---
DATA_DIR = "/home/lpassaglia.iquir/DB_144K"
H5_FILE = os.path.join(DATA_DIR, "nmr_dataset_144280.h5")
LABEL_FILE = os.path.join(DATA_DIR, "vectors_13c_19v_144280.npy")
SMILES_FILE = os.path.join(DATA_DIR, "smiles_144280.npy")

CHECKPOINT_DIR = "/home/lpassaglia.iquir/DB_144K/checkpoints_V9"
MODEL_NAME = "nmr_144k_v9_19v_best.pth" 
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, MODEL_NAME)

BATCH_SIZE = 256
VAL_SPLIT = 0.1
SEED = 42

GROUP_NAMES = [
    "CH3", "CH2", "CH", "Cq",
    "CH3-O", "CH2-O", "CH-O", "Cq-O",
    "CH3-N", "CH2-N", "CH-N", "Cq-N",
    "=CH2", "=CH/Ar", "Cqsp2",
    "Aldeh", "Imina",
    "C-2X", "C-3X"      
]

ENTORNOS = {
    "Alifaticos (sp3)":               [0, 1, 2, 3],       
    "Heteroatomicos O/N (sp3)":       [4, 5, 6, 7, 8, 9, 10, 11],       
    "Sistemas X-Multiples (C-2X/3X)": [17, 18],
    "Carbonos sp2 (Olef/Arom/C=O)":   [12, 13, 14, 15, 16]           
}

def ajustar_conteo_doble_exacto_v9(pred_cruda, total_real, ch2_real):
    pred_int = np.floor(pred_cruda).astype(int)
    restos = pred_cruda - pred_int
    
    # Índices específicos V9 (1, 5, 9, 12)
    idx_ch2 = [1, 5, 9, 12] 
    idx_resto = [i for i in range(19) if i not in idx_ch2]
    
    ch2_asignados = sum(pred_int[i] for i in idx_ch2)
    ch2_faltantes = int(ch2_real - ch2_asignados)
    
    if ch2_faltantes > 0:
        orden_ch2 = sorted(idx_ch2, key=lambda i: restos[i])
        for i in orden_ch2[-ch2_faltantes:]: pred_int[i] += 1
    elif ch2_faltantes < 0:
        sobran = abs(ch2_faltantes)
        orden_ch2 = sorted(idx_ch2, key=lambda i: restos[i])
        for i in orden_ch2:
            if pred_int[i] > 0:
                pred_int[i] -= 1
                sobran -= 1
                if sobran == 0: break

    resto_real = total_real - ch2_real
    resto_asignados = sum(pred_int[i] for i in idx_resto)
    resto_faltantes = int(resto_real - resto_asignados)
    
    if resto_faltantes > 0:
        orden_resto = sorted(idx_resto, key=lambda i: restos[i])
        for i in orden_resto[-resto_faltantes:]: pred_int[i] += 1
    elif resto_faltantes < 0:
        sobran = abs(resto_faltantes)
        orden_resto = sorted(idx_resto, key=lambda i: restos[i])
        for i in orden_resto:
            if pred_int[i] > 0:
                pred_int[i] -= 1
                sobran -= 1
                if sobran == 0: break

    return pred_int

def analizar_confusiones_cruzadas(all_preds, all_targets):
    n_groups = len(GROUP_NAMES)
    error_matrix = np.zeros((n_groups, n_groups), dtype=int)

    for i in range(len(all_targets)):
        t = all_targets[i]
        p = all_preds[i]
        diff = p - t  
        under = [(g, -diff[g]) for g in range(n_groups) if diff[g] < 0]  
        over  = [(g,  diff[g]) for g in range(n_groups) if diff[g] > 0]  

        for (g_under, cnt_u) in under:
            for (g_over, cnt_o) in over:
                atrib = min(cnt_u, cnt_o)
                error_matrix[g_under][g_over] += atrib
    return error_matrix

def evaluate():
    print("=" * 60)
    print("  EVALUACION V9 (19 CLASES) - DOBLE RESTRICCION DURA")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"-> Dispositivo: {device.type.upper()}")

    if not os.path.exists(CHECKPOINT_PATH):
        print(f"[ERROR] No se encontro el modelo en: {CHECKPOINT_PATH}")
        return

    full_dataset = NMRDataset(H5_FILE, LABEL_FILE, SMILES_FILE)
    val_size = int(len(full_dataset) * VAL_SPLIT)
    train_size = len(full_dataset) - val_size
    
    generator = torch.Generator().manual_seed(SEED)
    _, val_ds = random_split(full_dataset, [train_size, val_size], generator=generator)

    use_pin_memory = torch.cuda.is_available()
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, 
                            num_workers=4, pin_memory=use_pin_memory)

    model = NMR_Net(num_classes=19).to(device)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.eval()

    all_preds_strict = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in val_loader:
            hsqc = inputs[0].to(device)
            proj = inputs[1].to(device)
            cond_tensor = inputs[2].to(device)
            
            outputs = model(hsqc, proj, cond_tensor)
            
            pred_raw = outputs.cpu().numpy()
            targs = targets.cpu().numpy().astype(int)
            cond_batch = cond_tensor.cpu().numpy()
            
            total_sig_batch = cond_batch[:, 0]
            total_ch2_batch = cond_batch[:, 1]
            
            for k in range(len(targs)):
                tot = int(total_sig_batch[k])
                ch2_tot = int(total_ch2_batch[k])
                p_estricta = ajustar_conteo_doble_exacto_v9(pred_raw[k], tot, ch2_tot)
                all_preds_strict.append(p_estricta)
                all_targets.append(targs[k])

    all_preds_strict = np.vstack(all_preds_strict)
    all_targets = np.vstack(all_targets)
    n_mols = len(all_targets)

    print(f"-> Set de validacion: {n_mols} moleculas")

    perfect_mask = np.all(all_preds_strict == all_targets, axis=1)
    ema = np.mean(perfect_mask) * 100
    mae_list = np.mean(np.abs(all_targets - all_preds_strict), axis=0)

    print("\n" + "=" * 60)
    print(f"  EXACT MATCH ACCURACY: {ema:.2f}%")
    print("=" * 60)
    
    print(f"\n{'GRUPO':<10} | {'MAE':>6}  {'Estado'}")
    print("-" * 40)
    for i, name in enumerate(GROUP_NAMES):
        val = mae_list[i]
        marker = "[EXCELENTE]" if val < 0.05 else "[ACEPTABLE]" if val < 0.1 else "[MEJORABLE]"
        print(f"{name:<10} | {val:.4f}  {marker}")

    print("\n" + "=" * 60)
    print("  ANALISIS DE ERRORES POR ENTORNO QUIMICO")
    print("=" * 60)

    for entorno_nombre, indices in ENTORNOS.items():
        errores_entorno = np.any(all_preds_strict[:, indices] != all_targets[:, indices], axis=1)
        n_err = np.sum(errores_entorno)
        pct = n_err / n_mols * 100
        mae_entorno = np.mean(np.abs(all_targets[:, indices] - all_preds_strict[:, indices]))
        print(f"\n  {entorno_nombre}")
        print(f"    Moleculas con error:  {n_err:>6} / {n_mols}  ({pct:.1f}%)")
        print(f"    MAE promedio entorno: {mae_entorno:.4f}")
        for i in indices:
            mae_g = mae_list[i]
            err_g = np.sum(all_preds_strict[:, i] != all_targets[:, i])
            print(f"      {GROUP_NAMES[i]:<10}: MAE={mae_g:.4f}  | Moleculas con error: {err_g} ({err_g/n_mols*100:.1f}%)")

    print("\n" + "=" * 60)
    print("  MAPA DE CONFUSIONES CRUZADAS ENTRE GRUPOS")
    print("=" * 60)
    
    error_matrix = analizar_confusiones_cruzadas(all_preds_strict, all_targets)

    for i, name in enumerate(GROUP_NAMES):
        fila = error_matrix[i].copy()
        fila[i] = 0
        total_errores = np.sum(fila)
        if total_errores == 0:
            continue
        top3_idx = np.argsort(fila)[::-1][:3]
        top3 = [(GROUP_NAMES[j], fila[j]) for j in top3_idx if fila[j] > 0]
        pct_grupo = np.sum(all_preds_strict[:, i] != all_targets[:, i]) / n_mols * 100
        print(f"  {name:<10} (falla en {pct_grupo:.1f}% moleculas) -> confunde con:")
        for dest, cnt in top3:
            pct_conf = cnt / total_errores * 100
            print(f"      {dest:<10}: {cnt:>5} senales  ({pct_conf:.1f}% de sus errores)")

    print("\n" + "=" * 60)
    print("  RESUMEN EJECUTIVO")
    print("=" * 60)
    grupos_mal = [(GROUP_NAMES[i], mae_list[i]) for i in range(19) if mae_list[i] >= 0.1]
    grupos_mal.sort(key=lambda x: -x[1])

    print(f"\n  EMA: {ema:.2f}%")
    if grupos_mal:
        print("  Fricciones remanentes (>=0.1 MAE):")
        for g, m in grupos_mal:
            print(f"    - {g}: {m:.4f}")
    else:
        print("  ¡TODOS LOS GRUPOS ESTÁN POR DEBAJO DE 0.1 MAE!")

if __name__ == "__main__":
    evaluate()