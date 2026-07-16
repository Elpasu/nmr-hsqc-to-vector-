# coding: ascii
"""
dump_predictions.py  -- corre en el CLUSTER (NMR_env, GPU o CPU)

Genera un archivo con las predicciones por molecula del set de validacion,
para alimentar la GUI de inspeccion (gui_inspector.py, corre en tu PC).

NO reentrena. Es solo un forward pass sobre el val set (~20k mols, minutos).

Salida: predictions_<experiment>.parquet  con columnas:
  idx, smiles, y_true (19 ints), y_pred_crude (19 ints), y_pred_assisted (19 ints)

Requiere en el mismo dir (o en el path): model_v10.py, dataset_v10.py
Uso:
  python dump_predictions.py
Ajusta las rutas de CONFIG abajo si hace falta.
"""
import os, yaml, numpy as np, torch
from torch.utils.data import DataLoader, random_split

# ---------- CONFIG (ajustar si tus rutas difieren) ----------
DB_YAML     = "config/db.yaml"          # fuente de verdad; si no existe, usa los defaults de abajo
BASE_DIR    = "/home/lpassaglia.iquir/DB_200k"
H5_FILE     = os.path.join(BASE_DIR, "nmr_dataset_v3_202465_fast.h5")   # el h5 RAPIDO
LABELS_FILE = os.path.join(BASE_DIR, "vectors_13c_19v_202465.npy")
SMILES_FILE = os.path.join(BASE_DIR, "smiles_202465.npy")
CKPT        = os.path.join(BASE_DIR, "checkpoints_V10_202k",
                           "nmr_202k_v10_2ch_fm_19v_best.pth")
EXPERIMENT  = "v10"
VAL_SPLIT   = 0.1
SEED        = 42
BATCH_SIZE  = 256
IDX_CH2     = [1, 5, 9, 12]          # CH2, CH2-O, CH2-N, =CH2
N_CLASSES   = 19
OUT_FILE    = f"predictions_{EXPERIMENT}.parquet"
# -----------------------------------------------------------

from model_v10 import NMR_Net
from dataset_v10 import NMRDataset


def oraculo_doble(pred_raw, total_real, ch2_real):
    """Ajuste de doble restriccion (modo asistido), identico al evaluate."""
    pred = np.floor(pred_raw).astype(int)
    rest = pred_raw - pred
    idx_rest = [i for i in range(N_CLASSES) if i not in IDX_CH2]

    # CH2
    falt = int(ch2_real - sum(pred[i] for i in IDX_CH2))
    if falt > 0:
        for i in sorted(IDX_CH2, key=lambda i: rest[i])[-falt:]:
            pred[i] += 1
    elif falt < 0:
        s = -falt
        for i in sorted(IDX_CH2, key=lambda i: rest[i]):
            if pred[i] > 0:
                pred[i] -= 1; s -= 1
                if s == 0: break
    # resto
    falt = int((total_real - ch2_real) - sum(pred[i] for i in idx_rest))
    if falt > 0:
        for i in sorted(idx_rest, key=lambda i: rest[i])[-falt:]:
            pred[i] += 1
    elif falt < 0:
        s = -falt
        for i in sorted(idx_rest, key=lambda i: rest[i]):
            if pred[i] > 0:
                pred[i] -= 1; s -= 1
                if s == 0: break
    return pred


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Dispositivo: {device}")
    print(f"[INFO] Cargando dataset...")
    ds = NMRDataset(H5_FILE, LABELS_FILE, SMILES_FILE)
    smiles_all = np.load(SMILES_FILE, allow_pickle=True)

    # mismo split que el training/eval
    val_size = int(len(ds) * VAL_SPLIT)
    train_size = len(ds) - val_size
    gen = torch.Generator().manual_seed(SEED)
    _, val_ds = random_split(ds, [train_size, val_size], generator=gen)
    val_indices = val_ds.indices                      # indices originales en el dataset
    loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=0, pin_memory=torch.cuda.is_available())

    print(f"[INFO] Val set: {len(val_ds)} moleculas")
    model = NMR_Net(num_classes=N_CLASSES).to(device)
    model.load_state_dict(torch.load(CKPT, map_location=device))
    model.eval()

    rows = []
    ptr = 0
    with torch.no_grad():
        for inputs, targets in loader:
            hsqc = inputs[0].to(device)
            proj = inputs[1].to(device)
            cond = inputs[2].to(device)
            out = model(hsqc, proj, cond).cpu().numpy()
            t = targets.cpu().numpy().astype(int)
            c = cond.cpu().numpy()
            for k in range(len(t)):
                orig_idx = val_indices[ptr]; ptr += 1
                y_true = t[k]
                total = int(c[k, 0]); ch2 = int(c[k, 1])
                y_crude = np.clip(np.floor(out[k]), 0, None).astype(int)
                y_assist = oraculo_doble(out[k], total, ch2)
                rows.append({
                    "idx": int(orig_idx),
                    "smiles": str(smiles_all[orig_idx]),
                    "y_true": y_true.tolist(),
                    "y_pred_crude": y_crude.tolist(),
                    "y_pred_assisted": y_assist.tolist(),
                })

    try:
        import pandas as pd
        df = pd.DataFrame(rows)
        df.to_parquet(OUT_FILE)
        print(f"[OK] Guardado {len(df)} filas -> {OUT_FILE}")
    except Exception as e:
        # fallback a JSON si no hay pyarrow
        import json
        alt = OUT_FILE.replace(".parquet", ".json")
        with open(alt, "w") as f:
            json.dump(rows, f)
        print(f"[OK] (fallback JSON) Guardado {len(rows)} filas -> {alt}")
        print(f"     (instala pyarrow para parquet: pip install pyarrow)")


if __name__ == "__main__":
    main()
