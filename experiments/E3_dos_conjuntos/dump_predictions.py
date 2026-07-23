# coding: ascii
"""
dump_predictions.py -- Exp E Fase 3: vuelca las predicciones del checkpoint
de este experimento sobre el val congelado (Exp D), para la GUI
(src/gui/gui_inspector.py, corre en tu PC). El modelo se instancia segun
model.arch del config (deepsets | settransformer).

NO reentrena. Solo forward pass sobre val_indices_frozen.npy.

Salida: predictions_<experiment_name>.parquet con columnas:
  idx, smiles, y_true (19 ints), y_pred_crude (19 ints), y_pred_assisted (19 ints),
  crosspeaks (lista de [delta_c, delta_h] en ppm), c13_shifts (lista de delta_c en ppm).

Los desplazamientos se leen del .npz CRUDO (en ppm, sin normalizar) para que la
GUI pueda dibujar el HSQC real y diagnosticar confusiones por shift (ej: CH2 vs CH2-N).

Uso:
  python dump_predictions.py --config config_deepsets.yaml
"""
import os
import argparse
import yaml
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader, Subset

N_CLASSES = 19
IDX_CH2 = [1, 5, 9, 12]   # CH2, CH2-O, CH2-N, =CH2


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def oraculo_doble(pred_raw, total_real, ch2_real):
    """Ajuste de doble restriccion (modo asistido), identico al evaluate."""
    pred = np.floor(pred_raw).astype(int)
    rest = pred_raw - pred
    idx_rest = [i for i in range(N_CLASSES) if i not in IDX_CH2]

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


def main(config_path):
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
    out_file = f"predictions_{cfg['experiment_name']}.parquet"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Dispositivo: {device}")

    if not os.path.exists(ckpt_path):
        print(f"[ERROR] No se encontro el checkpoint en:\n  {ckpt_path}")
        return
    if not os.path.exists(val_indices_path):
        print(f"[ERROR] No se encontro el split congelado en:\n  {val_indices_path}")
        print("        Corri primero experiments/D_val_congelado/split.py (Exp D).")
        return

    print("[INFO] Cargando dataset...")
    ds = NMRTwoSetsDataset(str(peaks_ch), str(peaks_13c), str(labels_path),
                           str(smiles_path), cfg["normalization"])
    smiles_all = np.load(smiles_path, allow_pickle=True)

    # --- Picos CRUDOS en ppm (sin normalizar) para la GUI (HSQC + delta 13C) ---
    npz_ch = np.load(peaks_ch)
    raw_peaks_ch = npz_ch["peaks"]         # (N, 32, 4): [dC, dH, amp0, amp1] en ppm
    raw_mask_ch = npz_ch["peaks_mask"]     # (N, 32)
    npz_13c = np.load(peaks_13c)
    raw_peaks_13c = npz_13c["peaks_13c"]   # (N, M, 1): [dC] en ppm
    raw_mask_13c = npz_13c["mask_13c"]     # (N, M)

    val_indices = np.load(val_indices_path)
    val_ds = Subset(ds, val_indices.tolist())
    loader = DataLoader(val_ds, batch_size=256, shuffle=False,
                        num_workers=0, pin_memory=(device.type == "cuda"))

    print(f"[INFO] Val set (congelado): {len(val_ds)} moleculas")
    model = build_model(cfg, num_classes=N_CLASSES).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    rows = []
    ptr = 0
    with torch.no_grad():
        for inputs, targets in loader:
            pch = inputs[0].to(device); mch = inputs[1].to(device)
            p13 = inputs[2].to(device); m13 = inputs[3].to(device)
            cond = inputs[4].to(device)
            out = model(pch, mch, p13, m13, cond).cpu().numpy()
            t = targets.cpu().numpy().astype(int)
            c = cond.cpu().numpy()
            for k in range(len(t)):
                orig_idx = int(val_indices[ptr]); ptr += 1
                total = int(c[k, 0]); ch2 = int(c[k, 1])

                m_ch = raw_mask_ch[orig_idx].astype(bool)
                cps = raw_peaks_ch[orig_idx][m_ch][:, :2]   # [dC, dH] en ppm
                crosspeaks = [[round(float(dc), 2), round(float(dh), 3)]
                              for dc, dh in cps]
                m_13 = raw_mask_13c[orig_idx].astype(bool)
                c13 = raw_peaks_13c[orig_idx][m_13][:, 0]   # dC en ppm
                c13_shifts = [round(float(x), 2) for x in c13]

                rows.append({
                    "idx": orig_idx,
                    "smiles": str(smiles_all[orig_idx]),
                    "y_true": t[k].tolist(),
                    "y_pred_crude": np.clip(np.floor(out[k]), 0, None).astype(int).tolist(),
                    "y_pred_assisted": oraculo_doble(out[k], total, ch2).tolist(),
                    "crosspeaks": crosspeaks,
                    "c13_shifts": c13_shifts,
                })

    try:
        import pandas as pd
        df = pd.DataFrame(rows)
        df.to_parquet(out_file)
        print(f"[OK] Guardado {len(df)} filas -> {out_file}")
    except Exception:
        import json
        alt = out_file.replace(".parquet", ".json")
        with open(alt, "w") as f:
            json.dump(rows, f)
        print(f"[OK] (fallback JSON) Guardado {len(rows)} filas -> {alt}")
        print("     (instala pyarrow para parquet: pip install pyarrow)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exp E Fase 3: dump de predicciones para la GUI")
    parser.add_argument("--config", type=str, default="config_deepsets.yaml")
    args = parser.parse_args()
    main(args.config)
