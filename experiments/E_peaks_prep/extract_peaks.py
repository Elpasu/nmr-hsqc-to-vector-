# coding: ascii
"""
extract_peaks.py -- Exp E Fase 1: convierte el HSQC de imagen a un conjunto
de picos (delta_C, delta_H, amp_ch0, amp_ch1) por molecula, via
blob-detection (blob_detect.py) + calibracion pixel->ppm (calibration.py).

Recorre el h5 de imagenes en chunks (config: extraction.chunk_size) para no
cargar las 202465 imagenes completas en RAM (2x256x256 float32 x 202465 ~=
106 GB sin comprimir). Los picos extraidos son mucho mas chicos que las
imagenes, asi que se acumulan enteros en memoria y el padding se calcula
al final, en una sola pasada sobre el h5.

Uso (en el cluster, login node, sin GPU):
    python extract_peaks.py --config config.yaml
"""
import argparse
from pathlib import Path

import numpy as np

from blob_detect import detect_peaks
from calibration import bin_to_ppm


def extract_peaks_from_molecule(hsqc, calib):
    """hsqc: (2, H, W) float array de una sola molecula. calib: dict con
    c13_ppm_min/max, h1_ppm_min/max, resolution. Devuelve lista de
    (delta_c, delta_h, amp_ch0, amp_ch1)."""
    ch0, ch1 = hsqc[0], hsqc[1]
    raw_peaks = detect_peaks(ch0, ch1)
    resolution = calib["resolution"]
    out = []
    for row_c, col_h, amp0, amp1 in raw_peaks:
        delta_c = bin_to_ppm(row_c, calib["c13_ppm_min"], calib["c13_ppm_max"], resolution)
        delta_h = bin_to_ppm(col_h, calib["h1_ppm_min"], calib["h1_ppm_max"], resolution)
        out.append((delta_c, delta_h, amp0, amp1))
    return out


def build_padded_arrays(peaks_per_molecule):
    """peaks_per_molecule: lista de N listas de tuplas de 4 floats.
    Devuelve (peaks_array (N, max_peaks, 4) float32, mask_array (N, max_peaks) bool)."""
    n = len(peaks_per_molecule)
    max_peaks = max((len(p) for p in peaks_per_molecule), default=0)
    peaks_array = np.zeros((n, max_peaks, 4), dtype=np.float32)
    mask_array = np.zeros((n, max_peaks), dtype=bool)
    for i, peaks in enumerate(peaks_per_molecule):
        for j, peak in enumerate(peaks):
            peaks_array[i, j] = peak
            mask_array[i, j] = True
    return peaks_array, mask_array


def main(config_path):
    import h5py
    import yaml

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    base_dir = Path(cfg["paths"]["base_dir"])
    h5_path = base_dir / cfg["paths"]["h5_filename"]
    out_path = base_dir / cfg["paths"]["peaks_output_filename"]
    chunk_size = int(cfg["extraction"]["chunk_size"])
    calib = cfg["hsqc_calibration"]

    print("=" * 60)
    print("  EXP E FASE 1: extraccion de picos")
    print("=" * 60)
    print(f"-> Leyendo: {h5_path}")

    peaks_per_molecule = []
    with h5py.File(h5_path, "r") as f:
        n_total = f["hsqc"].shape[0]
        print(f"-> Moleculas totales: {n_total}")
        for start in range(0, n_total, chunk_size):
            end = min(start + chunk_size, n_total)
            chunk = f["hsqc"][start:end]  # (chunk, 2, 256, 256)
            for i in range(chunk.shape[0]):
                peaks_per_molecule.append(extract_peaks_from_molecule(chunk[i], calib))
            print(f"   procesadas {end}/{n_total}", end="\r")
    print()

    peaks_array, mask_array = build_padded_arrays(peaks_per_molecule)
    n_counts = mask_array.sum(axis=1)
    print(f"-> max_peaks detectado: {peaks_array.shape[1]}")
    print(f"-> picos por molecula: min={n_counts.min()} max={n_counts.max()} "
          f"promedio={n_counts.mean():.2f}")

    with h5py.File(out_path, "w") as f:
        f.create_dataset("peaks", data=peaks_array, compression="lzf")
        f.create_dataset("peaks_mask", data=mask_array, compression="lzf")
        f.attrs["peak_fields"] = "delta_c_ppm,delta_h_ppm,amp_ch0,amp_ch1"
        f.attrs["source_h5"] = str(h5_path)
        for k, v in calib.items():
            f.attrs[f"calib_{k}"] = v

    print(f"\n[SAVE] {out_path}")
    print(">>> EXP E FASE 1 extract_peaks.py OK <<<")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exp E Fase 1: extraccion de picos")
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    main(args.config)
