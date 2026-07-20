# coding: ascii
"""
scripts/audit_data_pipeline.py -- diagnostico de la representacion HSQC:
cuanta "masa" del conteo total es estructuralmente invisible en HSQC,
que tan dispersa es la imagen, si el conteo visible correlaciona con la
imagen real, y si hay colisiones de binning en 256x256.

Requiere h5py (para leer las imagenes) ademas de numpy + pyyaml. Corre en
el login node, sin GPU (h5py no necesita CUDA). Lee solo una muestra
aleatoria de las imagenes (no las 202k) para no cargar el h5 entero.

Uso:
    python scripts/audit_data_pipeline.py --config config/db.yaml --n-sample 2000
"""
import argparse
from pathlib import Path

import numpy as np
import yaml

# Clases sin H propio (invisibles en HSQC por definicion quimica: HSQC solo
# muestra correlaciones C-H de un enlace). Marcadas por el nombre de clase
# ("Cq" = carbono cuaternario). Esta lista es una suposicion basada en la
# convencion de nombres del proyecto (config/db.yaml), no verificada
# molecula por molecula con RDKit -- si esta mal, corregir aca.
INVISIBLE_CLASSES = ["Cq", "Cq-O", "Cq-N", "Cqsp2"]


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def invisible_mass_report(labels, class_names):
    idx_invisible = [class_names.index(c) for c in INVISIBLE_CLASSES]
    idx_visible = [i for i in range(len(class_names)) if i not in idx_invisible]

    total_per_mol = labels.sum(axis=1).astype(float)
    invisible_per_mol = labels[:, idx_invisible].sum(axis=1).astype(float)
    visible_per_mol = labels[:, idx_visible].sum(axis=1).astype(float)

    mean_total = total_per_mol.mean()
    mean_invisible = invisible_per_mol.mean()
    frac_invisible = mean_invisible / mean_total * 100.0

    print(f"\n{'=' * 70}")
    print("  MASA INVISIBLE EN HSQC (clases sin H propio)")
    print(f"{'=' * 70}")
    print(f"  Clases tratadas como invisibles: {', '.join(INVISIBLE_CLASSES)}")
    print(f"  Promedio de conteo total por molecula:      {mean_total:.3f}")
    print(f"  Promedio de conteo invisible por molecula:  {mean_invisible:.3f}")
    print(f"  -> Fraccion invisible del conteo total:     {frac_invisible:.1f}%")
    print(f"  Promedio de conteo VISIBLE por molecula:    {visible_per_mol.mean():.3f}")
    return idx_visible, visible_per_mol


def sparsity_report(hsqc_batch):
    # hsqc_batch: (n, 2, 256, 256)
    n = hsqc_batch.shape[0]
    nonzero_ch0 = (hsqc_batch[:, 0] != 0).sum(axis=(1, 2))
    nonzero_ch1 = (hsqc_batch[:, 1] != 0).sum(axis=(1, 2))
    total_px = 256 * 256

    print(f"\n{'=' * 70}")
    print(f"  DISPERSION DE LA IMAGEN (muestra de {n} moleculas)")
    print(f"{'=' * 70}")
    print(f"  Canal 0 (DEPT): {nonzero_ch0.mean():.1f} px no-cero en promedio "
          f"({nonzero_ch0.mean() / total_px * 100:.3f}% de {total_px})")
    print(f"  Canal 1 (tipo CH): {nonzero_ch1.mean():.1f} px no-cero en promedio "
          f"({nonzero_ch1.mean() / total_px * 100:.3f}% de {total_px})")
    print(f"  -> El resto de la imagen (~{100 - nonzero_ch0.mean() / total_px * 100:.1f}%) "
          f"es espacio vacio que la CNN igual convoluciona.")
    return nonzero_ch0


def consistency_report(nonzero_ch0, visible_per_mol_sample):
    corr = np.corrcoef(nonzero_ch0, visible_per_mol_sample)[0, 1]
    print(f"\n{'=' * 70}")
    print("  CONSISTENCIA ETIQUETA <-> IMAGEN")
    print(f"{'=' * 70}")
    print(f"  Correlacion (pixeles no-cero canal 0) vs (conteo visible en label): {corr:.3f}")
    if corr < 0.7:
        print("  -> Correlacion mas baja de lo esperado: revisar si hay un bug de")
        print("     codificacion (label y imagen no se corresponden 1 a 1).")
    else:
        print("  -> Correlacion alta: la imagen y el label parecen consistentes.")


def collision_report(hsqc_batch, visible_per_mol_sample):
    # Cuenta "picos" como pixeles no-cero del canal 0 (no distingue picos
    # solapados en el mismo pixel -- eso es justamente lo que se quiere medir).
    nonzero_ch0 = (hsqc_batch[:, 0] != 0).sum(axis=(1, 2)).astype(float)
    deficit = visible_per_mol_sample - nonzero_ch0
    n_collision = int((deficit > 0).sum())
    n = len(visible_per_mol_sample)

    print(f"\n{'=' * 70}")
    print("  COLISIONES DE BINNING (label visible > picos no-cero en imagen)")
    print(f"{'=' * 70}")
    print(f"  Moleculas con posible colision: {n_collision} / {n} ({n_collision / n * 100:.1f}%)")
    if n_collision > 0:
        print(f"  Deficit promedio (label - pixeles) en esas moleculas: "
              f"{deficit[deficit > 0].mean():.2f}")
    else:
        print("  Deficit promedio: (ninguna colision detectada)")
    print("  -> Un deficit positivo sugiere que 2+ senales cercanas cayeron en el")
    print("     mismo pixel y se fusionaron en una sola marca (perdida real de")
    print("     informacion, corregible con mas resolucion o mejor codificacion).")


def main(config_path, n_sample):
    import h5py

    cfg = load_config(config_path)
    base_dir = Path(cfg["data"]["base_dir"])
    labels_path = base_dir / cfg["data"]["labels_19v"]
    h5_path = base_dir / cfg["data"]["h5_v3"]
    class_names = cfg["classes_19v"]

    print(f"-> Cargando labels desde: {labels_path}")
    labels = np.load(labels_path).astype(int)
    n_total = labels.shape[0]
    print(f"-> Moleculas totales: {n_total}")

    idx_visible, visible_per_mol = invisible_mass_report(labels, class_names)

    n_sample = min(n_sample, n_total)
    rng = np.random.default_rng(42)
    sample_idx = np.sort(rng.choice(n_total, size=n_sample, replace=False))

    print(f"\n-> Leyendo {n_sample} imagenes HSQC (muestra aleatoria, seed=42) desde: {h5_path}")
    with h5py.File(h5_path, "r") as f:
        hsqc_sample = f["hsqc"][sample_idx]  # (n_sample, 2, 256, 256)

    visible_sample = visible_per_mol[sample_idx]

    nonzero_ch0 = sparsity_report(hsqc_sample)
    consistency_report(nonzero_ch0, visible_sample)
    collision_report(hsqc_sample, visible_sample)

    print("\n>>> AUDIT PIPELINE OK <<<")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auditoria del pipeline de datos HSQC")
    parser.add_argument("--config", type=str, default="config/db.yaml")
    parser.add_argument("--n-sample", type=int, default=2000,
                        help="Cantidad de moleculas a muestrear para las imagenes (default 2000)")
    args = parser.parse_args()
    main(args.config, args.n_sample)
