# Exp E — Fase 1b: Picos desde el pkl original

Corre LOCAL en tu máquina Windows (no en el cluster). Cada fuente vive en su
propia carpeta, configurada en `config_pkl.yaml`:

- `base_dir_144k`: `E:\Proyectos\SciTrix\ScitrixDB\DB_nmr_to_vector\144K\`
  (pkl, `mol_ids_144280.npy`, `smiles_144280.npy`).
- `base_dir_58k`: `E:\Proyectos\SciTrix\ScitrixDB\DB_nmr_to_vector\58K\`
  (pkl, `mol_ids_58185.npy`, `smiles_58185.npy`).
- `base_dir_202k`: `E:\Proyectos\SciTrix\ScitrixDB\DB_nmr_to_vector\202K_suma\`
  (`smiles_202465.npy`, `vectors_13c_19v_202465.npy` — el dataset real
  contra el que se verifica alineación; ahí también se guarda la salida
  `peaks_pkl_202465.npz`).

## Antes de correr

Confirmá que `202K_suma` tiene `smiles_202465.npy` y
`vectors_13c_19v_202465.npy` (si todavía están bajando, esperá a que
termine antes de correr `extract_peaks_pkl.py`).

## Orden de comandos

1. Smoke test obligatorio (no toca ningún dato real, usa archivos
   sintéticos en un directorio temporal):
   ```bash
   cd experiments/E_peaks_prep
   python tests/test_ch_connectivity.py
   python tests/test_extract_peaks_pkl.py
   python tests/test_verify_alignment.py
   python tests/test_smoke_pkl.py
   ```
   Todos deben terminar en `>>> ... OK <<<`.
2. Correr sobre los datos reales (202465 moléculas):
   ```bash
   python extract_peaks_pkl.py --config config_pkl.yaml
   ```
   Si falla en "[ERROR] desajuste de alineacion en indice N" — pará ahí,
   avisá con el índice y los dos SMILES que reportó, no sigas.
3. Si termina OK, va a imprimir el mismo tipo de reporte que Fase 1 (match
   exacto %, % con colisión, déficit promedio) — comparalo contra el
   88.75% de colisión de Fase 1 y avisá los números.
