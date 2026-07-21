# Exp E — Fase 1b: Picos desde el pkl original

Corre LOCAL en tu máquina Windows (no en el cluster) — el pkl de 144k y sus
arrays ya están en `E:\Proyectos\SciTrix\ScitrixDB\DB-Batch0\`.

## Antes de correr

Copiá a esa misma carpeta, si todavía no están:
- Desde `snmgt01:/data/contrib/pci_78/Lucas/DB_58K/`:
  `nmr_calculated_data_scaled_58k.pkl`, `mol_ids_58185.npy`,
  `smiles_58185.npy`.
- Desde donde tengas `smiles_202465.npy` y `vectors_13c_19v_202465.npy`
  (login-1 `DB_200k` u otro lado) — son archivos chicos.

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
