# Exp E — Fase 1b: Picos desde el pkl original (sin binning) — design

> Continuación de Fase 1 (blob-detection). Fase 1 midió 88.75% de moléculas con
> colisión real de picos por límite de resolución de la imagen 256×256
> (`sigma=0.5` ⇒ un blob equivale a ~3.45 ppm en δC y ~0.25 ppm en δH — dos
> carbonos dentro de esa ventana quedan indistinguibles, tanto para
> blob-detection como para la CNN). Esta fase reprocesa los picos directamente
> desde los shifts DFT del pkl original, sin pasar por ningún binning, para
> ver si la colisión cae a niveles marginales.

## Fuentes de datos y dónde corre

Corre **local**, en la máquina Windows de Lucas — el pkl y los arrays de la
parte de 144k ya están ahí (`E:\Proyectos\SciTrix\ScitrixDB\DB-Batch0\`), y
esto permite ejecutar y verificar el script directamente en esta sesión en
vez de ir y venir por copy-paste del cluster.

Archivos necesarios, todos en la misma carpeta local:

- **144k (ya están):** `nmr_calculated_data_scaled_144K.pkl`,
  `mol_ids_144280.npy`, `smiles_144280.npy`.
- **58k (Lucas copia desde `snmgt01:/data/contrib/pci_78/Lucas/DB_58K/`):**
  `nmr_calculated_data_scaled_58k.pkl`, `mol_ids_58185.npy`,
  `smiles_58185.npy`.
- **Verificación (Lucas copia desde donde esté disponible, ej. login-1
  `DB_200k` o `snmgt01`):** `smiles_202465.npy`,
  `vectors_13c_19v_202465.npy` — ambos archivos chicos (solo strings/enteros,
  no imágenes).

## Matching mol_id ↔ smiles_202465

`mol_ids_144280.npy[i]` da el mol_id (clave del pkl) de `smiles_144280.npy[i]`,
para `i` en `[0, 144280)`. Mismo patrón con `mol_ids_58185.npy` /
`smiles_58185.npy` para el resto. Concatenando `smiles_144280.npy` +
`smiles_58185.npy` en ese orden se reconstruye el mismo orden que se usó para
generar `nmr_dataset_v3_202465_fast.h5` (144k originales primero, 58k nuevas
al final — mismo supuesto que ya validó Exp D con leak=0 sobre el val
congelado).

**Verificación de seguridad antes de confiar en el orden:** canonicalizar con
RDKit (reutilizando `canonicalize_smiles`, copiado tal cual de
`experiments/D_val_congelado/split.py`) tanto la concatenación local como
`smiles_202465.npy` real, y comparar posición por posición. Si hay algún
desajuste, el script frena y reporta el índice y los dos SMILES en conflicto
— no se genera ningún pico con una alineación no verificada.

## Extracción de picos (un pico por carbono)

Para cada molécula, usando su SMILES y su `nmr_shifts = pkl[mol_id]`
(`{atom_idx: shift}`, mismo formato que ya usa `Genera_mapas_de_pkl_v2.py`):

1. `mol = Chem.AddHs(Chem.MolFromSmiles(smiles))`.
2. Reconstruir conectividad C-H con `get_carbon_multiplicity` /
   `get_ch_connectivity_with_multiplicity`, copiadas tal cual de
   `Genera_mapas_de_pkl_v2.py` (líneas 128-148) — mismo cálculo de
   multiplicidad, sin modificar.
3. **Cambio respecto a Fase 1**: agrupar los pares C-H por `c_idx` (un pico
   por carbono, no por par C-H). En Fase 1 asumí que los H de un mismo
   carbono caen en el mismo pixel de la imagen — cierto casi siempre, pero no
   para H diastereotópicos con shifts distintos. Con el pkl tenemos el shift
   exacto de cada H, así que:
   - `delta_c` = `nmr_shifts[c_idx]`.
   - `delta_h` = promedio de `nmr_shifts[h_idx]` sobre los H del grupo (los
     que estén presentes en `nmr_shifts`; si ninguno lo está, se descarta el
     carbono — sin datos no hay pico).
   - `mult` = multiplicidad del grupo (constante dentro del grupo, viene de
     `get_carbon_multiplicity`).
   - `amp_ch0` = `(-1.0 if mult == 2 else +1.0) * mult` — misma convención
     DEPT del generador de imagen (CH2 negativo), sin normalizar (acá no hay
     imagen que normalizar).
   - `amp_ch1` = `mult / 3.0` — misma convención de "tipo CH" del canal 1.
   - Si `c_idx` no está en `nmr_shifts`, se descarta el carbono entero (sin
     δC no hay pico posible).

Esto hace que el conteo de picos sea directamente comparable con
`visible_label_count` (que cuenta carbonos, no protones) — la misma
comparación que ya usa `validate_peaks.py`.

## Salida

`.npz` (no `.h5` — evita depender de `h5py`, que no está instalado
localmente, y el volumen de datos es chico) con:
- `peaks (N, max_peaks, 4)` float32 — mismos 4 campos que Fase 1
  (`delta_c_ppm, delta_h_ppm, amp_ch0, amp_ch1`), pero valores continuos
  reales, no derivados de pixel.
- `peaks_mask (N, max_peaks)` bool.

Reutiliza `build_padded_arrays` de `experiments/E_peaks_prep/extract_peaks.py`
sin cambios — mismo contrato ya probado en Fase 1.

## Validación

Reutiliza sin modificar las funciones puras de
`experiments/E_peaks_prep/validate_peaks.py` (`visible_label_counts`,
`blob_counts_from_mask`, `validation_report`), corridas contra
`vectors_13c_19v_202465.npy` y el `peaks_mask` nuevo. Mismo criterio de
`INVISIBLE_CLASSES`. El reporte (match exacto %, % con colisión, déficit
promedio) es directamente comparable al de Fase 1.

## Testing

A diferencia de Fase 1, esta fase corre en la máquina local donde numpy y
RDKit **sí** están disponibles — los tests unitarios se pueden ejecutar y
verificar de verdad en esta sesión, no solo por revisión de código. Test de
agrupación por carbono con una molécula sintética simple armada a mano (ej.
etanol, `CH3-CH2-OH`, con shifts inventados pero conocidos) para verificar
que el agrupamiento por `c_idx` y el promedio de `delta_h` dan el resultado
esperado, incluyendo un caso con dos H de shifts distintos en el mismo
carbono (diastereotópicos) para confirmar que se promedian y no se cuentan
como dos picos.

## Fuera de alcance

El modelo de conjuntos (Fase 2, DeepSets) sigue siendo un spec separado,
decidido después de ver el resultado de esta validación — si la colisión cae
a niveles marginales, se arma la Fase 2 sobre esta representación; si sigue
alta, hay que reconsiderar el enfoque de raíz.
