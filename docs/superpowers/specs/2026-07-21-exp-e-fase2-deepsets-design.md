# Exp E — Fase 2: modelo DeepSets sobre picos (design)

> Fase 2 de Exp E. Fase 1b (`docs/Runs/RESULTS.md`, sección "Exp E — Fase 1b")
> confirmó que la representación de picos extraída del pkl original preserva
> 97.19% del conteo visible del label (2.19% de colisión real, marginal) —
> mucho mejor que el 88.75% de colisión de la imagen 256×256 (Fase 1). Esta
> fase arma y deja lista para entrenar la primera arquitectura sobre esa
> representación: un modelo DeepSets.

## Motivación y alcance

Tres arquitecturas distintas sobre la imagen (V10, Exp B, Exp C) mostraron las
mismas confusiones de clase persistentes (`Cqsp2`↔`=CH/Ar`, `CH2`↔`CH2-N`) —
evidencia de que el problema es de representación, no arquitectónico. Fase 1b
ya validó que los picos del pkl casi no pierden información. Esta fase prueba
si un modelo que consume esos picos directamente (sin pasar por imagen)
mejora la EMA. Alcance: dejar la carpeta lista para `sbatch` — entrenar y
evaluar los corre Lucas en el cluster, como en Exp B/C.

## Arquitectura

DeepSets, primer candidato del workflow original (más simple que Set
Transformer; se escala a Set Transformer solo si esto muestra que el enfoque
de conjuntos funciona). Por cada pico `(δC, δH, amp_ch0, amp_ch1)` — hasta 32
por molécula, con máscara de válidos — un MLP compartido `4 → 64 → 64`
(mismos pesos para todos los picos de todas las moléculas, así la salida es
invariante a permutación del orden de los picos). Los vectores resultantes
se agregan por **promedio enmascarado** (solo sobre los picos válidos de
`peaks_mask`; si una molécula no tiene picos válidos, el agregado es cero en
vez de dividir por cero) — se eligió promedio en vez de suma porque el
número de picos varía mucho entre moléculas (0 a 32, según Fase 1b) y la
suma haría que el agregado dependiera del tamaño de la molécula, no solo de
su composición.

El vector agregado (64) se concatena con `cond_tensor` (8: total_señales,
total_CH2, C,H,N,O,S,Hal — igual convención que V10/B/C, se sigue calculando
igual: total_señales y total_CH2 salen del label, la fórmula molecular sale
de RDKit sobre el SMILES) → MLP de fusión `72 → 128 → 64 → 19`.

**Se elimina la imagen HSQC y las proyecciones 1D (`vec_c`/`vec_h`)** — los
picos ya contienen `δC`/`δH` reales sin binning, así que `vec_c`/`vec_h`
(derivadas del mismo binning de 256 bins que falló en Fase 1) quedarían
redundantes.

**Tamaño del modelo, decisión deliberada:** se mantiene chico (capacidad del
orden de Exp C, no de V10) a propósito — V10 (8.6M parámetros) sobreajustó y
dio peor resultado que Exp C (223k parámetros, ~38x menos). No se aprovecha
el presupuesto extra de GPU (esto entrena mucho más rápido que la CNN, sin
convoluciones sobre 256×256) para agrandar el modelo en esta misma corrida,
para no mezclar dos variables (representación de datos vs capacidad) en un
solo experimento y perder comparabilidad con V10/B/C. Si el resultado es
bueno, el tiempo de GPU sobrante se usa en un experimento aparte (Set
Transformer o una variante más grande de DeepSets).

## Datos

`experiments/E2_deepsets/dataset_e2.py`, `NMRPeaksDataset`: carga
`peaks_pkl_202465.npz` (`peaks`, `peaks_mask`) completo en memoria al
iniciar — es chico (~100MB), no hay h5py de por medio para los picos.
`cond_tensor` se calcula igual que `dataset_v10.py` (RDKit sobre
`smiles_202465.npy` + label). Reutiliza el split congelado de Exp D
(`val_indices_frozen.npy` + `split_utils.py` copiado, mismo patrón que Exp
B/C) — mismo val que B y C, comparación directa.

`num_workers: 0` se mantiene como estándar del proyecto aunque técnicamente
esta vez no aplique el motivo original (deadlock de h5py) — no hay h5py en
el dataset de picos, pero se mantiene por consistencia y porque el dataset
ya es rápido sin paralelizar (todo en memoria, sin I/O por ítem).

## Dónde corre y qué falta

Entrenamiento en login-1 (GPU), como el resto de los experimentos. `peaks_pkl_202465.npz`
está solo en la máquina local de Lucas (`DB_nmr_to_vector/202K_suma/`) — hay
que copiarlo a `DB_200k/` en el cluster antes de correr. Se documenta en el
README de la carpeta.

## Hiperparámetros

Mismos que el resto de los experimentos, para que la comparación sea
limpia: Adam lr=0.001, `ReduceLROnPlateau` patience=8/factor=0.7, batch=64,
epochs=100, seed=42. Sin regularización explícita (dropout/weight_decay),
misma decisión que Exp C tras la falla de Exp B.

## Testing

Smoke test offline (rule 5 de CLAUDE.md): forward pass con un batch
sintético (`peaks`, `peaks_mask`, `cond`), sin datos reales, verificando
shape de salida `(B, 19)` y conteo de parámetros razonable (mismo patrón que
`test_forward.py` de Exp C) — corre localmente antes de cualquier `sbatch`.

## Fuera de alcance

Set Transformer y cualquier variante de capacidad más grande quedan para un
experimento aparte, después de ver el resultado de esta corrida.
