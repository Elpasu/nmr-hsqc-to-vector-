# RATIONALE — Exp B: Regularización

## Hipótesis

El V10 tiene overfitting medido y documentado (train 0.013 vs val 0.031 en la corrida
real; val estancado desde ~ep48 mientras train sigue bajando — ver `docs/Runs/RESULTS.md`).
Los modelos V4-V6 tenían dropout y weight_decay; se perdieron entre V6 y V9 y nunca se
repusieron. Reponerlos debería achicar esa brecha train/val y, potencialmente, mejorar la
generalización (EMA en val).

## Qué causa del diagnóstico ataca

Causa #2 del diagnóstico del Exp A: overfitting sin regularización. No ataca el modality
collapse (desbalance de ramas conv/1D/FM) — eso lo ataca Exp C.

## Qué cambia exactamente respecto al V10

- `model_v11b.py`: copia de `model_v10.py` + `nn.Dropout(p=dropout)` después de la ReLU
  de `fc_fusion1` y de `fc_fusion2`. `dropout` es parámetro del constructor de `NMR_Net`,
  no un valor fijo — viene del config.
- `train.py`: copia de `train_v10.py` + `weight_decay=cfg['regularization']['weight_decay']`
  en el `optim.Adam`. `dropout=0.25` y `weight_decay=0.00001` (ya están en
  `config/db.yaml`, sección `regularization`, y se copian al `config.yaml` propio de este
  experimento).
- **Split de entrenamiento:** en vez de `random_split(seed=42)` (lo que hace
  `train_v10.py`), usa el val congelado que generó Exp D
  (`DB_200k/val_indices_frozen.npy`). El train set se reconstruye al arrancar el
  entrenamiento con la misma lógica de deduplicación/leak que usó `split.py` de Exp D
  (`canonicalize_smiles` + `remove_leaking_from_train`, copiadas a `split_utils.py` de
  este experimento — self-contained, no se reimporta Exp D). Esto no requiere que vuelvas
  a correr nada de Exp D; es un cálculo de menos de un minuto al arrancar el training, no
  por época.
- Todo lo demás queda igual al V10: mismo `ConstrainedMSELoss` (lambda_sum=0.5), mismo
  scheduler (`ReduceLROnPlateau`, patience=8, factor=0.7), 100 épocas, `num_workers=0`,
  mismo batch_size (64) y learning_rate (0.001).

## Qué métrica esperás mover y cuánto

El gap train/val (loss) debería achicarse respecto al de V10 (train 0.013 / val 0.031).
La EMA cruda/asistida sobre el val congelado no debería empeorar respecto a la referencia
"V10-on-frozen-val" (0.93% cruda / 90.66% asistida, `docs/Runs/RESULTS.md`); idealmente
mejora. Ojo: esa referencia está inflada por contaminación train/val (documentado en
`docs/Runs/RESULTS.md` y en el `RATIONALE.md` de Exp D) — Exp B, en cambio, SÍ excluye el
val congelado de su entrenamiento, así que su número es limpio. La comparación más honesta
no es solo "¿la EMA subió?", sino también "¿el gap train/val bajó?": eso es lo que prueba
que la regularización está haciendo su trabajo, independientemente de cuánto se mueva la
EMA en este val específico.

## Criterio de éxito/fracaso

- **Éxito:** el gap train/val final es menor que el de V10 (0.031-0.013=0.018), y la EMA
  cruda sobre el val congelado no cae por debajo de un margen razonable respecto a la
  referencia. Shapes del smoke test idénticos a V10 (dropout no cambia dimensiones).
- **Fracaso:** dropout/weight_decay tan agresivos que el modelo underfittea (val loss no
  baja, o EMA cruda cae fuerte) — señal de que 0.25/1e-5 son demasiado agresivos para este
  dataset y hace falta ajustar.
