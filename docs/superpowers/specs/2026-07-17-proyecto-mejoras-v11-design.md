# Diseño: Proyecto de mejoras V11 (estructura y convenciones)

> Spec de proceso, no de un experimento puntual. Define cómo se organizan y
> comparan los experimentos D, B, C del `docs/WORKFLOW_V11_para_ClaudeCode.md`.
> El detalle de implementación de cada experimento se define en su propio
> plan (`superpowers:writing-plans`), empezando por D.

## Contexto

El V10 (2 canales + Fórmula Molecular + 19 clases + 202k) es el baseline
actual. El Exp A (diagnóstico, ya corrido fuera de este repo) encontró:

- EMA cruda (modelo solo): 0.61%
- EMA asistida (con oráculo de doble restricción): 74.92%
- La EMA cruda es la métrica primaria para comparar experimentos.
- Causas de bajo rendimiento identificadas: overfitting sin regularización
  (Exp B lo ataca) y "modality collapse" por desbalance de ramas —
  65536 features de la conv vs 128 del 1D vs 8 de la FM (Exp C lo ataca).
- El val set usado para esos números no está deduplicado ni congelado
  (Exp D lo resuelve).

`docs/Runs/RESULTS.md` todavía tiene "TBD" en la fila de V10 — desactualizado
respecto al hallazgo del Exp A. Se corrige como parte del arranque de este
proyecto (ver sección "Housekeeping").

**Orden de trabajo:** D primero (fija el split comparable), después B, después
C. E (representación de picos como set) queda para una fase posterior, fuera
de alcance de este documento.

## Estructura de carpetas

```
experiments/
├── D_val_congelado/
│   ├── RATIONALE.md
│   ├── config.yaml
│   ├── split.py            # genera val_indices_frozen.npy (dedup + val fijo)
│   ├── evaluate.py         # evaluate_v10.py adaptado: lee split congelado
│   ├── model_v10.py        # copia exacta del baseline (no cambia en D)
│   ├── dataset_v10.py      # copia exacta del baseline (no cambia en D)
│   ├── tests/test_forward.py
│   ├── run_eval.sh         # re-evalúa el ckpt V10 YA entrenado sobre el split nuevo
│   └── README.md
├── B_regularizacion/
│   ├── RATIONALE.md
│   ├── config.yaml
│   ├── model_v11b.py       # copia de model_v10.py + dropout (fc_fusion1/2)
│   ├── dataset_v10.py      # copia exacta (no cambia)
│   ├── train.py            # copia de train_v10.py + weight_decay + split congelado
│   ├── evaluate.py
│   ├── dump_predictions.py
│   ├── tests/test_forward.py
│   ├── run_train.sh / run_eval.sh
│   └── README.md
└── C_gap/                  # misma forma que B; model_v11c.py con GAP en vez de flatten
```

Cada carpeta de experimento es **100% autocontenida**: todo archivo que no
cambia respecto al baseline se **copia tal cual**, no se importa vía
`sys.path`. Esto sigue el patrón que ya usa `dump_predictions.py` en `src/`
("Requiere en el mismo dir"), y es deliberado: las carpetas de experimento
son lo que se sincroniza al cluster, y no pueden depender de rutas relativas
a otras partes del repo que quizás no viajen con ellas.

Costo aceptado: duplicación de código entre carpetas. Mitigación: cada
`README.md` deja explícito qué archivos son copia exacta del baseline V10 y
cuáles cambian, para que quede claro dónde mirar si hay que actualizar algo
en los tres a la vez.

## Convención de config

Un solo `config.yaml` por experimento, con el mismo esquema que ya usa
`train_v10.py`: `experiment_name`, `paths.*`, `hyperparameters.*`,
`system.*`. Hereda valores de `config/db.yaml` copiándolos a mano (no hay
loader dual-yaml — ver memoria `exp-config-convention`, esa convención ya se
descartó por un bug real).

Nombres de las 19 clases e índices de CH2 (`[1, 5, 9, 12]`) quedan
hardcodeados en cada script, igual que en `evaluate_v10.py` y
`dump_predictions.py` hoy. Seed=42 hardcodeado (no en config), para
reproducir el split exacto.

## Cadena de dependencia D → B → C

1. **Exp D** corre primero:
   - `split.py` carga `smiles_202465.npy`, canonicaliza con RDKit, detecta
     duplicados internos, y arma `val_indices_frozen.npy` con las 14 428
     moléculas originales de las 144k (Opción B del workflow histórico).
   - El archivo `val_indices_frozen.npy` se guarda en el **cluster, junto a
     los datos** (`DB_200k/val_indices_frozen.npy`), **no se versiona en
     git** — mismo criterio que los checkpoints y los `.h5`, que tampoco
     están en el repo. `config.yaml` de cada experimento lo referencia con
     una ruta relativa a `base_dir`.
   - Además, Exp D **re-evalúa el checkpoint V10 ya entrenado**
     (`nmr_202k_v10_2ch_fm_19v_best.pth`) sobre el split nuevo — solo un
     forward pass sobre val, sin reentrenar, sin gastar cola de GPU nueva.
     Esto da un número **"V10-on-frozen-val"** real, que es la referencia
     válida para comparar B y C (el 0.61%/74.92% original quedan como
     referencia histórica, de un split distinto y no comparable pie a pie).
   - Ese resultado se agrega como fila propia en `docs/Runs/RESULTS.md`.

2. **Exp B** y **Exp C** leen `val_indices_frozen.npy` desde su `config.yaml`
   en vez de usar `random_split`. Cada uno se compara contra
   "V10-on-frozen-val", no contra el V10 original.

3. **Combinar B+C** (val congelado + regularización + GAP en un único
   candidato) queda **fuera de alcance** de este proyecto por ahora. Se
   decide después, con números reales de B y C en mano, como un experimento
   propio (posible `experiments/F_combinado/` o similar) — no se compromete
   en los `RATIONALE.md` de B/C.

## Testing y verificación

Cada experimento tiene su propio `tests/test_forward.py`: forward pass con
datos dummy (sin GPU, sin checkpoint, sin h5 real), corrible en el login
node. Es obligatorio correrlo y confirmar los shapes esperados antes de
proponer cualquier `sbatch` — regla dura del proyecto, ya rota una vez.

## Housekeeping antes de arrancar D

`docs/Runs/RESULTS.md` tiene "TBD" en EMA cruda/asistida de la fila V10.
Antes de empezar el plan de Exp D, se actualiza esa fila con los números
reales del Exp A (0.61% / 74.92%) para que el repo refleje el estado real
conocido.

## Qué NO cubre este documento

- El detalle de implementación de D, B o C (arquitectura exacta, criterio de
  aceptación numérico, contenido del `RATIONALE.md` de cada uno) — eso es
  el `write-plan` de cada experimento, uno a la vez, empezando por D.
- Exp E (representación de picos como set) — fase separada, futura.
- El experimento combinado B+C — decisión futura, post-resultados.

## Próximo paso

`superpowers:writing-plans` para **Exp D** (val congelado), el primero de la
cola. Su `RATIONALE.md` y el plan se presentan para aprobación antes de
escribir código.
