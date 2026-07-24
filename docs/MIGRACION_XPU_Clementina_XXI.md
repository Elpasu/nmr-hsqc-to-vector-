# Migración a Intel XPU — Clementina XXI

> **Documento de referencia permanente del proyecto.**
> Describe por qué, cómo y con qué criterios se migra el proyecto NMR HSQC→Vector
> desde GPUs NVIDIA (CUDA, clúster login-1) a GPUs Intel (XPU / Ponte Vecchio,
> clúster Clementina XXI). Pensado para que cualquier integrante del grupo entienda,
> dentro de uno o dos años, las decisiones tomadas y pueda reproducir el entorno.
>
> **Autor de la migración:** Lucas Passaglia (UCA Team).
> **Fecha de inicio:** 2026-07-23. **Fecha de cierre (Fases 0-4):** 2026-07-24.
> **Estado global:** migración de E3 a XPU **validada** (Fases 0-4 completas). Fase 5
> (optimización) es opcional y queda pendiente. Extender la migración a otros experimentos
> (V10, B, C, E2, F, D) es una decisión aparte, fuera de alcance de este documento — ver §5.
> Detalle en [§11 Estado actual (checklist)](#11-estado-actual-checklist).

---

## Índice

1. [Motivación](#1-motivación)
2. [Resumen del análisis de viabilidad](#2-resumen-del-análisis-de-viabilidad)
3. [Hardware de Clementina XXI](#3-hardware-de-clementina-xxi)
4. [Diferencias respecto al clúster anterior](#4-diferencias-respecto-al-clúster-anterior)
5. [Objetivos de la migración](#5-objetivos-de-la-migración)
6. [Estrategia elegida](#6-estrategia-elegida)
7. [Fases del trabajo](#7-fases-del-trabajo)
8. [Riesgos identificados](#8-riesgos-identificados)
9. [Decisiones técnicas](#9-decisiones-técnicas)
10. [Cómo reproducir el entorno](#10-cómo-reproducir-el-entorno)
11. [Estado actual (checklist)](#11-estado-actual-checklist)
12. [Mapa de cambios previstos en el código (solo E3)](#12-mapa-de-cambios-previstos-en-el-código-solo-e3)
13. [Organización de datos y rutas](#13-organización-de-datos-y-rutas)

---

## 1. Motivación

El proyecto se desarrolló y entrenó históricamente sobre GPUs **NVIDIA A10** en el
clúster **login-1** (IQUIR), usando el backend **CUDA** de PyTorch. Se abre ahora el
acceso a un nuevo clúster HPC, **Clementina XXI**, cuyos nodos montan **GPUs Intel Data
Center GPU Max 1550 (arquitectura Ponte Vecchio)**. Migrar permite:

- Ejecutar entrenamientos en infraestructura de mayor capacidad (HBM abundante, más
  dispositivos por nodo, XMX para precisión mixta).
- No depender de la disponibilidad y la cola del clúster anterior.
- Habilitar, más adelante, escalado multi-GPU y precisión BF16 que la A10 no ofrecía en
  las mismas condiciones.

La migración es **de plataforma de cómputo**, no de la ciencia ni de la arquitectura de
los modelos: el objetivo es que **el mismo código produzca los mismos resultados** sobre
hardware Intel.

---

## 2. Resumen del análisis de viabilidad

Se auditó la base de código completa (ver el informe de viabilidad previo, 2026-07-23).
Conclusión: **migración de riesgo BAJO**. El proyecto está escrito en **PyTorch idiomático
y agnóstico de dispositivo**: el hardware se selecciona con una sola variable `device`
(`torch.device("cuda" if torch.cuda.is_available() else "cpu")`) y todo se mueve con
`.to(device)`.

Hallazgos clave (verificados por búsqueda exhaustiva, no supuestos):

- **No hay** AMP / `autocast` / `GradScaler` (entrena en **FP32 puro**).
- **No hay** `DistributedDataParallel`, `DataParallel`, NCCL ni multi-GPU (todo single-GPU,
  `--gres=gpu:1`).
- **No hay** `torch.compile`, CUDA Graphs, Streams ni Events.
- **No hay** kernels CUDA personalizados ni extensiones C++/CUDA compiladas.
- **No hay** `torch_geometric`, `torch_scatter`, `torch_sparse`, `torch_cluster`,
  `pytorch_lightning`, `accelerate` ni `torchvision`.
- **No hay** un solo `.cuda()` explícito en todo el árbol.

Las únicas referencias a CUDA son cuatro patrones repetidos, **todos ya protegidos por
guardas** (`if device.type == 'cuda'` / `if torch.cuda.is_available()`), de modo que el
código ya corre en CPU sin cambios y correrá en XPU con modificaciones mínimas y
localizadas. Los modelos usan exclusivamente operadores estándar soportados por el backend
XPU (`Linear`, `LayerNorm`, `Embedding`, `bmm`, `softmax`, `masked_fill`, `nan_to_num`,
`softplus`, `relu`, `Conv2d`, `MaxPool2d`).

**El grueso del esfuerzo no está en el código, sino en el entorno HPC**: construir el
build de PyTorch-XPU, cargar el runtime oneAPI y reescribir los scripts SLURM y las rutas
para el nuevo clúster.

---

## 3. Hardware de Clementina XXI

Cada nodo de cómputo:

| Componente | Especificación |
|---|---|
| **CPU** | 2× Intel Xeon Max 9462 — 64 cores en total, 2 sockets |
| **Memoria** | 64 GB HBM (en paquete) + 512 GB RAM (DDR) |
| **GPU** | 4× Intel Data Center GPU Max 1550 (Ponte Vecchio) |
| **Por GPU** | 2 *tiles*/stacks por tarjeta, 128 GB HBM2e, unidades **XMX** (matrix engines) para BF16/FP16 |
| **Dispositivos XPU visibles** | Hasta **8** (4 tarjetas × 2 tiles), según cómo se expongan |
| **Software** | Stack **Intel oneAPI** (DPC++/Level-Zero), driver Intel GPU; PyTorch con backend **XPU** |

Notas de interés para el proyecto:

- La **HBM de alto ancho de banda** favorece cargas limitadas por memoria; sin embargo, en
  E3 el modelo es pequeño (~70k parámetros) y el cuello de botella real es la carga de datos
  (ver [§8](#8-riesgos-identificados)).
- Las **XMX** habilitan BF16/FP16 acelerado — una palanca de rendimiento futura, **fuera del
  alcance de esta primera migración** (se mantiene FP32 para preservar paridad de resultados).
- Los **8 dispositivos XPU por nodo** habilitan multi-GPU futuro (DDP + backend CCL), también
  **fuera de alcance** por ahora (E3 es single-GPU).

---

## 4. Diferencias respecto al clúster anterior

| Aspecto | login-1 (anterior) | Clementina XXI (nuevo) |
|---|---|---|
| GPU | NVIDIA A10 (23 GB, CUDA) | Intel GPU Max 1550 (PVC, XPU) |
| Backend PyTorch | `cuda` | `xpu` |
| Runtime | CUDA / cuDNN | Intel oneAPI (DPC++, Level-Zero) |
| Partición SLURM | `gpua10_hi` | **`gpunode`** (límite 2 días) |
| Recurso GPU en SLURM | `#SBATCH --gres=gpu:1` | **`#SBATCH --gres=gpu:intel_xt1550:1`** (GRES `intel_xt1550`, 4/nodo) |
| Activación de entorno | `conda activate .../NMR_env` | `conda activate /data/contrib/pci_78/envs/nmr_xpu` (+ `module load intel/2025.3.0` si hace falta) |
| Módulos | — | `intel/2025.3.0` (oneAPI) · `apptainer/1.4.5-3` (contenedores, fallback) |
| Level Zero | — | `ZE_FLAT_DEVICE_HIERARCHY=FLAT` (8 tiles); **`unset ZE_AFFINITY_MASK`** en cada job |
| Datasets | `/home/lpassaglia.iquir/DB_200k` | `/data/contrib/pci_78/Lucas/DB_202K` |
| Repositorio | clon en `$HOME` del usuario | clon en el espacio de trabajo de cada usuario |
| Nº de GPUs usadas | 1 (A10) | 1 (single-tile) en esta fase; hasta 8 disponibles |

> Valores **confirmados** el 2026-07-23 con `sinfo`/`module avail`/`scontrol show node` en
> Clementina (ver [§10.0](#100-datos-del-clúster-confirmados-2026-07-23)). Lo único que resta
> validar en un nodo GPU es `torch.xpu.is_available()` → `True` (checkpoint de la Fase 0).

---

## 5. Objetivos de la migración

**En alcance (esta iteración):**

1. Migrar **únicamente el experimento campeón**: **Exp E Fase 3 — Set Transformer**
   (`experiments/E3_dos_conjuntos/`, config `config_settransformer.yaml`). Es el mejor
   resultado del proyecto según `docs/Runs/RESULTS.md`: **EMA asistida 91.35 %**, best val
   loss **0.0097** — la primera vez que se cruza el objetivo de ~90 % asistida con
   evaluación limpia.
2. Que E3 **entrene, evalúe y prediga sobre Intel XPU** produciendo resultados equivalentes
   a los de CUDA (paridad numérica dentro de tolerancia).
3. Crear un **entorno compartido para todo el grupo** en `/data/contrib/pci_78`, reproducible
   y activable por cualquier integrante sin reconstruir dependencias.
4. Dejar el proyecto **desacoplado de las rutas físicas** (rutas configurables, sin
   hardcodear) para poder cambiar de clúster/ubicación con facilidad.

**Fuera de alcance (por ahora, explícitamente):**

- Migrar el resto de los experimentos (V10, B, C, E2, F, D). Se evaluará **después** de
  validar E3.
- Precisión mixta BF16/FP16 (XMX).
- Entrenamiento multi-GPU / multi-tile (DDP + CCL).
- Optimización del pipeline de datos.

> **Por qué E3 primero:** es el modelo con mejores resultados y, además, el más simple de
> migrar (no usa la imagen HSQC ni `Conv2d` pesado; su entrada son listas de picos y es un
> modelo de ~70k parámetros). Es el piloto ideal para validar la plataforma XPU con el menor
> riesgo posible antes de generalizar.

---

## 6. Estrategia elegida

Migración **conservadora, incremental y validada**, tratada como una migración de software
para producción:

- **Un cambio pequeño por vez.** Nada de refactors masivos.
- **El proyecto se mantiene siempre en estado funcional.** El código actual ya corre en CPU
  y en CUDA; los cambios de XPU serán aditivos y guardados, sin romper esas rutas.
- **Se explica y justifica antes de modificar** cualquier archivo.
- **Se valida al final de cada fase** y se **frena para revisión** antes de continuar. No se
  avanza de fase sin aprobación.
- **El experimento campeón (E3) es el piloto.** Recién con E3 validado en XPU se decide si
  extender la migración al resto.
- **Prioridad: correctitud y mantenibilidad sobre velocidad.** Preferible una migración
  lenta, limpia y bien documentada.

---

## 7. Fases del trabajo

Cada fase termina en un **checkpoint de validación** y **espera aprobación** antes de la
siguiente.

| Fase | Nombre | Objetivo | Criterio de validación |
|---|---|---|---|
| **0** | Entorno compartido XPU | Construir el env conda compartido en `/data/contrib/pci_78`, con PyTorch-XPU + oneAPI. Relevar partición, módulos y `--gres` reales de Clementina. | `python -c "import torch; print(torch.xpu.is_available())"` → `True`; `torch.xpu.get_device_properties(0)` lista la GPU Max 1550. |
| **1** | Abstracción de dispositivo (solo E3) | Añadir soporte `xpu` a la selección de dispositivo y a las guardas (`synchronize`, `pin_memory`, `manual_seed`) en E3. Cambios pequeños, aditivos y guardados. | Smoke tests de E3 pasan en CPU **y** en XPU; el código sigue corriendo en CUDA sin cambios de comportamiento. |
| **2** | Validación funcional / paridad | Correr los smoke tests de E3 en XPU; test de paridad numérica CPU↔XPU en un batch (foco en la atención enmascarada `masked_fill(-inf)`+softmax+`nan_to_num`). Cargar un checkpoint existente y reproducir una evaluación conocida. | Shapes correctos; diferencia CPU↔XPU dentro de tolerancia FP32; evaluación reproduce EMA esperada. |
| **3** | Entorno de ejecución HPC | Reescribir los `.sh` SLURM de E3 para Clementina XXI (partición, `--gres`, `module load` oneAPI en lugar del `conda activate` de login-1) y reapuntar rutas de datos. | `sbatch` encola y corre sin error de recursos/entorno; el job ve la GPU. |
| **4** | Corrida de referencia (single-tile, FP32) | Entrenamiento completo de E3-SetTransformer en 1 tile de PVC. Comparar val loss y EMA contra la corrida A10 (0.0097 / 91.35 %). | Resultados equivalentes a CUDA dentro de tolerancia → **migración de E3 validada**. |
| **5** *(opcional)* | Optimización | BF16 vía XMX; escalado multi-tile/multi-GPU (DDP + CCL); alivio del pipeline de datos. Cada uno medido contra el baseline de Fase 4. | Mejora de throughput sin degradar EMA. |

**Esfuerzo estimado:** Fases 0–4 (migración funcional de E3) = **bajo**, dominado por el
setup de entorno HPC, no por código. Fase 5 = variable según cuánto rendimiento se busque.

---

## 8. Riesgos identificados

Clasificación: **Bajo / Medio / Alto / Crítico**. Acotada al alcance actual (E3).

| Riesgo | Nivel | Detalle y mitigación |
|---|---|---|
| Operadores del modelo no soportados en XPU | **Bajo** | E3 usa solo ops estándar (`Linear`, `LayerNorm`, `Embedding`, `bmm`, `softmax`, `masked_fill`, `nan_to_num`, `softplus`). Todas soportadas. |
| Selección de dispositivo sin rama `xpu` | **Bajo** | Cambio de 1–2 líneas, aditivo y guardado (Fase 1). |
| `torch.cuda.synchronize` / `manual_seed` | **Bajo** | Ya guardados; se reemplazan por variantes agnósticas/XPU o quedan inertes. |
| Build torch-XPU + oneAPI correcto para PVC | **Medio** | Punto de fricción real: casar versiones torch-XPU / oneAPI / driver Level-Zero del clúster. Es setup de entorno (Fase 0), no de código. |
| Reproducibilidad del entorno compartido | **Medio** | Sin `requirements`/`environment.yml` versionado hoy. Mitigación: congelar `environment.yml` en el repo al construir el env (Fase 0). |
| Paridad numérica FP32 CPU↔XPU (atención enmascarada) | **Bajo** | Patrón `masked_fill(-inf)`+softmax ya mitigado con `nan_to_num`. Se valida con test explícito (Fase 2). |
| Cuello de botella de I/O de datos | **Medio** | E3 carga los `.npz`/`.npy` completos a RAM en `__init__` y usa `num_workers=0`. No lo causa la migración, pero limita el rendimiento igual que en A10. Fuera de alcance atacarlo ahora. |
| Rutas / SLURM / módulos del clúster anterior | **Medio** | Hardcodeados a login-1. Se reescriben en Fase 3; se documentan en [§13](#13-organización-de-datos-y-rutas). |
| Kernels CUDA / extensiones compiladas | **Inexistente** | No hay ninguno. Riesgo nulo — el bloqueante clásico de estas migraciones no aplica. |
| Aprovechar 8 tiles / BF16 | **Bajo (oportunidad)** | No migrarlo ahora = rendimiento sin explotar, no un bloqueo. Fase 5. |

**No se identificó ningún riesgo Crítico.**

---

## 9. Decisiones técnicas

Registro de decisiones (estilo ADR abreviado). Cada una se revisará si aparece evidencia
en contra.

- **D1 — Migrar E3 primero, como piloto.** Mejor resultado del proyecto y el más simple de
  portar. El resto espera a validar E3.
- **D2 — PyTorch con backend XPU (upstream) como base; IPEX (Intel Extension for PyTorch)
  opcional.** El código actual no importa IPEX y no lo necesita para funcionar. Se evaluará
  IPEX solo como optimización posterior (Fase 5). Empezar con XPU upstream reduce piezas
  móviles.
- **D3 — FP32 primero, BF16 después.** Preservar paridad de resultados con CUDA es prioritario
  sobre velocidad. BF16/XMX queda para Fase 5.
- **D4 — Single-tile primero, multi-GPU después.** E3 hoy es single-GPU; no se introduce
  distributed training en esta iteración.
- **D5 — Selección de dispositivo `cuda → xpu → cpu`.** Se generaliza la lógica actual para
  detectar XPU sin romper CUDA ni CPU. Cambio aditivo y guardado (detalle en [§12](#12-mapa-de-cambios-previstos-en-el-código-solo-e3)).
- **D6 — Entorno compartido vía Conda, en prefijo compartido.** Se elige **conda** (no
  contenedor, no venv+módulos como primario) por ser el flujo que el grupo ya conoce
  (`NMR_env`) y por permitir un prefijo compartido activable por todos. Ubicación:
  `/data/contrib/pci_78/envs/`. Ver [§10](#10-cómo-reproducir-el-entorno). *(Si el stack
  oneAPI/PyTorch-XPU resultara frágil de reproducir con conda, se reconsiderará Apptainer en
  Fase 0.)*
- **D7 — Desacople de rutas, documentado ahora, implementado después.** Las rutas físicas
  (datasets, checkpoints, repo) no se hardcodean. Por ahora **solo se documentan** las nuevas
  ubicaciones; el refactor de configuración se hará como paso controlado (ver [§13](#13-organización-de-datos-y-rutas)),
  no en este documento.
- **D8 — El código sigue corriendo en CUDA y CPU tras la migración.** Los cambios son
  aditivos: nada de reemplazar `cuda` por `xpu` a secas. Esto mantiene el proyecto funcional
  en cualquier plataforma y permite comparar.

---

## 10. Cómo reproducir el entorno

> **Estado: Fase 0 en ejecución.** Los datos del clúster ya están confirmados (tabla 10.0).
> Lo único pendiente de validar en un nodo GPU es `torch.xpu.is_available()`. El método de
> instalación de PyTorch-XPU no está documentado en la wiki de Clementina, así que se usa el
> canal oficial de PyTorch (wheels XPU upstream) con fallback vía módulo oneAPI / contenedor
> apptainer.

### 10.0 Datos del clúster confirmados (2026-07-23)

| Dato | Valor |
|---|---|
| Nodo login/gestión | `snmgt01` (Clementina XXI, nombre de fantasía "Clementina") |
| Partición GPU | `gpunode` (TIMELIMIT 2 días) |
| GRES | `gpu:intel_xt1550:4` → 4× Intel GPU Max 1550 por nodo |
| Tiles | 2 por GPU → **8 tiles/nodo**, 64 GB c/u (512 GB VRAM total) |
| CPU / RAM por nodo | 64 cores / ~506 GB |
| Toolkit oneAPI | `intel/2025.3.0` (árbol `/data/shared/oneapi/2025.3`; carga DPC++, MKL, TBB, UMF) |
| Contenedores | `apptainer/1.4.5-3` (cargado por defecto) — fallback |
| Runtime XPU | operativo (apps `lammps-*-xpu`, `amber-sycl`); `/dev/dri/renderD128-131` visibles en el nodo GPU |
| Docs / soporte | https://docs.clementinaxxi.org.ar · tickets: https://tickets.clementinaxxi.org.ar |

### 10.1 Variables Level Zero (imprescindibles al ejecutar)

- **`ZE_FLAT_DEVICE_HIERARCHY=FLAT`** (default): expone los 8 tiles como 8 dispositivos XPU
  de 64 GB — el modo recomendado para cargas single-tile, **el que usa E3** (un tile sobra
  para ~70k parámetros). `COMPOSITE` daría 4 dispositivos de 128 GB.
- **`unset ZE_AFFINITY_MASK` en todos los jobs de SLURM.** La wiki de Clementina advierte que
  SLURM setea mal `ZE_AFFINITY_MASK` y oculta tiles; sin este `unset`, PyTorch puede no ver
  la GPU.
- Selección opcional de dispositivo SYCL: `ONEAPI_DEVICE_SELECTOR=level_zero:*`.

### 10.2 Diseño del entorno compartido (Conda, prefijo compartido)

Objetivo: **un único entorno** que todo el grupo activa por ruta, sin reconstruir nada.
Se construye en el **nodo de login** (es trabajo de CPU: conda + pip); solo la *validación*
final se hace en un nodo GPU.

- **Ubicación:** `/data/contrib/pci_78/envs/nmr_xpu`.
- **Se crea con `--prefix` directamente en la ubicación final** (los entornos conda **no son
  relocalizables**: no construir en otro lado y mover).
- **Permisos:** lo mantiene un responsable (Lucas); el grupo tiene **lectura + ejecución**.
- **Reproducibilidad:** al terminar, exportar `environment_xpu.yml` y **versionarlo en el repo**.

### 10.3 Construcción (recipe, se corre en el nodo de login)

```bash
# Paso 1 — crear el env en el prefijo compartido (los envs conda NO son relocalizables)
conda create --prefix /data/contrib/pci_78/envs/nmr_xpu python=3.11 -y
conda activate /data/contrib/pci_78/envs/nmr_xpu

# Paso 2 — PyTorch con backend XPU (canal oficial upstream; el wheel trae su propio
#          runtime oneAPI). Solo torch: E3 no usa torchvision/torchaudio.
pip install torch --index-url https://download.pytorch.org/whl/xpu

# Paso 3 — dependencias del proyecto (todas CPU, idénticas a las de CUDA)
pip install numpy scipy pandas pyyaml matplotlib rdkit h5py streamlit altair

# Paso 4 — permisos de grupo (lectura + ejecución)
chmod -R g+rX /data/contrib/pci_78/envs/nmr_xpu
```

> E3 en sí solo necesita `torch`, `numpy`, `rdkit`, `pyyaml`; el resto se instala para que el
> env sea reutilizable por el resto del proyecto y por el grupo.

### 10.4 Validación en un nodo GPU (checkpoint de la Fase 0)

`torch.xpu.is_available()` solo da `True` en un nodo `gpunode`, no en login.

```bash
srun -p gpunode --gres=gpu:intel_xt1550:1 --pty bash
unset ZE_AFFINITY_MASK                        # gotcha de SLURM (ver 10.1)
conda activate /data/contrib/pci_78/envs/nmr_xpu
python -c "import torch; print('torch', torch.__version__); \
print('xpu?', torch.xpu.is_available()); \
print('n_dev', torch.xpu.device_count()); \
print(torch.xpu.get_device_properties(0))"
```

- **Esperado:** `xpu? True`, `n_dev` ≥ 1, y las propiedades nombrando *Intel ... Max 1550*.
- **Si da `False`:** reintentar cargando el toolkit del sistema antes del `python`:
  `module load intel/2025.3.0`. Si aun así falla → ticket a soporte (posible mismatch
  driver ↔ versión del wheel) y evaluar el fallback por contenedor `apptainer`.

> **Resultado (2026-07-23, nodo `cn073`):** `torch 2.13.0+xpu` · `xpu? True` · `n_dev 2`
> (los 2 tiles de la GPU pedida, modo FLAT, 64 GB c/u) · *Intel(R) Data Center GPU Max 1550*,
> Level-Zero, driver `1.6.33578`, `has_fp16=1`, `has_fp64=1`. **No hizo falta
> `module load intel/2025.3.0`**: el wheel de PyTorch trae su propio runtime oneAPI. Fase 0 ✔.

Una vez validado, congelar el env para el grupo:

```bash
conda env export --prefix /data/contrib/pci_78/envs/nmr_xpu > env/environment_xpu.yml
```

> Freeze validado del 2026-07-23 versionado en [`env/environment_xpu.yml`](../env/environment_xpu.yml).
> El wheel `torch==2.13.0+xpu` arrastra vía pip el runtime Intel completo (`intel-sycl-rt`,
> `dpcpp-cpp-rt`, `mkl`, `onemkl-sycl-*`, `umf`, `tbb`) **y `oneccl`** — este último habilita
> el multi-GPU/multi-tile con DDP de la Fase 5 sin dependencias de sistema extra. También trae
> `triton-xpu` (base para `torch.compile` en XPU, opcional a futuro).

### 10.5 Uso por cualquier integrante del grupo

```bash
unset ZE_AFFINITY_MASK
conda activate /data/contrib/pci_78/envs/nmr_xpu
cd <tu_clon_del_repo>/experiments/E3_dos_conjuntos
python tests/test_forward_settransformer.py   # smoke test antes de entrenar (rule 5)
```

### 10.6 Dependencias del proyecto (inventario)

Ninguna es específica de GPU salvo PyTorch. `numpy`, `scipy` (solo en `E_peaks_prep`),
`pandas`, `matplotlib`, `pyyaml`, `rdkit`, `h5py`, `streamlit`, `altair` corren en CPU y no
cambian entre CUDA y XPU. **E3 en particular** solo necesita: `torch`, `numpy`, `rdkit`
(fórmula molecular), `pyyaml`.

---

## 11. Estado actual (checklist)

Leyenda: `[x]` hecho y validado · `[~]` en curso · `[ ]` pendiente.

- [x] **Análisis de viabilidad** (informe técnico, 2026-07-23).
- [x] **Documento de migración** (este archivo).
- [x] **Etapa 2 — Rama `feature/intel-xpu-support`** creada (desde `main`, `ea3324a`); todo el desarrollo posterior va ahí.
- [x] **Fase 0 — Entorno compartido XPU** en `/data/contrib/pci_78/envs/nmr_xpu`. Validado el 2026-07-23 en `cn073` (`gpunode`): **`torch 2.13.0+xpu`, `torch.xpu.is_available() == True`**, GPU *Intel Data Center GPU Max 1550* (Level-Zero, driver `1.6.33578`). `device_count()==2` = los 2 tiles de la GPU pedida (FLAT). Entorno congelado en [`env/environment_xpu.yml`](env/environment_xpu.yml).
- [x] **Fase 1 — Abstracción de dispositivo** en E3 (aditiva y guardada). Nuevo módulo
  [`device_utils.py`](../experiments/E3_dos_conjuntos/device_utils.py) con `pick_device()`,
  `wants_pin_memory()`, `synchronize()` y `seed_everything()`; integrado en `train.py`,
  `evaluate.py` y `dump_predictions.py`. `system.device` del config pasa a ser **respetado**
  (`"auto"|"cuda"|"xpu"|"cpu"`), los 7 configs de E3 quedaron en `"auto"`. Validado en CPU:
  los 7 tests de E3 pasan (incluye los 11 casos nuevos de
  [`tests/test_device_utils.py`](../experiments/E3_dos_conjuntos/tests/test_device_utils.py)).
  **Falta validar en XPU real** (es el checkpoint de la Fase 2).
- [x] **Fase 2 — Validación funcional / paridad** CPU↔XPU en E3. **Validada el 2026-07-23 en
  `cn073`.** `tests/test_device_utils.py`: 11/11, `pick_device('auto')` → `xpu` en hardware
  real. [`tests/test_paridad_cpu_xpu.py`](../experiments/E3_dos_conjuntos/tests/test_paridad_cpu_xpu.py):
  forward, molécula 100% enmascarada (el caso de riesgo,
  `masked_fill(-inf)`+softmax+`nan_to_num`), gradientes y determinismo — los 4 pasan, con
  diferencias CPU↔XPU de `~1e-7`–`1e-9` (ruido de FP32 entre hardware, muy por debajo de la
  tolerancia `atol=2e-5`). **Migración de E3 a XPU funcionalmente validada.**
- [x] **Fase 3 — SLURM + rutas** de E3 para Clementina XXI. `sbatch run_train_settransformer_clementina.sh`
  validado de punta a punta el 2026-07-23 (job 1489556, `cn073`): los 4 chequeos previos
  pasaron en el nodo real, `torch.xpu.is_available() == True` con `n_dev=2`, y el entrenamiento
  arrancó con `[INFO] Dispositivo: xpu`. Encontrado y corregido en el camino: `set -euo pipefail`
  mataba el job al activar conda (los hooks de oneAPI referencian variables sin default, ej.
  `SETVARS_CALL`) — solucionado acotando `set +u`/`set -u` alrededor de la activación (ver §13.6).
- [x] **Fase 4 — Corrida de referencia** E3-SetTransformer en XPU (single-tile, FP32); paridad con A10.
  **Cerrada el 2026-07-24.** Job de entrenamiento 1489559 + evaluación 1489606 (`cn073`, `.err`
  limpios en los dos). Resultado: **val loss 0.0086** (vs 0.0097 en A10) y **EMA asistida 92.12%
  / 92.14% v2** (vs 91.35% en A10) — dentro de tolerancia, de hecho ligeramente mejor. La pequeña
  diferencia viene de que el scheduler disparó en un punto distinto por ruido de FP32 acumulado
  entre hardware (ya visto en la paridad numérica de Fase 2), no de un error de implementación.
  Contrapartida esperada: ~1.8× más lento (70.3 min vs 39.0 min) — FP32 puro sin XMX/BF16, fuera
  de alcance de esta fase. Detalle completo y logs crudos en
  `docs/Runs/RESULTS.md` (sección "Migración XPU") y `docs/Runs/XPU_Clementina_E3_settransformer/`.
  **Migración de E3 a Intel XPU validada de punta a punta.**
- [ ] **Fase 5 *(opcional)* — Optimización** (BF16, multi-tile, pipeline de datos).

---

## 12. Mapa de cambios previstos en el código (solo E3)

> **Implementado (Fase 1, 2026-07-23).** La tabla queda como registro de qué se tocó y qué
> no. Los cambios son **aditivos y guardados**: el código sigue corriendo en CUDA y en CPU.

Puntos de contacto con CUDA en E3 (referencias verificadas):

| Archivo | Línea(s) | Patrón actual | Cambio previsto |
|---|---|---|---|
| `experiments/E3_dos_conjuntos/train.py` | 24–25 | `if torch.cuda.is_available(): torch.cuda.manual_seed(...)` | Añadir rama XPU para la semilla (o dejar que `torch.manual_seed` la fije; ya está guardado). |
| `train.py` | 26–27 | `torch.backends.cudnn.deterministic/benchmark` | Inocuo en XPU (no-op). Sin cambio funcional; se puede dejar. |
| `train.py` | 117 | `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")` | Generalizar a `cuda → xpu → cpu` (helper único). |
| `train.py` | 126 | `use_pin = ... and device.type == 'cuda'` | Extender guarda a incluir `'xpu'`. |
| `train.py` | 161–162 | `if device.type == 'cuda': torch.cuda.synchronize()` | `torch.xpu.synchronize()` en XPU (o sync genérica). |
| `experiments/E3_dos_conjuntos/evaluate.py` | ~214 | selección de `device` | Igual que `train.py:117`. |
| `evaluate.py` | ~231 | guarda de `pin_memory` | Igual que `train.py:126`. |
| `evaluate.py` | `torch.load(..., map_location=device)` | Ya portable | Sin cambio (los checkpoints `state_dict` son agnósticos). |
| `experiments/E3_dos_conjuntos/dump_predictions.py` | ~53, ~80 | `device` + `pin_memory` | Igual que arriba. |
| `experiments/E3_dos_conjuntos/config_settransformer.yaml` | `system.device: "cuda"` | Campo informativo (el código usa `is_available()`, no este valor) | Actualizar a `"xpu"`/`"auto"` por coherencia documental. |

### 12.1 Lo que se implementó (Fase 1)

Módulo nuevo `experiments/E3_dos_conjuntos/device_utils.py` (al lado de `split_utils.py`):

| Función | Rol |
|---|---|
| `pick_device(prefer, has_cuda=, has_xpu=)` | Selección `cuda → xpu → cpu` (D5) |
| `wants_pin_memory(device)` | `True` en cuda y xpu, `False` en cpu |
| `synchronize(device)` | Despacha al backend correcto; no-op en cpu |
| `seed_everything(seed)` | Semillas de python/numpy/torch + acelerador presente |

Dos decisiones tomadas en la implementación, que **no** estaban en el boceto original:

- **D9 — `system.device` del config pasa a ser respetado.** Antes era decorativo (el código
  hacía `is_available()` e ignoraba el YAML). Ahora acepta `"auto"|"cuda"|"xpu"|"cpu"`, lo
  que cumple la regla dura 3 del proyecto (*nada hardcodeado; todo sale del config*).
  `"auto"` preserva exactamente el comportamiento histórico.
- **D10 — Pedir un backend ausente levanta `RuntimeError`, no cae a CPU.** Si el config dice
  `xpu` y el job no ve la GPU, revienta al arrancar. El fallback silencioso haría entrenar
  días en CPU sin enterarse — el mismo patrón de fallo caro que ya está en las reglas duras
  (deadlock de `num_workers`, split corrupto). `"auto"` sigue cayendo a CPU sin drama, que es
  lo que necesitan los smoke tests.

> **Corolario de D9:** los 7 configs de E3 decían `device: "cuda"`. Al volverse significativo
> el campo, dejarlos así los habría hecho **exigir** CUDA y romper en cualquier máquina sin
> ella. Los 7 quedaron en `"auto"`. Para la corrida de producción en Clementina se pondrá
> `"xpu"` explícito (Fase 3), que es justamente donde D10 protege.

**Hallazgo:** `torch.xpu` **existe como atributo incluso en builds CPU/CUDA sin soporte XPU**
(verificado en `torch 2.13.0+cpu`), así que el `hasattr(torch, "xpu")` del boceto original no
sirve como probe. El probe real llama `torch.xpu.is_available()` dentro de un `try/except`.

**Validación (2026-07-23, en CPU, local):** los 7 tests de E3 pasan, incluidos los 11 casos
nuevos de `tests/test_device_utils.py`, que cubren las tres plataformas inyectando los probes
(`has_cuda`/`has_xpu`) — ninguna máquina tiene CUDA y XPU a la vez. `train.py`, `evaluate.py`
y `dump_predictions.py` compilan e importan. **Falta correr en XPU real: es la Fase 2.**

**Lo que NO se toca:** la arquitectura del modelo (`model_e3_settransformer.py`), el dataset
(`dataset_e3.py`), la lógica de split (`split_utils.py`), el oráculo (`oraculo.py`, NumPy
puro), ni la loss. La migración es de plataforma, no de modelo.

Archivos SLURM a reescribir en **Fase 3** (no Fase 1): `run_train_settransformer.sh`,
`run_eval.sh`, `run_train_deepsets.sh`, `run_train_scaling.sh`. Cambios para Clementina XXI:

- Cabeceras `#SBATCH`: `--partition=gpua10_hi` → **`--partition=gpunode`**; `--gres=gpu:1` →
  **`--gres=gpu:intel_xt1550:1`**.
- Entorno: `source .../conda.sh` + `conda activate .../NMR_env` (login-1) →
  `conda activate /data/contrib/pci_78/envs/nmr_xpu` (+ `module load intel/2025.3.0` si la
  validación de Fase 0 lo requirió).
- **Añadir `unset ZE_AFFINITY_MASK`** antes de invocar `python` (gotcha de SLURM en
  Clementina; ver [§10.1](#101-variables-level-zero-imprescindibles-al-ejecutar)).
- Ruta del repo: `cd ~/nmr-hsqc-to-vector-/...` → el clon real del usuario.
- Rutas de datos: `base_dir` del config → `/data/contrib/pci_78/Lucas/DB_202K` (ver [§13](#13-organización-de-datos-y-rutas)).

---

## 13. Organización de datos y rutas

### 13.1 Ubicaciones en Clementina XXI

- **Datasets (compartidos, ya existentes):** `/data/contrib/pci_78/Lucas/DB_202K`.
  Son **exactamente los mismos archivos** que en el clúster anterior. Para E3 se necesitan:
  - `peaks_pkl_202465.npz` (crosspeaks C-H)
  - `peaks_13c_202465.npz` (¹³C, con cuaternarios)
  - `vectors_13c_19v_202465.npy` (labels, 19 clases)
  - `smiles_202465.npy` (para la fórmula molecular)
  - `val_indices_frozen.npy` (split congelado de Exp D)
- **Entorno compartido:** `/data/contrib/pci_78/envs/nmr_xpu` (ver [§10](#10-cómo-reproducir-el-entorno)).
- **Repositorio:** clonado en el **espacio de trabajo de cada usuario** (no compartido).

### 13.2 Principio de diseño (a implementar como paso controlado, no ahora)

El proyecto debe quedar **desacoplado de las rutas físicas**:

- Rutas **configurables**, no hardcodeadas en los `.py`.
- Fácil de cambiar entre clústeres/ubicaciones (un solo lugar a editar).
- Consistente con la regla dura del proyecto: *"nada hardcodeado; rutas y constantes salen
  siempre del config"* (ver `CLAUDE.md` y `config/db.yaml`).

Situación actual (a resolver más adelante): `config_settransformer.yaml` trae
`base_dir: "/home/lpassaglia.iquir/DB_200k"` hardcodeado, y los `.sh` traen rutas absolutas de
login-1. La reapuntada a `/data/contrib/pci_78/Lucas/DB_202K` se hará en la **Fase 3** como
cambio controlado y revisado.

### 13.3 Cómo quedó resuelto (Fase 3, 2026-07-23)

Se implementó el desacople anunciado en D7, sin duplicar el YAML por clúster (duplicarlo
garantiza que tarde o temprano se desincronicen los hiperparámetros, y eso sí rompe la
comparabilidad de la regla dura 8). `config_settransformer.yaml` usa sintaxis de shell:

```yaml
base_dir: "${NMR_DATA_DIR:-/home/lpassaglia.iquir/DB_200k}"
device:   "${NMR_DEVICE:-auto}"
```

La expansión la hace [`config_utils.py`](../experiments/E3_dos_conjuntos/config_utils.py)
(`expand_env` + `load_config`), que reemplaza los tres `load_config` duplicados de
`train.py`, `evaluate.py` y `dump_predictions.py`. Comportamiento:

| Contexto | `NMR_DATA_DIR` | Resultado |
|---|---|---|
| login-1 (histórico) | sin exportar | `/home/lpassaglia.iquir/DB_200k`, `device: auto` — **intacto** |
| Clementina (`.sh` nuevos) | exportada | `/data/contrib/pci_78/Lucas/DB_202K`, `device: xpu` |

Una `${VAR}` sin valor **y** sin default levanta `RuntimeError` al arrancar, en vez de dejar el
literal y morir mucho después con un `FileNotFoundError` de path absurdo.

**`NMR_DEVICE=xpu` en los `.sh` de Clementina es deliberado:** combinado con D10, si el job no
ve la GPU el entrenamiento aborta al arrancar en lugar de caer a CPU y quemar horas de cola.

**Ubicaciones (repo en Clementina):** `/home/lpassaglia/nmr-hsqc-to-vector-`.

### 13.4 Conda en los jobs (verificado 2026-07-23)

En un job no interactivo conda **no está inicializado**: hay que sourcear
`<base>/etc/profile.d/conda.sh` antes de `conda activate`, o el job muere con
"CommandNotFoundError: Your shell has not been properly configured".

`conda info --base` en Clementina devuelve **`/home/<user>/miniconda3`** — es decir, cada
usuario tiene **su propio conda en el HOME**, no hay uno compartido. Por eso los `.sh` derivan
la ruta de `$HOME` en vez de hardcodearla:

```bash
CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
```

Así el mismo `.sh` sirve para cualquier integrante del grupo (objetivo 3 del §5). Si alguien
tiene conda en otro lado, exporta `CONDA_SH` y listo. El script **aborta con un mensaje
explícito** si no lo encuentra, en vez de fallar de forma oscura. Lo mismo con el repo:
`NMR_REPO` con default `$HOME/nmr-hsqc-to-vector-`.

> **Lo compartido es el env** (`/data/contrib/pci_78/envs/nmr_xpu`), no el conda que lo activa.

### 13.5 Proxy de red (verificado 2026-07-23)

Los nodos de Clementina **no resuelven DNS externo directamente**: un `git clone` desde GitHub
falla con `Could not resolve host: github.com`. Hay que exportar el proxy del clúster antes:

```bash
export http_proxy=172.28.3.3:3128
export https_proxy=172.28.3.3:3128
```

Con eso el clone funciona normalmente. Necesario para clonar y para cualquier `pip install`.

### 13.6 `git pull`/`clone` falla con el env activado (verificado 2026-07-23)

Con `nmr_xpu` activado, `git pull`/`clone` por HTTPS puede fallar así:

```
/usr/libexec/git-core/git-remote-https: symbol lookup error: /lib64/libldap.so.2:
undefined symbol: EVP_md2, version OPENSSL_3.0.0
```

El env prioriza su propio OpenSSL en `LD_LIBRARY_PATH`, y `git-remote-https` (binario del
sistema) termina cargando una combinación de libs incompatible entre sí. Solución (probada en
`cn073`): limpiar la variable solo para ese comando, sin desactivar el env —

```bash
LD_LIBRARY_PATH= git pull
```

---

*Fin del documento. Se actualiza a medida que avanzan las fases (ver
[§11 checklist](#11-estado-actual-checklist)).*
