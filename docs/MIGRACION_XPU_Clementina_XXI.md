# Migración a Intel XPU — Clementina XXI

> **Documento de referencia permanente del proyecto.**
> Describe por qué, cómo y con qué criterios se migra el proyecto NMR HSQC→Vector
> desde GPUs NVIDIA (CUDA, clúster login-1) a GPUs Intel (XPU / Ponte Vecchio,
> clúster Clementina XXI). Pensado para que cualquier integrante del grupo entienda,
> dentro de uno o dos años, las decisiones tomadas y pueda reproducir el entorno.
>
> **Autor de la migración:** Lucas Passaglia (UCA Team).
> **Fecha de inicio:** 2026-07-23.
> **Estado global:** en curso — ver [§11 Estado actual (checklist)](#11-estado-actual-checklist).

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
- [ ] **Fase 1 — Abstracción de dispositivo** en E3 (aditiva y guardada).
- [ ] **Fase 2 — Validación funcional / paridad** CPU↔XPU en E3.
- [ ] **Fase 3 — SLURM + rutas** de E3 para Clementina XXI.
- [ ] **Fase 4 — Corrida de referencia** E3-SetTransformer en XPU (single-tile, FP32); paridad con A10.
- [ ] **Fase 5 *(opcional)* — Optimización** (BF16, multi-tile, pipeline de datos).

---

## 12. Mapa de cambios previstos en el código (solo E3)

> **Todavía NO implementado.** Esta sección lista, para trazabilidad, exactamente qué se
> tocará en la Fase 1 y qué **no**. Los cambios serán **aditivos y guardados**: el código
> debe seguir corriendo en CUDA y en CPU.

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

Idea de la generalización de dispositivo (**ilustrativa**, se define en Fase 1):

```python
def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    return torch.device("cpu")
```

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

> **Importante:** por decisión explícita, **todavía no se modifican las rutas**. Esta sección
> las deja registradas para tenerlas presentes al diseñar cada fase; el cambio efectivo es un
> paso posterior con su propia validación.

---

*Fin del documento. Se actualiza a medida que avanzan las fases (ver
[§11 checklist](#11-estado-actual-checklist)).*
