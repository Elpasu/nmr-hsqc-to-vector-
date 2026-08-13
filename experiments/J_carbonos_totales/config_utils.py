# coding: ascii
"""config_utils.py -- carga de configs con expansion de variables de entorno.

Fase 3 de la migracion a Intel XPU (docs/MIGRACION_XPU_Clementina_XXI.md).

Problema que resuelve: `paths.base_dir` estaba hardcodeado a la ruta de
login-1 (`/home/lpassaglia.iquir/DB_200k`), asi que el mismo config no servia
en Clementina XXI (`/data/contrib/pci_78/Lucas/DB_202K`). En vez de duplicar
el YAML por cluster -- lo que garantiza que tarde o temprano se desincronicen
los hiperparametros -- los paths admiten la sintaxis de shell:

    base_dir: "${NMR_DATA_DIR:-/home/lpassaglia.iquir/DB_200k}"

El .sh de cada cluster exporta NMR_DATA_DIR; si nadie la exporta, se usa el
default y el comportamiento historico en login-1 queda intacto.

Cumple la regla dura 3 del proyecto (nada hardcodeado) y la decision D7 del
documento de migracion (desacople de rutas fisicas).
"""
import os
import re

import yaml

# ${VAR} o ${VAR:-default}. El default puede contener cualquier cosa menos '}'.
_VAR_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def expand_env(obj):
    """Expande ${VAR} y ${VAR:-default} en todos los strings de `obj`.

    Recorre dicts y listas recursivamente. Devuelve una estructura NUEVA (no
    muta la entrada). Los valores que no son string se dejan tal cual.

    Una ${VAR} sin valor y sin default levanta RuntimeError: dejarla literal
    haria que el error aparezca mucho despues como un FileNotFoundError con un
    path raro, en vez de fallar claro al arrancar.
    """
    if isinstance(obj, dict):
        return {k: expand_env(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [expand_env(v) for v in obj]
    if not isinstance(obj, str):
        return obj

    def _sub(m):
        name, default = m.group(1), m.group(2)
        val = os.environ.get(name)
        if val is not None:
            return val
        if default is not None:
            return default
        raise RuntimeError(
            f"La variable de entorno {name} no esta definida y el config no da "
            f"un default. Exportala (ej: export {name}=/ruta) o usa la sintaxis "
            f"${{{name}:-/ruta/por/defecto}} en el YAML.")

    return _VAR_RE.sub(_sub, obj)


def load_config(path):
    """Carga un YAML del proyecto y expande sus variables de entorno."""
    with open(path, "r", encoding="utf-8") as f:
        return expand_env(yaml.safe_load(f))
