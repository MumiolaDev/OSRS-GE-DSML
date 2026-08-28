"""
configuracion.py — configuración local persistida en un archivo JSON
(config.json en la raíz del proyecto, gitignorado — nunca credenciales
commiteadas), para que la app de escritorio pueda editar el token de
Telegram y la ruta de la DB desde una UI en vez de pedirle al usuario que
setee variables de entorno del sistema operativo a mano.

Prioridad de resolución (ver obtener()): variable de entorno del proceso >
config.json > None. Las variables de entorno ganan a propósito — un
despliegue en modo servidor/NSSM que ya las tiene seteadas
(docs/despliegue_24_7.md) no debe empezar a leer un config.json que capaz
ni existe en esa máquina; config.json es la ruta pensada para el uso
desktop de un usuario sin conocimientos técnicos.
"""
import json
import os

CONFIG_PATH = 'config.json'

# clave de config.json -> variable de entorno equivalente (None si no hay
# una hoy, como db_path).
CLAVES = {
    'telegram_bot_token': 'TELEGRAM_BOT_TOKEN',
    'telegram_chat_id': 'TELEGRAM_CHAT_ID',
    'db_path': None,
}


def cargar_configuracion(config_path=CONFIG_PATH):
    """dict con lo que haya en config.json, o {} si no existe o está mal
    formado — un archivo corrupto no debe romper el arranque de la app."""
    if not os.path.exists(config_path):
        return {}
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def guardar_configuracion(valores, config_path=CONFIG_PATH):
    """Sobreescribe config.json con `valores` (dict) completo — el
    llamador (escritorio/paginas/pagina_configuracion.py) debe leer con
    cargar_configuracion(), mezclar los campos que cambian, y volver a
    guardar el dict completo (no hace merge acá)."""
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(valores, f, indent=2, ensure_ascii=False)


def obtener(clave, config_path=CONFIG_PATH):
    """Resuelve `clave` (una de CLAVES) con la prioridad: variable de
    entorno del proceso > config.json > None."""
    var_entorno = CLAVES.get(clave)
    if var_entorno:
        valor = os.environ.get(var_entorno)
        if valor:
            return valor
    return cargar_configuracion(config_path).get(clave) or None
