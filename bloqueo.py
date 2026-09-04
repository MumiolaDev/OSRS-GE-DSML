"""
bloqueo.py — candado de instancia única para el recolector.

Con el recolector corriendo como servicio de systemd (ver
docs/despliegue_24_7.md) y la app de escritorio pudiendo arrancar su propio
hilo de recolección (escritorio/hilo_recolector.py), es fácil terminar con
DOS recolectores sobre la misma DB: el servicio y la app. No corrompe nada
—SQLite en WAL con busy_timeout aguanta los dos escritores— pero duplica
las requests a la API, duplica el reentrenamiento (dos procesos fiteando
XGBoost sobre los mismos ítems al mismo tiempo, pisándose el .pkl) y hace
que el relleno de huecos al arrancar se ejecute dos veces en paralelo.

La solución es un candado `flock` sobre un archivo al lado de la DB. Se usa
flock y no un archivo de PID porque **el kernel libera el candado solo
cuando el proceso muere**, sea como sea que muera (kill -9, corte de luz,
crash): no existe el problema del candado rancio que hay que limpiar a
mano, ni la carrera de "leo el PID, chequeo si vive, pero entre las dos
cosas el PID se reusó".

El archivo guarda además un JSON con quién lo tiene (pid, origen, desde
cuándo) para que la app pueda mostrar "el recolector ya corre como
servicio" en vez de un error opaco — el contenido es informativo, la
exclusión mutua la da el flock, no el contenido.
"""
import fcntl
import json
import logging
import os
import time

# origen: quién tomó el candado, para poder decirlo en la UI.
ORIGEN_SERVICIO = 'servicio'      # recolector.py corriendo solo (systemd/consola)
ORIGEN_APP = 'app'                # hilo del recolector dentro de la app de escritorio


def ruta_bloqueo(db_path):
    """El candado vive al lado de la DB y sigue a la ruta configurada: dos
    DBs distintas (ej. una de pruebas) son dos recolectores legítimamente
    simultáneos, no un conflicto."""
    return str(db_path) + '.lock'


class BloqueoRecolector:
    """
    Uso normal:

        bloqueo = BloqueoRecolector('data/osrs_ge.db', origen=ORIGEN_APP)
        if not bloqueo.adquirir():
            ...  # bloqueo.duenio_previo tiene el JSON de quien lo tiene
        try:
            ...
        finally:
            bloqueo.liberar()

    También sirve como context manager, pero ahí un candado ya tomado
    levanta RuntimeError en vez de devolver False.
    """

    def __init__(self, db_path='data/osrs_ge.db', origen=ORIGEN_SERVICIO):
        self.db_path = db_path
        self.origen = origen
        self.path = ruta_bloqueo(db_path)
        self._fd = None
        self.duenio_previo = None   # dict con la info del que lo tiene, si adquirir() falló

    def adquirir(self):
        """True si quedó tomado; False si ya lo tiene otro proceso (y en ese
        caso deja en self.duenio_previo lo que se sepa de él)."""
        directorio = os.path.dirname(self.path)
        if directorio:
            os.makedirs(directorio, exist_ok=True)
        # 'a+' y no 'w': abrir en modo escritura truncaría el archivo ANTES
        # de saber si el candado está libre, borrándole la info de dueño al
        # proceso que sí lo tiene.
        fd = open(self.path, 'a+', encoding='utf-8')
        try:
            fcntl.flock(fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            fd.close()
            self.duenio_previo = leer_info_bloqueo(self.path)
            return False

        self._fd = fd
        fd.seek(0)
        fd.truncate()
        json.dump({
            'pid': os.getpid(),
            'origen': self.origen,
            'inicio': int(time.time()),
            'db_path': self.db_path,
        }, fd)
        fd.flush()
        return True

    def liberar(self):
        if self._fd is None:
            return
        try:
            # Vaciar el contenido antes de soltar: si queda el JSON viejo, un
            # lector que mire el archivo justo después ve un dueño que ya no
            # existe. El flock se suelta al cerrar el descriptor.
            self._fd.seek(0)
            self._fd.truncate()
            self._fd.flush()
            fcntl.flock(self._fd.fileno(), fcntl.LOCK_UN)
        except OSError as e:
            logging.warning(f"bloqueo.py: no se pudo liberar limpiamente el candado: {e}")
        finally:
            self._fd.close()
            self._fd = None

    def __enter__(self):
        if not self.adquirir():
            raise RuntimeError(f"Ya hay un recolector corriendo: {self.duenio_previo}")
        return self

    def __exit__(self, *exc):
        self.liberar()
        return False


def leer_info_bloqueo(path):
    """El JSON que dejó el dueño del candado, o None si no se puede leer
    (archivo inexistente, vacío, o a medio escribir)."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            contenido = f.read().strip()
    except OSError:
        return None
    if not contenido:
        return None
    try:
        return json.loads(contenido)
    except json.JSONDecodeError:
        return None


def hay_recolector_corriendo(db_path='data/osrs_ge.db'):
    """
    Sin tomar el candado: dict con la info del recolector que esté
    corriendo sobre esa DB, o None si no hay ninguno.

    Lo usa la app de escritorio para no ofrecer "Iniciar recolector" cuando
    el servicio de systemd ya lo tiene andando, y estado.py para el
    resumen de una línea. Chequea el flock de verdad (no si el archivo
    existe): un candado que quedó de un proceso muerto ya está libre para
    el kernel y acá tiene que leerse como "no hay nadie".
    """
    path = ruta_bloqueo(db_path)
    if not os.path.exists(path):
        return None
    try:
        fd = open(path, 'a+', encoding='utf-8')
    except OSError:
        return None
    try:
        fcntl.flock(fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        # No se pudo tomar => lo tiene alguien => hay recolector corriendo.
        info = leer_info_bloqueo(path)
        fd.close()
        return info or {'pid': None, 'origen': 'desconocido', 'inicio': None}
    # Se pudo tomar => estaba libre => no hay recolector. Soltarlo enseguida.
    fcntl.flock(fd.fileno(), fcntl.LOCK_UN)
    fd.close()
    return None
