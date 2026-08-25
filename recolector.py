import os
import sqlite3
import schedule
import time
import logging
from datetime import datetime
from osrs_ge_api import OSRSGeAPI
from base_de_datos import OSRSBaseDatos
from metricas import calcular_resumen_todos
from entrenador import entrenar_modelo_global
from mantenimiento import archivar_datos_antiguos


# Configurar logging: a archivo además de consola. Necesario para poder
# diagnosticar caídas cuando esto corre como servicio de fondo (sin consola
# visible) — ver docs/despliegue_24_7.md.
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s |--| %(levelname)s |--| %(message)s',
    handlers=[
        logging.FileHandler('logs/recolector.log', encoding='utf-8'),
        logging.StreamHandler(),
    ],
)

# Lista de ítems a monitorear. Vacía = sin filtro, se recolectan TODOS los
# ítems que devuelve la API (uso normal). Para pruebas rápidas o acotadas,
# poner acá una lista de item_ids, ej: [377, 440, 563, 564, 561]
ITEM_IDS = []
api = OSRSGeAPI()

def collect(table, func, interval_name ,db, timestamp = None):
    """Función genérica para recolectar datos."""

    try:
        print(f"Recolectando datos {interval_name}")
        df = func(timestamp=timestamp)  # obtener prices ( id, intervalo)

        if df.empty:
            logging.warning(f"No se obtuvieron datos para {interval_name}")
            return

        # La API entrega el snapshot completo (todos los ítems del juego).
        # Si ITEM_IDS no está vacío, filtramos a solo esos ítems.
        if ITEM_IDS:
            df = df[df['item_id'].isin(ITEM_IDS)]

            if df.empty:
                logging.warning(f"Ninguno de los ITEM_IDS monitoreados tenía datos para {interval_name}")
                return

        OSRSBaseDatos.insertar_precios(db, table, df)

        logging.info(f"Insertados {len(df)} registros en {table}")

        return df

    except Exception as e:
        logging.error(f"Error en collect_{interval_name}: {e}")

def collect_5min(db, timestamp=None):
    return collect('precios_5m', api.get_historical_5min, '5m', db, timestamp=timestamp)

def collect_1h(db, timestamp=None):
    return collect('precios_1h', api.get_historical_1h, '1h',db, timestamp=timestamp)

def collect_6h(db, timestamp=None):
    return collect('precios_6h', api.get_historical_6h, '6h', db, timestamp=timestamp)


INTERVALS = {
    '5m': (collect_5min, 5 * 60),
    '1h': (collect_1h, 60 * 60),
    '6h': (collect_6h, 6 * 60 * 60),
}

TABLA_POR_INTERVALO = {'5m': 'precios_5m', '1h': 'precios_1h', '6h': 'precios_6h'}


def job_diario(db):
    """
    Refresca el screener y reentrena el modelo. Se llama "diario" por lo que
    hace (mismo rol que un job de reentrenamiento diario en producción), pero
    durante la fase de pruebas se programa cada hora (ver `__main__` más
    abajo) para iterar más rápido mientras se acumula historial — al pasar a
    producción, volver a una cadencia diaria. Si el refresh del resumen
    falla, se salta el reentrenamiento: obtener_top_items_liquidez() depende
    de que resumen_actual esté fresca, así que reentrenar con una tabla
    vieja/vacía no tiene sentido.
    """
    try:
        logging.info("Job diario: refrescando resumen_actual...")
        resumen = calcular_resumen_todos(db)
        n = db.guardar_resumen(resumen)
        logging.info(f"resumen_actual actualizado: {n} ítems")
    except Exception as e:
        logging.error(f"Error refrescando resumen_actual, se omite el reentrenamiento: {e}")
        return

    try:
        logging.info("Job diario: reentrenando modelo...")
        entrenar_modelo_global(db)
    except Exception as e:
        logging.error(f"Error reentrenando el modelo: {e}")


def job_semanal(db):
    """Archiva a resolución diaria y purga los datos horarios fuera de la
    ventana de retención (ver mantenimiento.py)."""
    try:
        logging.info("Job semanal: archivando datos antiguos...")
        archivar_datos_antiguos(db)
    except Exception as e:
        logging.error(f"Error archivando datos antiguos: {e}")


def backfill(interval, start_ts, end_ts, db, delay=1.0):
    """
    Descarga datos históricos para un rango de tiempo, un snapshot por cada
    paso del intervalo (ej. cada hora para '1h'), respetando `delay` segundos
    de pausa entre requests para no saturar la API.

    interval: '5m', '1h' o '6h'
    start_ts / end_ts: timestamps unix (segundos), alineados al intervalo
    """
    collect_func, step = INTERVALS[interval]

    ts = start_ts
    n_calls = 0
    total_filas = 0
    while ts <= end_ts:
        df = collect_func(db, timestamp=ts)
        n_calls += 1
        total_filas += len(df) if df is not None else 0

        if n_calls % 24 == 0:
            logging.info(
                f"Backfill {interval}: {n_calls} llamadas, {total_filas} filas "
                f"acumuladas (última ts={ts}, {datetime.utcfromtimestamp(ts)} UTC)"
            )

        ts += step
        time.sleep(delay)

    logging.info(f"Backfill {interval} completo: {n_calls} llamadas, {total_filas} filas totales")
    return n_calls, total_filas


def backfill_faltantes(interval, start_ts, end_ts, db, delay=1.0, on_progreso=None):
    """
    Como `backfill()`, pero solo pide a la API los timestamps que la tabla
    todavía no tiene — pensado para rellenar huecos de recolección
    intermitente sin volver a pedir lo que ya está guardado (INSERT OR
    IGNORE ya lo protegía de duplicar, pero re-pedir miles de timestamps
    existentes desperdicia llamadas a una API pública sin necesidad).

    on_progreso: callback opcional, se llama cada ~24 requests (útil para
    ej. disparar un reentrenamiento periódico mientras se rellenan huecos
    largos, sin acoplar esta función a entrenador.py).
    """
    collect_func, step = INTERVALS[interval]
    tabla = TABLA_POR_INTERVALO[interval]

    conn = sqlite3.connect(db.db_path)
    c = conn.cursor()
    c.execute(f'SELECT DISTINCT timestamp FROM {tabla} WHERE timestamp BETWEEN ? AND ?', (start_ts, end_ts))
    existentes = {row[0] for row in c.fetchall()}
    conn.close()

    faltantes = [ts for ts in range(start_ts, end_ts + 1, step) if ts not in existentes]
    logging.info(f"Backfill de faltantes {interval}: {len(faltantes)} timestamps a descargar")

    n_calls = 0
    total_filas = 0
    for ts in faltantes:
        df = collect_func(db, timestamp=ts)
        n_calls += 1
        total_filas += len(df) if df is not None else 0

        if n_calls % 24 == 0:
            logging.info(
                f"Backfill de faltantes {interval}: {n_calls}/{len(faltantes)} llamadas, "
                f"{total_filas} filas acumuladas"
            )
            if on_progreso:
                on_progreso()

        time.sleep(delay)

    logging.info(f"Backfill de faltantes {interval} completo: {n_calls} llamadas, {total_filas} filas totales")
    return n_calls, total_filas


def rellenar_huecos_al_inicio(db, intervalos=('1h', '5m', '6h'), delay=1.0, retrain_cada_segundos=600):
    """
    Se corre una vez al arrancar el recolector: por cada intervalo en
    `intervalos`, mira desde cuándo hay datos en su tabla y rellena con
    `backfill_faltantes` todo lo que falte hasta ahora. Soluciona la
    intermitencia sola — antes había que acordarse de correr
    backfill_historico.py a mano después de cada corte.

    Si el hueco es grande (el recolector estuvo apagado varios días), va
    reentrenando el modelo cada `retrain_cada_segundos` mientras rellena,
    en vez de esperar al final — mismo mecanismo que usaba
    backfill_historico.py. Con huecos chicos (el caso normal de reiniciar
    el proceso) esto no llega a dispararse y el arranque es rápido.

    Si una tabla todavía no tiene ningún dato (item/intervalo nuevo, ej. la
    primera vez que se activó 6h) no hay huecos que rellenar — la
    recolección normal ya se encarga de sembrarla.

    El límite superior del rango se recorta un `step` hacia atrás (por
    intervalo) para no pedirle a la API el bucket todavía en curso: ese
    snapshot no está cerrado ni agregado del lado de la wiki todavía, así
    que siempre vuelve vacío (dispara "No se obtuvieron datos" sin que sea
    un error real). Ese último tramo se termina rellenando solo con la
    próxima recolección programada (5m/1h/6h), una vez cerrado.
    """
    ahora = int(time.time())
    ultimo_retrain = time.time()

    def tick():
        nonlocal ultimo_retrain
        if time.time() - ultimo_retrain >= retrain_cada_segundos:
            logging.info("Relleno de huecos en curso: reentrenando con lo descargado hasta ahora...")
            try:
                job_diario(db)
            except Exception as e:
                logging.error(f"Error en reentrenamiento intermedio durante el relleno de huecos: {e}")
            ultimo_retrain = time.time()

    for interval in intervalos:
        tabla = TABLA_POR_INTERVALO[interval]
        conn = sqlite3.connect(db.db_path)
        c = conn.cursor()
        c.execute(f'SELECT MIN(timestamp) FROM {tabla}')
        inicio = c.fetchone()[0]
        conn.close()

        if inicio is None:
            logging.info(f"{tabla}: sin datos todavía, nada que rellenar.")
            continue

        _, step = INTERVALS[interval]
        limite = ahora - step

        if limite < inicio:
            logging.info(f"{tabla}: sin huecos cerrados que rellenar todavía.")
            continue

        inicio_legible = datetime.fromtimestamp(inicio).strftime('%Y-%m-%d %H:%M:%S')
        limite_legible = datetime.fromtimestamp(limite).strftime('%Y-%m-%d %H:%M:%S')
        logging.info(f"=== Relleno de huecos al iniciar: {interval} desde {inicio_legible} hasta {limite_legible} ===")
        try:
            backfill_faltantes(interval, inicio, limite, db, delay=delay, on_progreso=tick)
        except Exception as e:
            logging.error(f"Error rellenando huecos de {interval}: {e}")


if __name__ == "__main__":
    db = OSRSBaseDatos('data/osrs_ge.db')
    logging.info("Iniciando recolector...")

    # Catálogo de ítems (id -> nombre, members, buy_limit). Se actualiza cada
    # vez que arranca el recolector; los datos cambian muy rara vez.
    mapping = api.get_item_mapping()
    n_items = db.guardar_items(mapping)
    logging.info(f"Catálogo de ítems actualizado: {n_items} ítems")

    # Ejecutar inmediatamente al arrancar
    data_5m = collect_5min(db)
    data_1h = collect_1h(db)
    data_6h = collect_6h(db)

    # Rellenar huecos dejados por cortes anteriores antes de entrar al loop
    # — así el recolector se pone al día solo en cada arranque.
    rellenar_huecos_al_inicio(db)

    # Programar tareas
    schedule.every(5).minutes.do(collect_5min, db)
    schedule.every(1).hour.do(collect_1h, db)
    schedule.every(6).hours.do(collect_6h, db)
    # Fase de pruebas: job_diario cada hora (no una vez al día) para iterar
    # más rápido — volver a schedule.every().day.at("03:00") en producción.
    schedule.every(1).hour.do(job_diario, db)
    schedule.every().sunday.at("04:00").do(job_semanal, db)
    while True:

        schedule.run_pending()
        time.sleep(1)

