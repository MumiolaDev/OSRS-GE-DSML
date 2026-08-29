import os
import sqlite3
import schedule
import time
import logging
from datetime import datetime
from osrs_ge_api import OSRSGeAPI
from base_de_datos import OSRSBaseDatos
from metricas import calcular_resumen_todos
from entrenador import entrenar_desde_config
from mantenimiento import ejecutar_mantenimiento_semanal


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


def _programar_cada_n_minutos_alineado(minutos, func, db):
    """
    Registra `func` para correr en cada múltiplo de `minutos` dentro de la
    hora (:00, :05, ..., :55 si minutos=5), alineado al reloj de pared — a
    diferencia de `schedule.every(N).minutes`, que es relativo al momento en
    que arrancó el proceso (si el recolector arranca a las 10:03, dispararía
    a las 10:08, 10:13...). La librería `schedule` no tiene una primitiva
    nativa de "cada N minutos alineado al reloj"; se logra registrando un
    job por cada minuto múltiplo de N vía `every().hour.at(':MM')`, que sí
    es una hora de reloj absoluta.
    """
    for m in range(0, 60, minutos):
        schedule.every().hour.at(f":{m:02d}", "UTC").do(func, db)


INTERVALS = {
    '5m': (collect_5min, 5 * 60),
    '1h': (collect_1h, 60 * 60),
    '6h': (collect_6h, 6 * 60 * 60),
}

TABLA_POR_INTERVALO = {'5m': 'precios_5m', '1h': 'precios_1h', '6h': 'precios_6h'}


def _entrenar_desde_config(db, cfg, guardar_en_disco=True, calcular_metricas_horizonte=False):
    """
    Reentrena (en vivo) un modelo de modelos_config — regresor o
    clasificador, sobre su tabla/ventana/universo de ítems tal cual estén
    configurados (ver entrenador._kwargs_desde_modelo_config, que interpreta
    modo_seleccion='manual'/'liquidez') — es lo que le permite a
    job_horario/job_diario reentrenar tanto los 3 modelos productivos
    sembrados en la migración como cualquier modelo que el usuario defina
    después desde la app de escritorio, sin distinguir entre unos y otros.

    Actualiza modelos_config.ultimo_entrenamiento_ts SOLO si el
    entrenamiento efectivamente corrió (no en un corte temprano por falta
    de ítems/historial suficiente) — así la app de escritorio no muestra
    "entrenado hace un momento" sobre un modelo que en realidad nunca
    llegó a entrenar. No se hace dentro de un try/except acá porque el
    llamador (job_horario/job_diario) ya envuelve cada llamada a esta
    función en el suyo propio.

    Devuelve True/False según si entrenó de verdad — lo usa
    escritorio/paginas/pagina_modelos.py para el botón "Entrenar ahora".
    """
    overrides = dict(guardar_en_disco=guardar_en_disco)
    if cfg['tipo'] != 'clasificador':
        overrides['calcular_metricas_horizonte'] = calcular_metricas_horizonte

    resultado = entrenar_desde_config(db, cfg, **overrides)
    exito = (resultado[1] is not None) if cfg['tipo'] == 'clasificador' else bool(resultado)

    if exito:
        db.actualizar_modelo_config(cfg['model_id'], ultimo_entrenamiento_ts=int(time.time()))
    return exito


def job_horario(db):
    """
    Corre cada hora (:05, unos minutos después de collect_1h para no competir
    por I/O/CPU con la recolección): refresca resumen_actual y reentrena
    todos los modelos de modelos_config con cadencia='horaria' y
    estado='activo' (ver base_de_datos.py y _entrenar_desde_config) — antes
    eran 2 llamadas hardcodeadas ('global_horario' y 'f2p10_100gp_clasif',
    la variante productiva del clasificador direccional: 10 ítems F2P más
    líquidos con precio_minimo=100 y Steel bar excluido, item_id 2353 — la
    configuración que en el backtest walk-forward de 90 días convirtió una
    estrategia perdedora, comprar a ciegas, -67.6M gp, en ganadora, +7.3M
    gp), ahora conviven en la misma tabla con cualquier modelo que el
    usuario defina desde la app de escritorio con esa misma cadencia. Si el
    refresh del resumen falla, se salta el reentrenamiento entero:
    obtener_top_items_liquidez() (la usan los modelos con
    modo_seleccion='liquidez') depende de que resumen_actual esté fresca.
    Cada modelo se reentrena en su propio try/except, para que un fallo en
    uno no tumbe a los demás ni a las alertas.
    """
    try:
        logging.info("Job horario: refrescando resumen_actual...")
        resumen = calcular_resumen_todos(db)
        n = db.guardar_resumen(resumen)
        logging.info(f"resumen_actual actualizado: {n} ítems")
    except Exception as e:
        logging.error(f"Error refrescando resumen_actual, se omite el reentrenamiento: {e}")
        return

    for cfg in db.listar_modelos_config(cadencia='horaria', estado='activo'):
        try:
            logging.info(f"Job horario: reentrenando {cfg['model_id']} ({cfg['tipo']})...")
            _entrenar_desde_config(db, cfg, guardar_en_disco=True)
        except Exception as e:
            logging.error(f"Error reentrenando {cfg['model_id']}: {e}")

    try:
        from alertas import evaluar_alertas
        evaluar_alertas(db)
    except Exception as e:
        logging.error(f"Error evaluando alertas: {e}")


def job_diario(db):
    """
    Corre una vez al día a las 03:00: reentrena todos los modelos de
    modelos_config con cadencia='diaria' y estado='activo' (antes,
    solo 'global_diario' hardcodeado) y calcula métricas de horizonte
    (2..6 pasos, evaluacion.py) para los de tipo='regresor' — el
    clasificador no tiene esa noción, ver entrenador.py — que son caras y
    por eso no se calculan en cada corrida horaria. Reusa resumen_actual,
    ya refrescada por job_horario en la misma hora — no la recalcula de
    nuevo.
    """
    for cfg in db.listar_modelos_config(cadencia='diaria', estado='activo'):
        try:
            logging.info(f"Job diario: reentrenando {cfg['model_id']} ({cfg['tipo']})...")
            _entrenar_desde_config(
                db, cfg, guardar_en_disco=True,
                calcular_metricas_horizonte=(cfg['tipo'] == 'regresor'),
            )
        except Exception as e:
            logging.error(f"Error reentrenando {cfg['model_id']}: {e}")


def job_semanal(db):
    """
    Archiva/purga/podda todas las tablas con retención y libera espacio
    liberado con incremental_vacuum (ver mantenimiento.ejecutar_mantenimiento_semanal).
    Registra la corrida en mantenimiento_estado — es lo que le permite a
    verificar_catchup_semanal() detectar si esto no corrió en su ventana
    programada (domingo 04:00 UTC) porque el proceso estuvo apagado justo
    entonces, y disparar un catch-up al arrancar en vez de esperar hasta el
    domingo siguiente.
    """
    try:
        logging.info("Job semanal: mantenimiento (retención, purga, vacuum)...")
        ejecutar_mantenimiento_semanal(db)
        db.registrar_corrida('job_semanal', int(time.time()))
    except Exception as e:
        logging.error(f"Error en el mantenimiento semanal: {e}")


def verificar_catchup_semanal(db, umbral_dias=8):
    """
    Si job_semanal no corrió en los últimos `umbral_dias` (más que una
    semana, con margen), lo corre ahora mismo en vez de esperar al próximo
    domingo 04:00 UTC — cubre el caso de que el proceso haya estado
    apagado justo en esa ventana. Se llama al arrancar, junto con
    rellenar_huecos_al_inicio().
    """
    ultima = db.obtener_ultima_corrida('job_semanal')
    ahora = int(time.time())
    if ultima is None or ahora - ultima > umbral_dias * 86400:
        logging.info("job_semanal: ventana perdida (o primera corrida) — haciendo catch-up ahora...")
        job_semanal(db)


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


def backfill_faltantes(interval, start_ts, end_ts, db, delay=1.0, on_progreso=None, debe_detener=None):
    """
    Como `backfill()`, pero solo pide a la API los timestamps que la tabla
    todavía no tiene — pensado para rellenar huecos de recolección
    intermitente sin volver a pedir lo que ya está guardado (INSERT OR
    IGNORE ya lo protegía de duplicar, pero re-pedir miles de timestamps
    existentes desperdicia llamadas a una API pública sin necesidad).

    on_progreso (opcional): callback `f(interval, n_calls, total)` llamado
    en cada iteración del loop — sin dependencia de Qt ni de ningún otro
    framework, solo una función plana. Es lo que le permite a
    escritorio/hilo_recolector.py mostrar una barra de progreso real
    (número de descargas hechas sobre el total) en vez de un simple
    "cargando" indeterminado; sin callback (default None) el
    comportamiento es idéntico al de siempre.

    debe_detener (opcional): callback `f() -> bool`, chequeado antes de
    cada request — si devuelve True, corta el loop ahí mismo (con lo ya
    descargado hasta ese punto) en vez de terminar todo `faltantes`. Sin
    esto, pedir que el recolector se detenga mientras corre un backfill
    largo no tenía ningún efecto hasta que ese backfill terminaba solo —
    ver escritorio/hilo_recolector.py, que pasa `lambda: self._detener`.

    Devuelve (n_calls, total_filas, faltantes) — `faltantes` es la lista de
    timestamps que efectivamente hacía falta pedir, para que el llamador
    (rellenar_huecos_al_inicio) pueda agruparla en rangos contiguos y
    disparar el replay histórico (replay_historico.py) solo sobre los
    tramos que realmente tenían un hueco.
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
    total = len(faltantes)
    for ts in faltantes:
        if debe_detener is not None and debe_detener():
            logging.info(f"Backfill de faltantes {interval}: interrumpido por pedido externo ({n_calls}/{total})")
            break

        df = collect_func(db, timestamp=ts)
        n_calls += 1
        total_filas += len(df) if df is not None else 0

        if on_progreso is not None:
            on_progreso(interval, n_calls, total)

        if n_calls % 24 == 0:
            logging.info(
                f"Backfill de faltantes {interval}: {n_calls}/{total} llamadas, "
                f"{total_filas} filas acumuladas"
            )

        time.sleep(delay)

    logging.info(f"Backfill de faltantes {interval} completo: {n_calls} llamadas, {total_filas} filas totales")
    return n_calls, total_filas, faltantes


def _agrupar_en_rangos(timestamps, step):
    """
    Agrupa una lista de timestamps (no necesariamente ordenada) en rangos
    contiguos (inicio, fin) según `step` — ej. [100, 105, 110, 200, 205] con
    step=5 da [(100, 110), (200, 205)]. Usado para acotar el replay
    histórico (replay_historico.ejecutar_replay) exactamente a los tramos
    que tenían un hueco real, sin re-simular checkpoints en tramos donde ya
    había cobertura en vivo.
    """
    if not timestamps:
        return []
    ts_ordenados = sorted(timestamps)
    rangos = []
    inicio = fin = ts_ordenados[0]
    for ts in ts_ordenados[1:]:
        if ts == fin + step:
            fin = ts
        else:
            rangos.append((inicio, fin))
            inicio = fin = ts
    rangos.append((inicio, fin))
    return rangos


def rellenar_huecos_al_inicio(db, intervalos=('1h', '5m', '6h'), delay=1.0, on_progreso=None, debe_detener=None):
    """
    on_progreso (opcional): callback `f(fase, actual, total)` — se pasa
    tal cual a backfill_faltantes() (fase = el intervalo, ej. '1h') y a
    replay_historico.ejecutar_replay() (fase = 'replay') para que
    escritorio/hilo_recolector.py pueda mostrar una barra de progreso real
    durante el relleno de huecos y el replay que puede disparar. Ver el
    docstring de backfill_faltantes.

    debe_detener (opcional): callback `f() -> bool`, se pasa tal cual a
    backfill_faltantes()/ejecutar_replay() (corta cada uno a mitad de
    camino) y además se chequea entre intervalos y entre rangos de replay,
    para no arrancar un tramo nuevo si ya se pidió parar.

    Se corre una vez al arrancar el recolector: por cada intervalo en
    `intervalos`, mira desde cuándo hay datos en su tabla y rellena con
    `backfill_faltantes` todo lo que falte hasta ahora. Soluciona la
    intermitencia sola — antes había que acordarse de correr
    backfill_historico.py a mano después de cada corte.

    Solo `precios_1h` alimenta el modelo/screener (build_training_set,
    calcular_resumen_todos), así que solo para ese intervalo, después de
    rellenar los datos crudos, se agrupan los huecos detectados en rangos
    contiguos (_agrupar_en_rangos) y se dispara replay_historico.ejecutar_replay
    por cada uno — reentrena en los momentos exactos en que job_horario/
    job_diario habrían corrido durante ese hueco, en vez de esperar a un
    solo reentrenamiento final con todo el historial (ver
    replay_historico.py para el porqué). Para 5m/6h no hace falta: no
    alimentan ningún entrenamiento.

    Si una tabla todavía no tiene ningún dato (item/intervalo nuevo, ej. la
    primera vez que se activó 6h) no hay huecos que rellenar — la
    recolección normal ya se encarga de sembrarla.

    El límite superior del rango se recorta un `step` hacia atrás (por
    intervalo) para no pedirle a la API el bucket todavía en curso: ese
    snapshot no está cerrado ni agregado del lado de la wiki todavía, así
    que siempre vuelve vacío (dispara "No se obtuvieron datos" sin que sea
    un error real). Ese último tramo se termina rellenando solo con la
    próxima recolección programada (5m/1h/6h), una vez cerrado.

    El límite inferior del rango se acota a `mantenimiento.RETENCION_DIAS`
    de cada tabla, no a `MIN(timestamp)` — sin esto, un hueco viejo (ej. la
    tabla tiene algún dato suelto de hace más de un año) hace que esto
    intente rellenar/replayear meses de historia que `mantenimiento.py` va
    a purgar en la próxima corrida semanal de todos modos. No vale la pena
    gastar horas de replay walk-forward (ver replay_historico.py) sobre
    datos que no se van a conservar.

    `ahora = int(time.time())` no cae en un múltiplo exacto del `step` del
    intervalo, y la API devuelve 400 Bad Request si se le pide un timestamp
    no alineado (ver el gotcha documentado en CLAUDE.md/osrs_ge_api.py) —
    por eso `limite_retencion` y `limite` se calculan a partir de `ahora`
    ya alineado hacia abajo al `step` de cada intervalo, no de `ahora`
    crudo. `RETENCION_DIAS[tabla] * 86400` y `step` son ambos múltiplos del
    `step`, así que alinear una sola vez alcanza para que toda la
    aritmética de más abajo quede alineada también.
    """
    from mantenimiento import RETENCION_DIAS
    from replay_historico import ejecutar_replay

    ahora = int(time.time())

    for interval in intervalos:
        if debe_detener is not None and debe_detener():
            logging.info("rellenar_huecos_al_inicio: interrumpido por pedido externo")
            break

        tabla = TABLA_POR_INTERVALO[interval]
        _, step = INTERVALS[interval]
        ahora_alineado = ahora - (ahora % step)

        conn = sqlite3.connect(db.db_path)
        c = conn.cursor()
        c.execute(f'SELECT MIN(timestamp) FROM {tabla}')
        inicio = c.fetchone()[0]
        conn.close()

        if inicio is None:
            logging.info(f"{tabla}: sin datos todavía, nada que rellenar.")
            continue

        limite_retencion = ahora_alineado - RETENCION_DIAS[tabla] * 86400
        if inicio < limite_retencion:
            logging.info(
                f"{tabla}: MIN(timestamp) es más viejo que la retención "
                f"({RETENCION_DIAS[tabla]}d) — no se rellena esa parte, se va a purgar sola."
            )
            inicio = limite_retencion

        limite = ahora_alineado - step

        if limite < inicio:
            logging.info(f"{tabla}: sin huecos cerrados que rellenar todavía.")
            continue

        inicio_legible = datetime.fromtimestamp(inicio).strftime('%Y-%m-%d %H:%M:%S')
        limite_legible = datetime.fromtimestamp(limite).strftime('%Y-%m-%d %H:%M:%S')
        logging.info(f"=== Relleno de huecos al iniciar: {interval} desde {inicio_legible} hasta {limite_legible} ===")
        try:
            _, _, faltantes = backfill_faltantes(
                interval, inicio, limite, db, delay=delay, on_progreso=on_progreso, debe_detener=debe_detener,
            )
        except Exception as e:
            logging.error(f"Error rellenando huecos de {interval}: {e}")
            continue

        if interval != '1h' or not faltantes:
            continue

        rangos = _agrupar_en_rangos(faltantes, step)
        logging.info(f"Replay histórico: {len(rangos)} rango(s) de hueco detectado(s) en precios_1h")
        for inicio_rango, fin_rango in rangos:
            if debe_detener is not None and debe_detener():
                logging.info("rellenar_huecos_al_inicio: replay interrumpido por pedido externo")
                break
            try:
                ejecutar_replay(db, inicio_rango, fin_rango, on_progreso=on_progreso, debe_detener=debe_detener)
            except Exception as e:
                logging.error(f"Error en replay histórico del rango {inicio_rango}-{fin_rango}: {e}")


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
    # — así el recolector se pone al día solo en cada arranque. Incluye el
    # replay histórico si había huecos en precios_1h (ver
    # rellenar_huecos_al_inicio/replay_historico.py).
    rellenar_huecos_al_inicio(db)

    # Si job_semanal no corrió en su ventana programada (proceso apagado
    # justo el domingo 04:00 UTC), hacer catch-up ahora en vez de esperar
    # hasta el domingo siguiente.
    verificar_catchup_semanal(db)

    # Catch-up en vivo: refresca resumen_actual y el modelo productivo ahora
    # mismo, en vez de esperar al próximo :05 programado — importante sobre
    # todo después de un replay largo, para no dejar el dashboard/las
    # alertas con datos desactualizados hasta la próxima hora en punto.
    job_horario(db)

    # Programar tareas, alineadas al reloj de pared (:00/:05/:10... en vez de
    # relativas a cuándo arrancó este proceso) — ver _programar_cada_n_minutos_alineado.
    _programar_cada_n_minutos_alineado(5, collect_5min, db)
    schedule.every().hour.at(":00", "UTC").do(collect_1h, db)
    for h in (0, 6, 12, 18):
        schedule.every().day.at(f"{h:02d}:00", "UTC").do(collect_6h, db)
    # job_horario a :05, cinco minutos después de collect_1h, para no competir
    # por I/O/CPU con la recolección que acaba de correr en el mismo minuto.
    schedule.every().hour.at(":05", "UTC").do(job_horario, db)
    schedule.every().day.at("03:00", "UTC").do(job_diario, db)
    schedule.every().sunday.at("04:00", "UTC").do(job_semanal, db)
    while True:

        schedule.run_pending()
        time.sleep(1)

