"""
mantenimiento.py — retención, downsampling y vacuum de la base de datos.

Sin retención, las tablas de series de tiempo crecen sin límite: solo
precios_1h a resolución horaria son ~4.600 ítems x 24 filas/día ≈ 110.000
filas/día en el peor caso (~40M filas/año) — y precios_5m, 12x más denso,
no tenía NINGUNA retención hasta este archivo. Cada consulta (screener,
entrenamiento) se vuelve progresivamente más lenta a medida que estas
tablas crecen sin poda.

La política, por tabla:
- precios_1h: los datos horarios más viejos que `dias_retencion` (90 días
  default — recortado de 365, ver RETENCION_DIAS) se agregan a resolución
  diaria en `precios_1h_diario` (conserva la tendencia de largo plazo) y
  después se purgan — archivar_datos_antiguos().
- precios_5m/precios_6h: purga directa sin downsampling
  (purgar_datos_antiguos()) — ni el entrenamiento ni el screener trabajan
  sobre estas tablas (los dos usan precios_1h), así que no vale la pena
  mantener un agregado intermedio. precios_5m tiene un consumidor real —
  medir cada cuánto se completan de verdad las dos puntas de un flip dentro
  de la hora — pero para eso alcanza con datos recientes de los ítems que se
  operan, no con el catálogo entero: por eso se recolecta acotada
  (recolector.INTERVALOS_ACOTADOS_A_MODELOS) y con ventana de 14 días.
  precios_6h con ventana aún más corta (7 días, ver RETENCION_DIAS) — todavía
  no se usa para nada más que recolección/monitoreo puntual.
- predicciones: purga por ventana de tiempo (180 días default) —
  INSERT OR REPLACE ya deduplica corridas repetidas sobre el mismo período,
  así que el crecimiento neto es proporcional al tiempo, no al número de
  corridas.
- model_metrics: se poda solo el detalle por ítem (podar_metricas_por_item,
  90 días default); el agregado (item_id IS NULL) se conserva indefinidamente
  porque alimenta la serie histórica de calidad del dashboard y son pocas
  filas por corrida.

Las agregaciones/purgas se hacen en SQL directo (no trayendo todo a pandas)
porque son potencialmente millones de filas.

Pensado para correr periódicamente (ej. semanal, ver
ejecutar_mantenimiento_semanal() y recolector.job_semanal) desde el loop de
recolector.py, no en cada arranque.
"""

import logging
import sqlite3
import time
from datetime import datetime, timezone

# Ventanas de retención por tabla de precios, en un solo lugar — tanto
# ejecutar_mantenimiento_semanal() (purga/agregación) como
# recolector.rellenar_huecos_al_inicio() (para no intentar rellenar/replayear
# huecos más viejos que lo que de todos modos se va a purgar la próxima
# semana) usan estos mismos números, para no tener la ventana de retención
# duplicada — y potencialmente desincronizada — en dos archivos.
RETENCION_DIAS = {
    'precios_1h': 90,
    'precios_6h': 7,
    # 14 días, recortado de 30: precios_5m dejó de guardar el catálogo entero
    # y ahora solo guarda los ítems de los modelos activos
    # (recolector.INTERVALOS_ACOTADOS_A_MODELOS), y su único consumidor es la
    # calibración de ejecución, que necesita datos RECIENTES — no una serie
    # larga. Con 30 días y todo el catálogo la tabla tendía a 634 MB.
    'precios_5m': 14,
}


def archivar_datos_antiguos(db, tabla='precios_1h', tabla_diaria='precios_1h_diario', dias_retencion=RETENCION_DIAS['precios_1h']):
    """
    Agrega a resolución diaria y purga de `tabla` todo lo anterior a
    `dias_retencion` días. Idempotente: correrlo dos veces seguidas sin
    datos nuevos que archivar no duplica nada (INSERT OR IGNORE en el
    agregado, y el DELETE de filas ya purgadas simplemente no encuentra
    nada la segunda vez).

    Nota: esto no recupera espacio en disco por sí solo (SQLite no achica el
    archivo con un DELETE). Correr `VACUUM` aparte y con poca frecuencia
    (ej. mensual) para eso — puede tardar y bloquea la DB mientras corre, no
    conviene meterlo en este job.
    """
    segundos_dia = 86400
    # El corte se alinea hacia ABAJO al inicio del día UTC: si se cortara en
    # un punto intermedio (ej. las 14:37 de hace 90 días), el día del borde
    # se archivaría con el promedio de sus primeras 14 horas y después se
    # borraría de precios_1h — y la corrida de la semana siguiente, que sí
    # vería el día completo, no podría corregirlo porque INSERT OR IGNORE
    # respeta la fila parcial ya escrita en (item_id, fecha). Así solo se
    # archivan días enteros, y el día del borde espera un día más.
    corte = ((int(time.time()) - dias_retencion * 86400) // segundos_dia) * segundos_dia

    conn = sqlite3.connect(db.db_path)
    c = conn.cursor()

    c.execute(
        f'''INSERT OR IGNORE INTO {tabla_diaria}
                (item_id, fecha, avg_high_price, avg_low_price, high_volume, low_volume)
            SELECT
                item_id,
                (timestamp / {segundos_dia}) * {segundos_dia} AS fecha,
                CAST(ROUND(AVG(avg_high_price)) AS INTEGER),
                CAST(ROUND(AVG(avg_low_price)) AS INTEGER),
                SUM(high_volume),
                SUM(low_volume)
            FROM {tabla}
            WHERE timestamp < ?
            GROUP BY item_id, fecha''',
        (corte,),
    )
    filas_archivadas = c.rowcount

    c.execute(f'DELETE FROM {tabla} WHERE timestamp < ?', (corte,))
    filas_purgadas = c.rowcount

    conn.commit()
    conn.close()

    logging.info(
        f"{tabla}: {filas_archivadas} filas diarias archivadas en {tabla_diaria}, "
        f"{filas_purgadas} filas horarias purgadas (corte={corte}, retención={dias_retencion}d)"
    )
    return filas_archivadas, filas_purgadas


TAREA_BACKFILL_ARCHIVO_DIARIO = 'backfill_archivo_diario'


def backfill_archivo_diario(db, desde_ts, hasta_ts, dias_por_lote=7, delay=1.0):
    """
    Rellena `precios_1h_diario` con historia REAL descargada de la API para
    [desde_ts, hasta_ts] — pensado para ir mucho más atrás que los 90 días
    de retención de `precios_1h` (ej. desde 2021-03-08, la fecha más vieja
    con datos reales en la API, ver el gotcha documentado en
    osrs_ge_api.py/CLAUDE.md), sin dejar la tabla operativa creciendo sin
    límite ni tocar la ventana de entrenamiento en vivo: `job_horario`
    sigue viendo siempre los últimos 90 días de `precios_1h`, esto no los
    toca.

    Cómo evita bloatear `precios_1h`: procesa en lotes de `dias_por_lote`
    días. Para cada lote, descarga los datos horarios reales con
    `recolector.backfill_faltantes()` a `precios_1h` (resumible en sí
    mismo: si se corta a mitad de un lote, la próxima corrida solo pide lo
    que falte de ESE lote, no lo redescarga entero), los agrega a
    resolución diaria en `precios_1h_diario` con la MISMA agregación que
    `archivar_datos_antiguos` (avg de precio redondeado, suma de volumen,
    `INSERT OR IGNORE` sobre la PK `(item_id, fecha)` — idempotente), y
    los borra de `precios_1h` antes de pasar al lote siguiente — así
    `precios_1h` nunca acumula más que un lote extra de datos temporales
    mientras esto corre, y vuelve a su tamaño normal (ventana de 90 días)
    apenas termina cada lote.

    Resumible ENTRE lotes vía `mantenimiento_estado`
    (`db.registrar_corrida`/`obtener_ultima_corrida`, tarea
    `TAREA_BACKFILL_ARCHIVO_DIARIO`): guarda el timestamp del último lote
    completado y, si el proceso se corta (ej. reinicio de la sesión), la
    próxima corrida con el mismo `desde_ts` arranca justo después en vez
    de reprocesar lotes ya archivados — se puede llamar de nuevo tal cual
    para retomar.

    Salvaguarda: `hasta_ts` NO puede ser más reciente que
    `RETENCION_DIAS['precios_1h']` días atrás de ahora — mismo `precios_1h`
    que usa el recolector en vivo; sin este límite, un `hasta_ts` mal
    puesto podría archivar-y-borrar datos que todavía están dentro de la
    ventana de entrenamiento activa. Lanza `ValueError` si se viola.

    delay: segundos entre requests a la API (ver recolector.backfill,
    siempre >= 1.0). Un rango de varios años a nivel de todos los ítems
    del juego son decenas de miles de llamadas — puede tardar muchas
    horas; pensado para correrse en background, no interactivo. Estimar
    el tamaño (rango en días × 24 × delay) y confirmar con el usuario
    antes de lanzar un rango grande — ver CLAUDE.md sobre backfills.
    """
    limite_ventana_viva = int(time.time()) - RETENCION_DIAS['precios_1h'] * 86400
    if hasta_ts > limite_ventana_viva:
        raise ValueError(
            f"hasta_ts ({hasta_ts}) cae dentro de la ventana de retención en vivo de precios_1h "
            f"({RETENCION_DIAS['precios_1h']}d) — bajalo a {limite_ventana_viva} o antes para no "
            "arriesgar la ventana de entrenamiento activa."
        )

    from recolector import backfill_faltantes

    paso_lote = dias_por_lote * 86400
    desde_ts = (desde_ts // 86400) * 86400  # alinear a inicio de día UTC

    ultimo_lote_ok = db.obtener_ultima_corrida(TAREA_BACKFILL_ARCHIVO_DIARIO)
    if ultimo_lote_ok is not None and ultimo_lote_ok >= desde_ts:
        desde_ts = ultimo_lote_ok + 1
        logging.info(f"backfill_archivo_diario: retomando desde el último lote completado ({desde_ts})")

    lote_inicio = desde_ts
    n_lotes = 0
    total_filas_diarias = 0
    while lote_inicio <= hasta_ts:
        lote_fin = min(lote_inicio + paso_lote - 3600, hasta_ts)
        logging.info(
            f"backfill_archivo_diario: lote {n_lotes + 1} "
            f"[{datetime.fromtimestamp(lote_inicio, timezone.utc)} - {datetime.fromtimestamp(lote_fin, timezone.utc)}]"
        )
        backfill_faltantes('1h', lote_inicio, lote_fin, db, delay=delay)

        conn = sqlite3.connect(db.db_path)
        c = conn.cursor()
        c.execute(
            '''INSERT OR IGNORE INTO precios_1h_diario
                    (item_id, fecha, avg_high_price, avg_low_price, high_volume, low_volume)
                SELECT
                    item_id,
                    (timestamp / 86400) * 86400 AS fecha,
                    CAST(ROUND(AVG(avg_high_price)) AS INTEGER),
                    CAST(ROUND(AVG(avg_low_price)) AS INTEGER),
                    SUM(high_volume),
                    SUM(low_volume)
                FROM precios_1h
                WHERE timestamp BETWEEN ? AND ?
                GROUP BY item_id, fecha''',
            (lote_inicio, lote_fin),
        )
        filas_diarias = c.rowcount
        total_filas_diarias += filas_diarias
        c.execute('DELETE FROM precios_1h WHERE timestamp BETWEEN ? AND ?', (lote_inicio, lote_fin))
        conn.commit()
        conn.close()

        db.registrar_corrida(TAREA_BACKFILL_ARCHIVO_DIARIO, lote_fin)
        logging.info(f"backfill_archivo_diario: lote {n_lotes + 1} archivado ({filas_diarias} filas diarias)")

        n_lotes += 1
        lote_inicio = lote_fin + 3600

    logging.info(
        f"backfill_archivo_diario completo: {n_lotes} lotes, {total_filas_diarias} filas diarias totales en precios_1h_diario"
    )
    return n_lotes, total_filas_diarias


def purgar_datos_antiguos(db, tabla, dias_retencion, columna_tiempo='timestamp'):
    """
    Purga (DELETE, sin agregación previa) filas de `tabla` más viejas que
    `dias_retencion`. A diferencia de archivar_datos_antiguos (que agrega a
    diario antes de purgar precios_1h), acá no hay downsampling: ninguna de
    las tablas a las que se aplica esto (precios_5m, precios_6h,
    predicciones) alimenta hoy ningún entrenamiento ni screener — cuando
    haga falta la tendencia de largo plazo, precios_1h (con su propia
    retención de 90 días, más lo que haya en precios_1h_diario vía
    archivar_datos_antiguos o backfill_archivo_diario) ya la conserva.

    Igual que archivar_datos_antiguos: el DELETE no recupera espacio en
    disco por sí solo — ver mantenimiento_vacuum().
    """
    corte = int(time.time()) - dias_retencion * 86400
    conn = sqlite3.connect(db.db_path)
    c = conn.cursor()
    c.execute(f'DELETE FROM {tabla} WHERE {columna_tiempo} < ?', (corte,))
    filas_purgadas = c.rowcount
    conn.commit()
    conn.close()

    logging.info(f"{tabla}: {filas_purgadas} filas purgadas (corte={corte}, retención={dias_retencion}d)")
    return filas_purgadas


def podar_metricas_por_item(db, dias_retencion=90):
    """
    Purga de model_metrics solo el detalle por ítem (item_id NOT NULL) más
    viejo que `dias_retencion` — crece con n_items x n_horizontes x
    n_checkpoints (sobre todo con las dos cadencias de modelo y el replay
    histórico, ver entrenador.py/replay_historico.py) y solo importa para
    debugging reciente. Las filas agregadas (item_id IS NULL, pocas por
    corrida) se conservan indefinidamente: alimentan la serie histórica de
    calidad del dashboard (tab "Calidad del modelo").
    """
    corte = int(time.time()) - dias_retencion * 86400
    conn = sqlite3.connect(db.db_path)
    c = conn.cursor()
    c.execute('DELETE FROM model_metrics WHERE item_id IS NOT NULL AND train_timestamp < ?', (corte,))
    filas_purgadas = c.rowcount
    conn.commit()
    conn.close()

    logging.info(f"model_metrics: {filas_purgadas} filas de detalle por ítem podadas (corte={corte}, retención={dias_retencion}d)")
    return filas_purgadas


def mantenimiento_vacuum(db, paginas=2000):
    """
    PRAGMA incremental_vacuum: libera hasta `paginas` páginas ya marcadas
    libres por los DELETE de arriba, sin reescribir el archivo completo
    como haría un VACUUM sin argumentos (que bloquea la DB entera mientras
    corre — no aceptable en un job periódico automático). Requiere que la
    DB esté en auto_vacuum=INCREMENTAL (`PRAGMA auto_vacuum` -> 2) — ya
    migrada una vez, a mano, con `PRAGMA auto_vacuum=INCREMENTAL; VACUUM;`
    sobre la DB detenida (reescribe el archivo completo, por eso es manual
    y no parte de este job); mientras eso no haya pasado, este PRAGMA
    simplemente no libera nada (no falla).
    """
    conn = sqlite3.connect(db.db_path)
    c = conn.cursor()
    c.execute(f'PRAGMA incremental_vacuum({paginas})')
    conn.commit()
    conn.close()
    logging.info(f"mantenimiento_vacuum: incremental_vacuum solicitado (hasta {paginas} páginas)")


def ejecutar_mantenimiento_semanal(db):
    """
    Rutina completa de mantenimiento — pensada para correr una vez por
    semana (ver recolector.job_semanal, que además registra la corrida
    para poder hacer catch-up si se perdió la ventana programada). Agrupa
    todos los pasos acá, en un solo lugar, para no duplicar la lista entre
    este módulo y recolector.py.
    """
    archivar_datos_antiguos(db)
    purgar_datos_antiguos(db, 'precios_5m', dias_retencion=RETENCION_DIAS['precios_5m'])
    purgar_datos_antiguos(db, 'precios_6h', dias_retencion=RETENCION_DIAS['precios_6h'])
    purgar_datos_antiguos(db, 'predicciones', dias_retencion=180)
    podar_metricas_por_item(db, dias_retencion=90)
    mantenimiento_vacuum(db)


if __name__ == '__main__':
    from base_de_datos import OSRSBaseDatos

    logging.basicConfig(level=logging.INFO, format='%(asctime)s |--| %(levelname)s |--| %(message)s')

    db = OSRSBaseDatos('data/osrs_ge.db')
    ejecutar_mantenimiento_semanal(db)
