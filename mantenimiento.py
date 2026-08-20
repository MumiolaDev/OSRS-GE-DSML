"""
mantenimiento.py — retención y downsampling de precios_1h.

Sin esto, precios_1h crece sin límite (recolectando el catálogo completo a 1h
son ~4.600 ítems x 24 filas/día ≈ 110.000 filas/día, ~40M filas/año), y cada
consulta (screener, entrenamiento) se vuelve progresivamente más lenta.

La política: los datos horarios más viejos que `dias_retencion` se agregan a
resolución diaria en `precios_1h_diario` (conserva la tendencia de largo
plazo) y después se purgan de `precios_1h` (ya no hace falta el detalle
horario para algo de hace más de un año). La agregación se hace en SQL
directo (no trayendo todo a pandas) porque son potencialmente millones de
filas.

Pensado para correr periódicamente (ej. semanal) desde el loop de
recolector.py, no en cada arranque.
"""

import logging
import sqlite3
import time


def archivar_datos_antiguos(db, tabla='precios_1h', tabla_diaria='precios_1h_diario', dias_retencion=365):
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
    corte = int(time.time()) - dias_retencion * 86400
    segundos_dia = 86400

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


if __name__ == '__main__':
    from base_de_datos import OSRSBaseDatos

    logging.basicConfig(level=logging.INFO, format='%(asctime)s |--| %(levelname)s |--| %(message)s')

    db = OSRSBaseDatos('data/osrs_ge.db')
    archivar_datos_antiguos(db)
