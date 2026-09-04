"""
evaluacion.py — evaluación del modelo a horizontes mayores a 1 paso.

entrenador.py ya mide el error de 1 paso (predecir t+1 y comparar contra el
precio real ya conocido). Acá se mide lo mismo pero a 2..N pasos, usando el
mismo pronóstico recursivo que ve el usuario en el dashboard y en las
alertas (prediccion.pronosticar_item) — es lo que responde "¿hasta qué
horizonte puedo confiar en una alerta?": el error compuesto crece con cada
paso recursivo (ver el docstring de prediccion.py), y sin esto no había
forma de cuantificar cuánto.

A diferencia de un pronóstico real hacia adelante (donde el "futuro"
todavía no existe), acá se parte de un timestamp del pasado y se compara
contra el precio real que ya está en la base de datos varios pasos después
— por eso reusa pronosticar_item con `hasta_timestamp` en vez de duplicar
la lógica recursiva.
"""

import logging
import sqlite3

import numpy as np
import pandas as pd

from prediccion import pronosticar_item


def evaluar_horizontes(db, bundle, puntos_eval, n_pasos=6, tabla='precios_1h'):
    """
    puntos_eval: lista de tuplas (item_id, timestamp) — puntos de partida a
    evaluar (típicamente una muestra de (item_id, timestamp_target) del
    test set de entrenador.py). Para cada uno, pronostica hasta `n_pasos`
    períodos hacia adelante partiendo de datos <= timestamp, y compara cada
    paso contra el precio real ya conocido en la DB.

    Devuelve un DataFrame con una fila por (item_id, punto, horizonte)
    evaluado: item_id, horizonte, error_abs (gp), acierto_direccional
    (bool) — vacío si ningún punto tenía suficiente historia o precio real
    posterior disponible para comparar.
    """
    target_col = bundle['target_col']

    por_item = {}
    for item_id, ts in puntos_eval:
        por_item.setdefault(int(item_id), []).append(int(ts))

    filas = []
    conn = sqlite3.connect(db.db_path)
    for item_id, puntos in por_item.items():
        real = pd.read_sql_query(
            f'SELECT timestamp, {target_col} AS precio FROM {tabla} WHERE item_id = ?',
            conn, params=[item_id],
        )
        if real.empty:
            continue
        real = real.set_index('timestamp')['precio']

        for punto in puntos:
            anteriores = real[real.index <= punto]
            if anteriores.empty:
                continue
            precio_actual = float(anteriores.iloc[-1])

            pronostico = pronosticar_item(
                db, item_id, bundle, n_pasos=n_pasos, tabla=tabla, hasta_timestamp=punto,
            )
            for _, fila in pronostico.iterrows():
                if int(fila['paso']) == 1:
                    # El horizonte 1 ya lo mide entrenador.py directamente sobre
                    # todo el test set (más barato, y en espacio log-retorno,
                    # comparable entre ítems de escalas de precio distintas).
                    # Guardarlo también acá — en espacio gp, sobre una muestra
                    # más chica — dejaría dos filas con la misma clave
                    # (model_name, train_timestamp, horizonte_horas=1,
                    # modo_evaluacion) pero en unidades distintas, sin forma de
                    # distinguirlas después. evaluar_horizontes mide el costo
                    # ADICIONAL de seguir extrapolando recursivamente más allá
                    # de ese primer paso.
                    continue
                ts_pred = int(fila['timestamp'])
                if ts_pred not in real.index:
                    # el punto de evaluación está tan cerca del final del
                    # historial que este paso todavía no tiene precio real
                    # con el cual compararse — se omite, no es un error.
                    continue
                precio_real = float(real.loc[ts_pred])
                # acierto_direccional queda en None (no True/False) cuando
                # el precio real no se movió respecto del punto de partida:
                # no hay dirección que acertar, y contarlo como fallo es lo
                # que hundía artificialmente la métrica del regresor (ver
                # entrenador.accuracy_direccional). Quien agrega estas filas
                # descarta los None antes de promediar.
                movimiento_real = np.sign(precio_real - precio_actual)
                acierto = (
                    None if movimiento_real == 0
                    else bool(movimiento_real == np.sign(fila['predicted_price'] - precio_actual))
                )
                filas.append({
                    'item_id': item_id,
                    'horizonte': int(fila['paso']),
                    'error_abs': abs(precio_real - fila['predicted_price']),
                    'acierto_direccional': acierto,
                })
    conn.close()

    if not filas:
        logging.warning("evaluar_horizontes: ningún punto de evaluación tenía precio real posterior disponible.")
    return pd.DataFrame(filas)
