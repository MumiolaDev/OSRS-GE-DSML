"""
backtest.py — simulación histórica de la estrategia de flip: comprar barato
(avg_low_price) y vender caro (avg_high_price) unas horas después,
descontando el impuesto GE, filtrado por liquidez mínima y opcionalmente
guiado por la señal del modelo.

No es un backtest sintético: tanto la decisión de comprar como el precio de
venta usan datos históricos REALES (precios_1h) que ya ocurrieron — lo
único que involucra una predicción es, con usar_modelo=True, la señal de
qué ítems comprar (los que el modelo esperaba que subieran); la
ganancia/pérdida simulada siempre se calcula contra lo que efectivamente
pasó, nunca contra la predicción.

Para horizonte_horas=1, usa directo las predicciones walk-forward que ya
dejó replay_historico.py en la tabla `predicciones` (rápido: evita repetir
inferencias, y es exactamente lo que el modelo predijo en ese momento real
del pasado). Para horizonte_horas>1, o si el rango pedido no fue
replayeado, cae a un modo más lento que llama
prediccion.pronosticar_item() por cada punto candidato (documentado como
fallback costoso más abajo) — no existe un walk-forward de H pasos
precalculado, solo de 1 paso.
"""

import logging
import sqlite3

import pandas as pd

from entrenador import MODEL_NAME_HORARIO
from metricas import calcular_impuesto_ge, dimensionar_oportunidad
from prediccion import cargar_modelo, pronosticar_item

PASO_SEGUNDOS = 3600  # precios_1h


def _precios_en_rango(db, tabla, desde_ts, hasta_ts):
    """Todos los precios de `tabla` en [desde_ts, hasta_ts], para todos los
    ítems, en una sola query — mucho más barato que una query por
    ítem/timestamp cuando se simula sobre un rango largo."""
    conn = sqlite3.connect(db.db_path)
    df = pd.read_sql_query(
        f'SELECT item_id, timestamp, avg_high_price, avg_low_price, high_volume, low_volume '
        f'FROM {tabla} WHERE timestamp BETWEEN ? AND ?',
        conn, params=(desde_ts, hasta_ts),
    )
    conn.close()
    return df


def _buy_limits(db, item_ids):
    if not item_ids:
        return {}
    conn = sqlite3.connect(db.db_path)
    placeholders = ','.join('?' * len(item_ids))
    df = pd.read_sql_query(
        f'SELECT item_id, buy_limit FROM items WHERE item_id IN ({placeholders})',
        conn, params=[int(i) for i in item_ids],
    )
    conn.close()
    return dict(zip(df['item_id'], df['buy_limit']))


def _senal_modelo_walkforward(db, model_name, desde_ts, hasta_ts):
    """
    Predicciones walk-forward de 1 paso (replay_historico.py) con timestamp
    (el momento predicho, no el de entrenamiento) en [desde_ts, hasta_ts].
    Cada fila es "el modelo, entrenado con datos hasta timestamp-1h, predijo
    este avg_low_price para timestamp" — solo sirve como atajo rápido
    cuando horizonte_horas=1 exactamente (para horizontes mayores no existe
    un walk-forward de H pasos precalculado, ver el fallback recursivo en
    simular_estrategia).
    """
    conn = sqlite3.connect(db.db_path)
    df = pd.read_sql_query(
        "SELECT item_id, timestamp, predicted_price FROM predicciones "
        "WHERE model_version = ? AND modo_evaluacion = 'walkforward' AND timestamp BETWEEN ? AND ?",
        conn, params=(model_name, desde_ts, hasta_ts),
    )
    conn.close()
    return df


def _resumen_vacio():
    return {'profit_total': 0.0, 'win_rate': None, 'n_trades': 0, 'profit_promedio': None}


def simular_estrategia(
    db, desde_ts, hasta_ts,
    volumen_24h_minimo=100,
    usar_modelo=True,
    model_name=MODEL_NAME_HORARIO,
    umbral_subida_pct=2.0,
    horizonte_horas=3,
    fraccion_participacion_volumen=0.15,
):
    """
    Simula, para cada hora `t` en [desde_ts, hasta_ts] y cada ítem con
    volumen_24h >= volumen_24h_minimo, comprar a avg_low_price(t) y vender a
    avg_high_price(t + horizonte_horas) — ambos precios reales, ya
    conocidos. Si usar_modelo=True, solo entra a un ítem cuando el modelo
    esperaba una subida >= umbral_subida_pct hacia ese horizonte (ver el
    docstring del módulo sobre de dónde sale esa señal). El impuesto GE se
    descuenta con metricas.calcular_impuesto_ge, y la cantidad de unidades
    se dimensiona de forma conservadora con metricas.dimensionar_oportunidad
    (no asume llenar el buy_limit completo).

    Devuelve (trades, resumen): `trades` es un DataFrame con una fila por
    operación simulada; `resumen` es un dict con profit_total, win_rate,
    n_trades, profit_promedio. Correr con usar_modelo=True/False por
    separado sobre el mismo rango para comparar "solo screener" vs
    "screener+modelo".
    """
    margen_carga = 24 * PASO_SEGUNDOS  # para poder calcular volumen_24h en el primer punto simulado
    precios = _precios_en_rango(
        db, 'precios_1h', desde_ts - margen_carga, hasta_ts + horizonte_horas * PASO_SEGUNDOS,
    )
    if precios.empty:
        logging.warning("simular_estrategia: sin datos de precios_1h en el rango pedido.")
        return pd.DataFrame(), _resumen_vacio()

    buy_limits = _buy_limits(db, precios['item_id'].unique().tolist())

    señal_modelo = pd.Series(dtype=float)
    usar_atajo_walkforward = usar_modelo and horizonte_horas == 1
    if usar_atajo_walkforward:
        pred_df = _senal_modelo_walkforward(db, model_name, desde_ts + PASO_SEGUNDOS, hasta_ts + PASO_SEGUNDOS)
        if pred_df.empty:
            logging.warning(
                f"simular_estrategia: sin predicciones walk-forward para '{model_name}' en este rango — "
                "correr replay_historico.ejecutar_replay() primero para que esto sea rápido. "
                "Usando el fallback lento (prediccion.pronosticar_item por punto) mientras tanto."
            )
        else:
            señal_modelo = pred_df.set_index(['item_id', 'timestamp'])['predicted_price']

    bundle = None  # se carga perezosamente, solo si hace falta el fallback recursivo

    trades = []
    for item_id, serie in precios.groupby('item_id'):
        serie = serie.sort_values('timestamp').set_index('timestamp')
        buy_limit = buy_limits.get(item_id)
        if not buy_limit:
            continue

        for t in serie.index:
            if t < desde_ts or t > hasta_ts:
                continue  # el margen de carga extra es solo para volumen/venta, no para simular acá

            avg_low = serie.at[t, 'avg_low_price']
            avg_high = serie.at[t, 'avg_high_price']
            if not avg_low or not avg_high:
                continue

            vol_24h = serie.loc[t - margen_carga + PASO_SEGUNDOS: t, ['high_volume', 'low_volume']].sum().sum()
            if vol_24h < volumen_24h_minimo:
                continue

            t_venta = t + horizonte_horas * PASO_SEGUNDOS

            if usar_modelo:
                pred = None
                if usar_atajo_walkforward and (item_id, t_venta) in señal_modelo.index:
                    pred = señal_modelo.loc[(item_id, t_venta)]
                else:
                    if bundle is None:
                        bundle = cargar_modelo(model_name=model_name)
                    pronostico = pronosticar_item(
                        db, item_id, bundle, n_pasos=horizonte_horas, tabla='precios_1h', hasta_timestamp=t,
                    )
                    if not pronostico.empty:
                        pred = pronostico.iloc[-1]['predicted_price']
                if pred is None:
                    continue
                subida_pct = (pred - avg_low) / avg_low * 100
                if subida_pct < umbral_subida_pct:
                    continue

            if t_venta not in serie.index:
                continue
            precio_venta_real = serie.at[t_venta, 'avg_high_price']
            if not precio_venta_real:
                continue

            impuesto = calcular_impuesto_ge(precio_venta_real, item_id)
            margen_neto = precio_venta_real - avg_low - impuesto
            volumen_1h_promedio = vol_24h / 24
            profit = dimensionar_oportunidad(
                margen_neto, buy_limit, volumen_1h_promedio, horizonte_horas, fraccion_participacion_volumen,
            )
            if profit is None:
                continue

            trades.append({
                'item_id': int(item_id),
                'timestamp_compra': int(t),
                'timestamp_venta': int(t_venta),
                'avg_low_price': float(avg_low),
                'precio_venta_real': float(precio_venta_real),
                'margen_neto': float(margen_neto),
                'profit': float(profit),
            })

    trades_df = pd.DataFrame(trades)
    if trades_df.empty:
        return trades_df, _resumen_vacio()

    resumen = {
        'profit_total': float(trades_df['profit'].sum()),
        'win_rate': float((trades_df['profit'] > 0).mean()),
        'n_trades': int(len(trades_df)),
        'profit_promedio': float(trades_df['profit'].mean()),
    }
    return trades_df, resumen


def _simular_trades_clasificador(db, test, fraccion_participacion_volumen=0.15, tabla='precios_1h'):
    """
    Toma un DataFrame de predicciones del clasificador direccional
    (columnas item_id, timestamp_target, price_actual, clase_predicha —
    el formato que devuelve entrenador.entrenar_clasificador_direccional)
    y simula comprar a avg_low_price(t)=price_actual cuando clase_predicha
    es 'sube' (2), vendiendo a avg_high_price(t+1) real — el horizonte es
    fijo en 1 paso, no hay pronóstico recursivo. Compartido entre
    simular_clasificador (un solo split holdout) y
    simular_clasificador_walkforward (muchos checkpoints acumulados) —
    misma lógica de simulación, distinta fuente de las predicciones.
    """
    candidatos = test[test['clase_predicha'] == 2].copy()  # 2 = 'sube'
    logging.info(f"simular_clasificador: {len(candidatos)} de {len(test)} filas predichas 'sube'")
    if candidatos.empty:
        return pd.DataFrame(), _resumen_vacio()

    item_ids = candidatos['item_id'].astype(int).unique().tolist()
    buy_limits = _buy_limits(db, item_ids)

    # Todo el historial de estos pocos ítems (10 por default) en una sola
    # query (obtener_precios_multi) — necesario porque el dataset de
    # features del clasificador no retiene avg_high_price/volumen crudos
    # (solo price_actual=avg_low en t y timestamp_target=t+1).
    precios_multi = db.obtener_precios_multi(item_ids, tabla)
    series_por_item = {
        iid: grupo.sort_values('timestamp').set_index('timestamp')
        for iid, grupo in precios_multi.groupby('item_id')
    }

    trades = []
    for _, fila in candidatos.iterrows():
        item_id = int(fila['item_id'])
        t_venta = int(fila['timestamp_target'])
        t_compra = t_venta - PASO_SEGUNDOS
        avg_low = float(fila['price_actual'])

        serie = series_por_item.get(item_id)
        buy_limit = buy_limits.get(item_id)
        if serie is None or not buy_limit or not avg_low or t_venta not in serie.index:
            continue
        precio_venta_real = serie.at[t_venta, 'avg_high_price']
        if not precio_venta_real:
            continue

        ventana = serie.loc[t_compra - 23 * PASO_SEGUNDOS: t_compra, ['high_volume', 'low_volume']]
        vol_24h = ventana.sum().sum() if not ventana.empty else 0.0
        volumen_1h_promedio = vol_24h / 24

        impuesto = calcular_impuesto_ge(precio_venta_real, item_id)
        margen_neto = precio_venta_real - avg_low - impuesto
        profit = dimensionar_oportunidad(margen_neto, buy_limit, volumen_1h_promedio, 1, fraccion_participacion_volumen)
        if profit is None:
            continue

        trades.append({
            'item_id': item_id,
            'timestamp_compra': t_compra,
            'timestamp_venta': t_venta,
            'avg_low_price': avg_low,
            'precio_venta_real': float(precio_venta_real),
            'margen_neto': float(margen_neto),
            'profit': float(profit),
        })

    trades_df = pd.DataFrame(trades)
    if trades_df.empty:
        return trades_df, _resumen_vacio()

    resumen = {
        'profit_total': float(trades_df['profit'].sum()),
        'win_rate': float((trades_df['profit'] > 0).mean()),
        'n_trades': int(len(trades_df)),
        'profit_promedio': float(trades_df['profit'].mean()),
    }
    return trades_df, resumen


def simular_clasificador(db, n_items=10, solo_f2p=True, umbral_pct=None, model_name=None, fraccion_participacion_volumen=0.15, tabla='precios_1h', precio_minimo=None, excluir_item_ids=None):
    """
    Backtest de entrenador.entrenar_clasificador_direccional sobre un único
    split holdout: entrena una vez y simula SOLO sobre el test set (nunca
    visto en entrenamiento — la regla de oro de cualquier backtest). Rápido
    (una sola corrida de entrenamiento), pero es "una sola foto del
    tiempo" — ver simular_clasificador_walkforward para una validación más
    robusta sobre muchos checkpoints históricos.

    n_items/solo_f2p: ver entrenar_clasificador_direccional — ej.
    n_items=10, solo_f2p=True (default) restringe el universo a los 10
    ítems free-to-play más líquidos (runas/materiales clásicos), en vez de
    los 200 ítems (F2P+members) del modelo global.

    Devuelve (trades, resumen), mismo formato que simular_estrategia().
    """
    from entrenador import entrenar_clasificador_direccional, MODEL_NAME_CLASIFICADOR, MODEL_NAME_CLASIF_F2P

    model_name = model_name or (MODEL_NAME_CLASIF_F2P if solo_f2p else MODEL_NAME_CLASIFICADOR)
    kwargs = {}
    if umbral_pct is not None:
        kwargs['umbral_pct'] = umbral_pct
    if precio_minimo is not None:
        kwargs['precio_minimo'] = precio_minimo
    if excluir_item_ids is not None:
        kwargs['excluir_item_ids'] = excluir_item_ids
    modelo, test = entrenar_clasificador_direccional(db, n_items=n_items, solo_f2p=solo_f2p, model_name=model_name, **kwargs)
    if test is None or test.empty:
        logging.warning("simular_clasificador: no se pudo entrenar/evaluar el clasificador.")
        return pd.DataFrame(), _resumen_vacio()

    return _simular_trades_clasificador(db, test, fraccion_participacion_volumen, tabla)


def simular_clasificador_walkforward(db, desde_ts, hasta_ts, n_items=10, solo_f2p=True, umbral_pct=None,
                                       model_name=None, fraccion_participacion_volumen=0.15, tabla='precios_1h',
                                       precio_minimo=None, excluir_item_ids=None):
    """
    Como simular_clasificador, pero sobre TODO un rango histórico vía
    replay_historico.ejecutar_replay_clasificador: cada predicción viene de
    un modelo reentrenado en ese momento exacto del pasado (walk-forward de
    verdad), no de un único split holdout. Más lento (un reentreno por
    checkpoint horario) pero mucho más robusto — la accuracy/PnL ya no
    depende de qué tan representativo resultó ser un solo corte de tiempo.

    precio_minimo: ver entrenador.entrenar_clasificador_direccional —
    descarta ítems baratos del universo F2P antes de elegir los n_items
    más líquidos, en cada checkpoint del rango.

    Devuelve (trades, resumen, predicciones) — `predicciones` es el
    DataFrame acumulado de todos los checkpoints (con clase_predicha/
    clase_real), útil para calcular accuracy agregada sobre todo el rango
    además del resultado de PnL.
    """
    from replay_historico import ejecutar_replay_clasificador

    predicciones = ejecutar_replay_clasificador(
        db, desde_ts, hasta_ts, n_items=n_items, solo_f2p=solo_f2p, model_name=model_name,
        umbral_pct=umbral_pct, precio_minimo=precio_minimo, excluir_item_ids=excluir_item_ids,
    )
    if predicciones.empty:
        logging.warning("simular_clasificador_walkforward: sin predicciones acumuladas en el rango pedido.")
        return pd.DataFrame(), _resumen_vacio(), predicciones

    trades, resumen = _simular_trades_clasificador(db, predicciones, fraccion_participacion_volumen, tabla)
    return trades, resumen, predicciones


if __name__ == '__main__':
    import time

    from base_de_datos import OSRSBaseDatos

    logging.basicConfig(level=logging.INFO, format='%(asctime)s |--| %(levelname)s |--| %(message)s')
    db = OSRSBaseDatos('data/osrs_ge.db')

    hasta = int(time.time())
    desde = hasta - 7 * 86400  # última semana, ajustar según qué rango ya tenga replay corrido

    for usar_modelo in (False, True):
        trades, resumen = simular_estrategia(db, desde, hasta, horizonte_horas=1, usar_modelo=usar_modelo)
        etiqueta = "screener+modelo" if usar_modelo else "solo screener"
        logging.info(f"[{etiqueta}] {resumen}")
