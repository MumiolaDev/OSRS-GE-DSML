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

Dos supuestos que este archivo hace EXPLÍCITOS (antes estaban implícitos y
ambos inflaban el resultado):

1. Nunca se opera con información del período en que se decide. La decisión
   se toma al cierre del período `t` — que es cuando la API publica sus
   datos — y la compra ocurre en `t+1`, no a `avg_low_price(t)`, que es el
   promedio de un período ya terminado.
2. No se sabe si una orden se completa. Cada trade se valúa en sus DOS
   cotas: optimista (comprar en la punta baja, vender en la alta: se captura
   el spread, requiere que las dos órdenes se completen) y pesimista
   (cruzar el spread en las dos puntas: ejecución garantizada). Ver
   _resumen — la diferencia entre las dos cotas es enorme y no la explica el
   modelo, la explica el spread.

Además se respeta el reset de 4 horas del límite de compra del GE
(HORAS_BUY_LIMIT): no se puede volver a comprar el buy_limit del mismo ítem
cada hora. El capital sigue siendo ilimitado — la simulación entra a todas
las oportunidades a la vez, así que profit_total es una cota superior de lo
que rendiría una bolsa acotada.

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
from preprocesamiento import PASO_SEGUNDOS_POR_TABLA

PASO_SEGUNDOS = 3600  # precios_1h — default; ver PASO_SEGUNDOS_POR_TABLA

# El límite de compra del Grand Exchange se resetea cada 4 horas: no se
# puede volver a comprar el buy_limit completo de un ítem cada hora. Sin
# esta restricción la simulación acumulaba posiciones solapadas del mismo
# ítem hora tras hora y el profit_total resultante no era alcanzable.
HORAS_BUY_LIMIT = 4


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
    return {
        'profit_total_optimista': 0.0, 'profit_total_pesimista': 0.0,
        'win_rate_optimista': None, 'win_rate_pesimista': None,
        'n_trades': 0, 'profit_promedio_optimista': None, 'profit_promedio_pesimista': None,
    }


def _resumen(trades_df):
    """
    Resumen con las DOS cotas de ejecución, nunca una sola.

    El proyecto simulaba únicamente el caso optimista: comprar al
    avg_low_price (poniendo una orden de compra en la punta baja) y vender al
    avg_high_price (orden de venta en la punta alta), asumiendo que las dos
    se completan siempre. Eso no es una estrategia de predicción sino de
    captura de spread, y su rentabilidad depende por completo de una
    probabilidad de ejecución que este backtest no modela.

    Medido sobre 365 horas reales de la API: comprando bajo y vendiendo alto,
    entre el 15% y el 84% de las horas cierran en ganancia según el ítem;
    cruzando el spread (comprar alto, vender bajo, o sea con ejecución
    inmediata garantizada) baja al 0-1%. La diferencia entera es el spread,
    no el modelo. El resultado real de operar está entre esas dos cotas,
    así que se informan las dos y la comparación honesta entre "con modelo" y
    "sin modelo" es la que se sostiene en ambas.
    """
    if trades_df.empty:
        return _resumen_vacio()
    return {
        'profit_total_optimista': float(trades_df['profit_optimista'].sum()),
        'profit_total_pesimista': float(trades_df['profit_pesimista'].sum()),
        'win_rate_optimista': float((trades_df['profit_optimista'] > 0).mean()),
        'win_rate_pesimista': float((trades_df['profit_pesimista'] > 0).mean()),
        'n_trades': int(len(trades_df)),
        'profit_promedio_optimista': float(trades_df['profit_optimista'].mean()),
        'profit_promedio_pesimista': float(trades_df['profit_pesimista'].mean()),
    }


def _profits_de_un_trade(item_id, precio_compra_bajo, precio_compra_alto,
                         precio_venta_alto, precio_venta_bajo,
                         buy_limit, volumen_1h_promedio, horas, fraccion_volumen):
    """
    (profit_optimista, profit_pesimista, margen_optimista, margen_pesimista)
    de un trade, con el impuesto GE descontado del precio de venta en cada
    caso. Optimista = comprar en la punta baja y vender en la alta (las dos
    órdenes se completan); pesimista = cruzar el spread en las dos puntas
    (ejecución inmediata garantizada). Ver _resumen.
    """
    margen_opt = precio_venta_alto - precio_compra_bajo - calcular_impuesto_ge(precio_venta_alto, item_id)
    margen_pes = precio_venta_bajo - precio_compra_alto - calcular_impuesto_ge(precio_venta_bajo, item_id)
    profit_opt = dimensionar_oportunidad(margen_opt, buy_limit, volumen_1h_promedio, horas, fraccion_volumen)
    profit_pes = dimensionar_oportunidad(margen_pes, buy_limit, volumen_1h_promedio, horas, fraccion_volumen)
    return profit_opt, profit_pes, margen_opt, margen_pes


def simular_estrategia(
    db, desde_ts, hasta_ts,
    volumen_24h_minimo=100,
    usar_modelo=True,
    model_name=MODEL_NAME_HORARIO,
    umbral_subida_pct=2.0,
    horizonte_horas=3,
    fraccion_participacion_volumen=0.15,
    respetar_buy_limit_4h=True,
):
    """
    Simula, para cada hora `t` en [desde_ts, hasta_ts] y cada ítem con
    volumen_24h >= volumen_24h_minimo, la decisión tomada al CIERRE de la
    hora `t` (que es cuando se conocen sus datos): comprar durante la hora
    siguiente, `t+1`, y vender `horizonte_horas` después de esa.

    Ese desfase de una barra no es un detalle: antes se compraba a
    avg_low_price(t), o sea al promedio de una hora que ya había terminado
    cuando se tomaba la decisión — un precio que ya no está disponible. Con
    un movimiento horario mediano del 0.2-0.9% (medido sobre datos reales),
    una barra de ventaja alcanza para inventar toda la rentabilidad.

    Si usar_modelo=True, solo entra a un ítem cuando el modelo esperaba una
    subida >= umbral_subida_pct (ver el docstring del módulo sobre de dónde
    sale esa señal). El impuesto GE se descuenta con
    metricas.calcular_impuesto_ge, y la cantidad de unidades se dimensiona
    de forma conservadora con metricas.dimensionar_oportunidad (no asume
    llenar el buy_limit completo). Con respetar_buy_limit_4h=True (default)
    tampoco se vuelve a entrar al mismo ítem antes de que se resetee su
    límite de compra.

    Devuelve (trades, resumen): `trades` es un DataFrame con una fila por
    operación simulada, con profit_optimista/profit_pesimista (ver
    _resumen); `resumen` trae las dos cotas. Correr con usar_modelo=True/
    False por separado sobre el mismo rango para comparar "solo screener"
    vs "screener+modelo" — la comparación solo vale si se sostiene en las
    dos cotas.
    """
    margen_carga = 24 * PASO_SEGUNDOS  # para poder calcular volumen_24h en el primer punto simulado
    precios = _precios_en_rango(
        db, 'precios_1h', desde_ts - margen_carga,
        hasta_ts + (horizonte_horas + 1) * PASO_SEGUNDOS,
    )
    if precios.empty:
        logging.warning("simular_estrategia: sin datos de precios_1h en el rango pedido.")
        return pd.DataFrame(), _resumen_vacio()

    buy_limits = _buy_limits(db, precios['item_id'].unique().tolist())

    señal_modelo = pd.Series(dtype=float)
    # El atajo ya no está limitado a horizonte_horas==1: la señal de entrada
    # es siempre la predicción de 1 paso para el período de compra, sea cual
    # sea cuánto se sostenga después la posición (ver el loop de abajo).
    usar_atajo_walkforward = usar_modelo
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

        ultima_compra_ts = None
        for t in serie.index:
            if t < desde_ts or t > hasta_ts:
                continue  # el margen de carga extra es solo para volumen/venta, no para simular acá

            # Precio conocido al momento de decidir (cierre de la hora t):
            # es contra este que se mide la señal del modelo, pero NO es a
            # este que se compra (ver más abajo).
            avg_low_conocido = serie.at[t, 'avg_low_price']
            if not avg_low_conocido:
                continue

            vol_24h = serie.loc[t - margen_carga + PASO_SEGUNDOS: t, ['high_volume', 'low_volume']].sum().sum()
            if vol_24h < volumen_24h_minimo:
                continue

            t_compra = t + PASO_SEGUNDOS
            t_venta = t_compra + horizonte_horas * PASO_SEGUNDOS

            if respetar_buy_limit_4h and ultima_compra_ts is not None:
                if t_compra - ultima_compra_ts < HORAS_BUY_LIMIT * PASO_SEGUNDOS:
                    continue

            if usar_modelo:
                # La señal es SIEMPRE la predicción de 1 paso para el período
                # de compra (t+1), tanto por el atajo walk-forward como por
                # el fallback. Es lo único que el modelo sabe decir de forma
                # confiable: el pronóstico recursivo a varios pasos compone
                # su propio error (ver prediccion.py) y, sobre todo, la
                # predicción walk-forward guardada en `predicciones` es
                # siempre de 1 paso. Con horizonte_horas>1 la señal sigue
                # siendo "el precio viene subiendo ahora", no una predicción
                # del momento de venta.
                pred = None
                if usar_atajo_walkforward and (item_id, t_compra) in señal_modelo.index:
                    pred = señal_modelo.loc[(item_id, t_compra)]
                else:
                    if bundle is None:
                        bundle = cargar_modelo(model_name=model_name)
                    pronostico = pronosticar_item(db, item_id, bundle, n_pasos=1, hasta_timestamp=t)
                    if not pronostico.empty:
                        pred = pronostico.iloc[-1]['predicted_price']
                if pred is None:
                    continue
                subida_pct = (pred - avg_low_conocido) / avg_low_conocido * 100
                if subida_pct < umbral_subida_pct:
                    continue

            if t_compra not in serie.index or t_venta not in serie.index:
                continue
            compra_baja = serie.at[t_compra, 'avg_low_price']
            compra_alta = serie.at[t_compra, 'avg_high_price']
            venta_alta = serie.at[t_venta, 'avg_high_price']
            venta_baja = serie.at[t_venta, 'avg_low_price']
            if not compra_baja or not compra_alta or not venta_alta or not venta_baja:
                continue

            profit_opt, profit_pes, margen_opt, margen_pes = _profits_de_un_trade(
                item_id, compra_baja, compra_alta, venta_alta, venta_baja,
                buy_limit, vol_24h / 24, horizonte_horas, fraccion_participacion_volumen,
            )
            if profit_opt is None or profit_pes is None:
                continue

            ultima_compra_ts = t_compra
            trades.append({
                'item_id': int(item_id),
                'timestamp_decision': int(t),
                'timestamp_compra': int(t_compra),
                'timestamp_venta': int(t_venta),
                'precio_compra_optimista': float(compra_baja),
                'precio_compra_pesimista': float(compra_alta),
                'precio_venta_optimista': float(venta_alta),
                'precio_venta_pesimista': float(venta_baja),
                'margen_neto_optimista': float(margen_opt),
                'margen_neto_pesimista': float(margen_pes),
                'profit_optimista': float(profit_opt),
                'profit_pesimista': float(profit_pes),
            })

    trades_df = pd.DataFrame(trades)
    return trades_df, _resumen(trades_df)


def _simular_trades_clasificador(db, test, fraccion_participacion_volumen=0.15, tabla='precios_1h',
                                 respetar_buy_limit_4h=True):
    """
    Toma un DataFrame de predicciones del clasificador direccional
    (columnas item_id, timestamp_target, price_actual, clase_predicha —
    el formato que devuelve entrenador.entrenar_clasificador_direccional)
    y simula el trade que esa señal habilita. Compartido entre
    simular_clasificador (un solo split holdout) y
    simular_clasificador_walkforward (muchos checkpoints acumulados) —
    misma lógica de simulación, distinta fuente de las predicciones.

    Cada fila del test set es "con datos hasta t, el modelo predice la clase
    del período t+1 (= timestamp_target)". La decisión se toma al cierre de
    t, así que el primer período en el que se puede operar es t+1: se compra
    durante t+1 y se vende durante t+2. Antes se compraba a `price_actual`,
    que es el avg_low de t — el promedio de un período que ya había
    terminado cuando la señal existía.

    El paso temporal sale de `tabla`, no fijo en una hora: con un modelo
    sobre precios_5m/precios_6h el emparejamiento compra/venta quedaba
    desplazado por usar 3600 siempre.
    """
    paso = PASO_SEGUNDOS_POR_TABLA.get(tabla, PASO_SEGUNDOS)
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
    ultima_compra_por_item = {}
    for _, fila in candidatos.sort_values('timestamp_target').iterrows():
        item_id = int(fila['item_id'])
        t_compra = int(fila['timestamp_target'])  # el período que el modelo predijo
        t_venta = t_compra + paso

        serie = series_por_item.get(item_id)
        buy_limit = buy_limits.get(item_id)
        if serie is None or not buy_limit:
            continue
        if t_compra not in serie.index or t_venta not in serie.index:
            continue

        ultima = ultima_compra_por_item.get(item_id)
        if respetar_buy_limit_4h and ultima is not None and t_compra - ultima < HORAS_BUY_LIMIT * 3600:
            continue

        compra_baja = serie.at[t_compra, 'avg_low_price']
        compra_alta = serie.at[t_compra, 'avg_high_price']
        venta_alta = serie.at[t_venta, 'avg_high_price']
        venta_baja = serie.at[t_venta, 'avg_low_price']
        if not compra_baja or not compra_alta or not venta_alta or not venta_baja:
            continue

        # Volumen de las 24h previas a la decisión (que se toma en t_compra
        # - paso, el último período con datos conocidos).
        t_decision = t_compra - paso
        ventana = serie.loc[t_decision - 23 * paso: t_decision, ['high_volume', 'low_volume']]
        vol_24h = ventana.sum().sum() if not ventana.empty else 0.0

        profit_opt, profit_pes, margen_opt, margen_pes = _profits_de_un_trade(
            item_id, compra_baja, compra_alta, venta_alta, venta_baja,
            buy_limit, vol_24h / 24, 1, fraccion_participacion_volumen,
        )
        if profit_opt is None or profit_pes is None:
            continue

        ultima_compra_por_item[item_id] = t_compra
        trades.append({
            'item_id': item_id,
            'timestamp_decision': t_decision,
            'timestamp_compra': t_compra,
            'timestamp_venta': t_venta,
            'precio_compra_optimista': float(compra_baja),
            'precio_compra_pesimista': float(compra_alta),
            'precio_venta_optimista': float(venta_alta),
            'precio_venta_pesimista': float(venta_baja),
            'margen_neto_optimista': float(margen_opt),
            'margen_neto_pesimista': float(margen_pes),
            'profit_optimista': float(profit_opt),
            'profit_pesimista': float(profit_pes),
        })

    trades_df = pd.DataFrame(trades)
    return trades_df, _resumen(trades_df)


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
        logging.info(
            f"[{etiqueta}] {resumen['n_trades']} trades | "
            f"optimista (captura el spread): {resumen['profit_total_optimista']:,.0f} gp | "
            f"pesimista (lo cruza): {resumen['profit_total_pesimista']:,.0f} gp"
        )
