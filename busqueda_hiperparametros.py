"""
busqueda_hiperparametros.py — ambiente de testeo APARTE de la app de
escritorio (nunca escribe en modelos_config del usuario ni toca ningún
.pkl productivo): recorre una grilla de configuraciones (ventana de
historial, umbral del clasificador) para uno o más grupos de ítems, entrena
cada una walk-forward sobre las últimas 2 semanas, y evalúa cada
configuración con un backtest de capital constante — se empieza con
`capital_inicial` gp y se van ejecutando compras/ventas reales a lo largo
de las 2 semanas, siguiendo la señal de "sube" a 1 hora de cada checkpoint
walk-forward, nunca con capital fantasma (una compra que no se puede pagar
no se ejecuta).

Deliberadamente caro — se prioriza fidelidad (walk-forward real, no un
solo split) sobre velocidad, mismo criterio que replay_historico.py. Cada
prueba entrena ~336 checkpoints (una por hora, 2 semanas); con varias
pruebas esto puede tardar horas — pensado para correrse en background, no
interactivo.

Resultados: se van agregando a `resultados_busqueda.csv` fila por fila (así
sobreviven aunque el proceso se corte a mitad de camino) y al final se
imprime un resumen rankeado por ganancia.
"""
import logging
import time
from datetime import datetime, timezone

import pandas as pd

from base_de_datos import OSRSBaseDatos
from metricas import calcular_impuesto_ge
from replay_historico import ejecutar_replay_modelo, ejecutar_replay_clasificador

DB_PATH = 'data/osrs_ge.db'
CSV_RESULTADOS = 'resultados_busqueda.csv'

DIAS_BACKTEST = 14          # "un periodo de 2 semanas", pedido explícito del usuario
PASO_SEGUNDOS = 3600        # "predicciones de la siguiente hora"
FRACCION_PARTICIPACION = 0.15  # mismo default conservador que metricas.dimensionar_oportunidad
CAPITAL_INICIAL = 10_000_000  # 10M gp, punto de partida razonable para un jugador con algo de capital


# ---------------------------------------------------------------------------
# Grilla de búsqueda
# ---------------------------------------------------------------------------
# Grupos de ítems: 'manual' (item_ids fijos, ej. el par de runas que dio el
# usuario como ejemplo) o 'liquidez' (mismos parámetros dinámicos que ya usa
# el clasificador productivo f2p10_100gp_clasif — se re-deriva el universo
# real en cada checkpoint histórico, no un conjunto fijo) — para poder
# comparar un grupo elegido a mano contra la configuración que YA tiene
# evidencia real de funcionar (ver CLAUDE.md/entrenador.py).
GRUPOS = [
    {
        'nombre': 'runas_cosmic_nature',
        'modo_seleccion': 'manual',
        'item_ids': [561, 564],  # Nature rune, Cosmic rune -- el ejemplo del usuario
    },
    {
        'nombre': 'f2p10_100gp_referencia',
        'modo_seleccion': 'liquidez',
        'n_items': 10, 'solo_f2p': True, 'precio_minimo': 100, 'excluir_item_ids': [2353],
    },
]

TABLA = 'precios_1h'  # ver el diagnóstico previo: la granularidad más confiable de las 3 (datos
                       # más densos, 90d de retención real) -- 5m/6h quedan fuera de esta ronda
VENTANAS_DIAS = [30, 60, 90]
UMBRALES_CLASIF_PCT = [0.3, 0.5, 0.8]


def _rango_backtest(db, tabla):
    """Últimos DIAS_BACKTEST días con datos reales cerrados en `tabla` —
    recorta el último bucket todavía en curso (mismo criterio que
    recolector.rellenar_huecos_al_inicio)."""
    import sqlite3
    conn = sqlite3.connect(db.db_path)
    c = conn.cursor()
    c.execute(f'SELECT MAX(timestamp) FROM {tabla}')
    ultimo = c.fetchone()[0]
    conn.close()
    paso = {'precios_5m': 300, 'precios_1h': 3600, 'precios_6h': 21600}[tabla]
    hasta_ts = (ultimo // paso) * paso - paso
    desde_ts = hasta_ts - DIAS_BACKTEST * 86400
    return desde_ts, hasta_ts


def _precio_en(db, tabla, item_id, ts, columna):
    import sqlite3
    conn = sqlite3.connect(db.db_path)
    c = conn.cursor()
    c.execute(f'SELECT {columna} FROM {tabla} WHERE item_id = ? AND timestamp = ?', (item_id, ts))
    fila = c.fetchone()
    conn.close()
    return fila[0] if fila else None


def _buy_limit(db, item_id):
    import sqlite3
    conn = sqlite3.connect(db.db_path)
    c = conn.cursor()
    c.execute('SELECT buy_limit FROM items WHERE item_id = ?', (item_id,))
    fila = c.fetchone()
    conn.close()
    return fila[0] if fila else None


def _volumen_1h_promedio(db, tabla, item_id, hasta_ts, horas=24):
    import sqlite3
    conn = sqlite3.connect(db.db_path)
    c = conn.cursor()
    c.execute(
        f'SELECT AVG(high_volume + low_volume) FROM {tabla} WHERE item_id = ? AND timestamp BETWEEN ? AND ?',
        (item_id, hasta_ts - horas * 3600, hasta_ts),
    )
    fila = c.fetchone()
    conn.close()
    return fila[0] or 0


def _senales_regresor(db, model_id, tabla, desde_ts, hasta_ts, umbral_subida_pct=1.0):
    """
    Convierte las predicciones walk-forward guardadas por
    entrenar_modelo_global (tabla `predicciones`, modo_evaluacion=
    'walkforward') en señales de compra: item_id, ts_entrada (=
    timestamp - paso, el momento en que se decide comprar),
    ts_salida (=timestamp, cuándo se vende), magnitud (% de suba
    predicho, usado para priorizar cuando el capital no alcanza para
    todas las señales del mismo momento).
    """
    import sqlite3
    paso = {'precios_5m': 300, 'precios_1h': 3600, 'precios_6h': 21600}[tabla]
    conn = sqlite3.connect(db.db_path)
    df = pd.read_sql_query(
        "SELECT item_id, timestamp, predicted_price FROM predicciones "
        "WHERE model_version = ? AND modo_evaluacion = 'walkforward' AND timestamp BETWEEN ? AND ?",
        conn, params=(model_id, desde_ts + paso, hasta_ts + paso),
    )
    conn.close()
    if df.empty:
        return pd.DataFrame(columns=['item_id', 'ts_entrada', 'ts_salida', 'magnitud'])

    filas = []
    for _, fila in df.iterrows():
        ts_salida = int(fila['timestamp'])
        ts_entrada = ts_salida - paso
        precio_entrada = _precio_en(db, tabla, int(fila['item_id']), ts_entrada, 'avg_low_price')
        if not precio_entrada:
            continue
        magnitud = (fila['predicted_price'] - precio_entrada) / precio_entrada * 100
        if magnitud >= umbral_subida_pct:
            filas.append({'item_id': int(fila['item_id']), 'ts_entrada': ts_entrada, 'ts_salida': ts_salida, 'magnitud': magnitud})
    return pd.DataFrame(filas)


def _senales_clasificador(test_df):
    """Igual que _senales_regresor, pero a partir del DataFrame acumulado
    de ejecutar_replay_clasificador (ya trae price_actual = precio de
    entrada, clase_predicha=2 es 'sube')."""
    if test_df.empty:
        return pd.DataFrame(columns=['item_id', 'ts_entrada', 'ts_salida', 'magnitud'])
    candidatos = test_df[test_df['clase_predicha'] == 2].copy()
    if candidatos.empty:
        return pd.DataFrame(columns=['item_id', 'ts_entrada', 'ts_salida', 'magnitud'])
    candidatos['ts_salida'] = candidatos['timestamp_target'].astype(int)
    candidatos['ts_entrada'] = candidatos['ts_salida'] - PASO_SEGUNDOS
    candidatos['item_id'] = candidatos['item_id'].astype(int)
    candidatos['magnitud'] = 1.0  # el clasificador no da magnitud, todas las señales 'sube' pesan igual
    return candidatos[['item_id', 'ts_entrada', 'ts_salida', 'magnitud']]


def backtest_capital_constante(db, tabla, senales, capital_inicial, fraccion_participacion=FRACCION_PARTICIPACION, fraccion_fn=None):
    """
    Simula, con capital REAL y limitado (no capital fantasma: una compra
    que no se puede pagar con lo disponible en ese momento no se ejecuta),
    las compras/ventas que habría hecho un usuario siguiendo `senales`
    (item_id, ts_entrada, ts_salida, magnitud) — precios de compra/venta
    siempre reales (ya ocurridos), nunca predichos, mismo criterio que
    backtest.py.

    Procesa cronológicamente: en cada ts_entrada único, primero CIERRA las
    posiciones cuyo ts_salida sea ese momento (libera capital), después
    ABRE posiciones nuevas según las señales de ese momento — priorizando
    las de mayor `magnitud` cuando el capital no alcanza para todas.
    Nunca hay más de PASO_SEGUNDOS de superposición (horizonte fijo a 1
    paso), así que el orden cierre-antes-que-apertura en el mismo tick ya
    alcanza para no necesitar un libro de órdenes más complejo.

    fraccion_fn (opcional): callable(magnitud) -> fracción del volumen de
    mercado a arriesgar en ESA señal, en vez de la constante
    `fraccion_participacion` fija para todas las señales por igual. Pensado
    para dimensionar por confianza: si `magnitud` es la probabilidad
    softmax de 'sube' (ver entrenador.entrenar_clasificador_direccional,
    columna prob_sube) en vez de un magnitud de retorno, una señal con 0.9
    de confianza puede arriesgar más que una con 0.35 — la idea de
    "meta-labeling" (Lopez de Prado, Advances in Financial Machine
    Learning): el modelo primario decide sube/baja, un segundo score de
    confianza decide CUÁNTO apostar. Default None mantiene el
    comportamiento de siempre (misma fracción fija para toda señal).

    Devuelve (trades_df, resumen) — resumen incluye capital_final,
    ganancia, ganancia_pct, n_trades, win_rate.
    """
    if senales.empty:
        return pd.DataFrame(), {
            'capital_inicial': capital_inicial, 'capital_final': capital_inicial,
            'ganancia': 0.0, 'ganancia_pct': 0.0, 'n_trades': 0, 'win_rate': None,
        }

    capital = capital_inicial
    posiciones_abiertas = []  # dicts: item_id, unidades, precio_entrada, ts_salida
    trades = []

    momentos = sorted(set(senales['ts_entrada']) | set(senales['ts_salida']))
    buy_limits_cache = {}
    volumen_cache = {}

    for t in momentos:
        # 1) Cerrar posiciones que vencen en este momento.
        siguen_abiertas = []
        for pos in posiciones_abiertas:
            if pos['ts_salida'] != t:
                siguen_abiertas.append(pos)
                continue
            precio_venta = _precio_en(db, tabla, pos['item_id'], t, 'avg_high_price')
            if not precio_venta:
                continue  # sin dato real de venta en ese momento, se pierde la posición (no se puede liquidar)
            impuesto = calcular_impuesto_ge(precio_venta, pos['item_id'])
            ingreso = precio_venta * pos['unidades'] - impuesto * pos['unidades']
            costo = pos['precio_entrada'] * pos['unidades']
            capital += ingreso
            trades.append({
                'item_id': pos['item_id'], 'ts_entrada': pos['ts_entrada'], 'ts_salida': t,
                'unidades': pos['unidades'], 'precio_entrada': pos['precio_entrada'],
                'precio_salida': precio_venta, 'ganancia': ingreso - costo,
            })
        posiciones_abiertas = siguen_abiertas

        # 2) Abrir posiciones nuevas según las señales de este momento,
        # priorizando mayor magnitud cuando el capital no alcanza para todas.
        señales_de_t = senales[senales['ts_entrada'] == t].sort_values('magnitud', ascending=False)
        for _, señal in señales_de_t.iterrows():
            item_id = int(señal['item_id'])
            precio_compra = _precio_en(db, tabla, item_id, t, 'avg_low_price')
            if not precio_compra or capital < precio_compra:
                continue
            if item_id not in buy_limits_cache:
                buy_limits_cache[item_id] = _buy_limit(db, item_id)
            buy_limit = buy_limits_cache[item_id]
            if not buy_limit:
                continue
            clave_vol = (item_id, t)
            if clave_vol not in volumen_cache:
                volumen_cache[clave_vol] = _volumen_1h_promedio(db, tabla, item_id, t)
            volumen_1h_promedio = volumen_cache[clave_vol]

            fraccion_efectiva = fraccion_fn(señal['magnitud']) if fraccion_fn is not None else fraccion_participacion
            tope_mercado = min(buy_limit, volumen_1h_promedio * fraccion_efectiva)
            tope_capital = capital // precio_compra
            unidades = int(min(tope_mercado, tope_capital))
            if unidades < 1:
                continue

            costo = unidades * precio_compra
            capital -= costo
            posiciones_abiertas.append({
                'item_id': item_id, 'unidades': unidades, 'precio_entrada': precio_compra,
                'ts_entrada': t, 'ts_salida': int(señal['ts_salida']),
            })

    # Posiciones que quedaron abiertas al final del rango (su ts_salida cae
    # fuera de `momentos`, ej. la última señal del backtest): se liquidan
    # al precio real si existe, si no se pierden (no hay forma de saber
    # qué pasó después del rango simulado).
    for pos in posiciones_abiertas:
        precio_venta = _precio_en(db, tabla, pos['item_id'], pos['ts_salida'], 'avg_high_price')
        if not precio_venta:
            continue
        impuesto = calcular_impuesto_ge(precio_venta, pos['item_id'])
        ingreso = precio_venta * pos['unidades'] - impuesto * pos['unidades']
        costo = pos['precio_entrada'] * pos['unidades']
        capital += ingreso
        trades.append({
            'item_id': pos['item_id'], 'ts_entrada': pos['ts_entrada'], 'ts_salida': pos['ts_salida'],
            'unidades': pos['unidades'], 'precio_entrada': pos['precio_entrada'],
            'precio_salida': precio_venta, 'ganancia': ingreso - costo,
        })

    trades_df = pd.DataFrame(trades)
    resumen = {
        'capital_inicial': capital_inicial,
        'capital_final': capital,
        'ganancia': capital - capital_inicial,
        'ganancia_pct': (capital - capital_inicial) / capital_inicial * 100,
        'n_trades': len(trades_df),
        'win_rate': float((trades_df['ganancia'] > 0).mean()) if not trades_df.empty else None,
    }
    return trades_df, resumen


def _kwargs_grupo(grupo):
    if grupo['modo_seleccion'] == 'manual':
        return {'item_ids': grupo['item_ids']}
    return {
        'n_items': grupo['n_items'], 'solo_f2p': grupo['solo_f2p'],
        'precio_minimo': grupo.get('precio_minimo'), 'excluir_item_ids': grupo.get('excluir_item_ids'),
    }


def probar_regresor(db, grupo, ventana_dias, desde_ts, hasta_ts):
    """
    ejecutar_replay_modelo() (a diferencia de ejecutar_replay_clasificador,
    que es parametrizable directo) necesita una fila real en modelos_config
    para saber qué entrenar en cada checkpoint -- se crea acá con
    cadencia='manual' (no compite por el tope de modelos activos ni se
    reentrena sola) y se borra al final del walk-forward: es un modelo de
    prueba, no debe quedar en "Mis modelos" de la app. Su historial en
    model_metrics SÍ se conserva (mismo criterio que
    OSRSBaseDatos.eliminar_modelo_config) -- queda como registro de la
    corrida de búsqueda si hace falta auditarla después.
    """
    model_id = f"busqueda_{grupo['nombre']}_regresor_v{int(ventana_dias)}d"
    kwargs = _kwargs_grupo(grupo)
    db.crear_modelo_config(
        model_id=model_id, nombre=f"[búsqueda] {grupo['nombre']} regresor {ventana_dias}d",
        tipo='regresor', cadencia='manual', modo_seleccion=grupo['modo_seleccion'],
        item_ids=kwargs.get('item_ids'), n_items=kwargs.get('n_items'), solo_f2p=kwargs.get('solo_f2p') or False,
        precio_minimo=kwargs.get('precio_minimo'), excluir_item_ids=kwargs.get('excluir_item_ids'),
        tabla=TABLA, ventana_dias=ventana_dias,
    )
    t0 = time.time()
    # desde_ts/hasta_ts explícitos: sin esto, ejecutar_replay_modelo
    # recorre TODO el historial disponible del ítem (pensado para el
    # walk-forward inicial al crear un modelo, ver hilo_walkforward.py) --
    # acá solo interesan los checkpoints dentro de la ventana del
    # backtest, varias veces menos trabajo.
    ejecutar_replay_modelo(db, model_id, desde_ts=desde_ts, hasta_ts=hasta_ts)
    duracion = time.time() - t0

    senales = _senales_regresor(db, model_id, TABLA, desde_ts, hasta_ts)
    trades, resumen = backtest_capital_constante(db, TABLA, senales, CAPITAL_INICIAL)
    db.eliminar_modelo_config(model_id)  # limpia el registro de búsqueda, conserva el historial en model_metrics
    return {
        'grupo': grupo['nombre'], 'tipo': 'regresor', 'ventana_dias': ventana_dias, 'umbral_pct': None,
        'duracion_s': round(duracion, 1), **resumen,
    }


def probar_clasificador(db, grupo, ventana_dias, umbral_pct, desde_ts, hasta_ts):
    model_id = f"busqueda_{grupo['nombre']}_clasif_v{int(ventana_dias)}d_u{umbral_pct}"
    kwargs = _kwargs_grupo(grupo)

    t0 = time.time()
    test_acumulado = ejecutar_replay_clasificador(
        db, desde_ts, hasta_ts, model_name=model_id, umbral_pct=umbral_pct,
        tabla=TABLA, ventana_dias=ventana_dias, **kwargs,
    )
    duracion = time.time() - t0

    senales = _senales_clasificador(test_acumulado)
    trades, resumen = backtest_capital_constante(db, TABLA, senales, CAPITAL_INICIAL)
    return {
        'grupo': grupo['nombre'], 'tipo': 'clasificador', 'ventana_dias': ventana_dias, 'umbral_pct': umbral_pct,
        'duracion_s': round(duracion, 1), **resumen,
    }


def ejecutar_busqueda():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s |--| %(levelname)s |--| %(message)s')
    db = OSRSBaseDatos(DB_PATH)
    desde_ts, hasta_ts = _rango_backtest(db, TABLA)
    logging.info(
        f"=== Búsqueda de hiperparámetros: {datetime.fromtimestamp(desde_ts, tz=timezone.utc)} "
        f"a {datetime.fromtimestamp(hasta_ts, tz=timezone.utc)} UTC ({DIAS_BACKTEST} días) ==="
    )

    resultados = []
    total_pruebas = len(GRUPOS) * (len(VENTANAS_DIAS) + len(VENTANAS_DIAS) * len(UMBRALES_CLASIF_PCT))
    n = 0
    for grupo in GRUPOS:
        for ventana_dias in VENTANAS_DIAS:
            n += 1
            logging.info(f"--- Prueba {n}/{total_pruebas}: {grupo['nombre']} | regresor | ventana={ventana_dias}d ---")
            try:
                resultado = probar_regresor(db, grupo, ventana_dias, desde_ts, hasta_ts)
                resultados.append(resultado)
                pd.DataFrame(resultados).to_csv(CSV_RESULTADOS, index=False)
                logging.info(f"Resultado: {resultado}")
            except Exception as e:
                logging.error(f"Prueba {n} falló: {e}")

            for umbral_pct in UMBRALES_CLASIF_PCT:
                n += 1
                logging.info(f"--- Prueba {n}/{total_pruebas}: {grupo['nombre']} | clasificador | ventana={ventana_dias}d | umbral={umbral_pct}% ---")
                try:
                    resultado = probar_clasificador(db, grupo, ventana_dias, umbral_pct, desde_ts, hasta_ts)
                    resultados.append(resultado)
                    pd.DataFrame(resultados).to_csv(CSV_RESULTADOS, index=False)
                    logging.info(f"Resultado: {resultado}")
                except Exception as e:
                    logging.error(f"Prueba {n} falló: {e}")

    df = pd.DataFrame(resultados)
    df.to_csv(CSV_RESULTADOS, index=False)
    logging.info(f"\n=== Búsqueda completa: {len(df)} pruebas, resultados en {CSV_RESULTADOS} ===")
    if not df.empty:
        ranking = df.sort_values('ganancia', ascending=False)
        logging.info("\n" + ranking.to_string(index=False))
    return df


if __name__ == '__main__':
    ejecutar_busqueda()
