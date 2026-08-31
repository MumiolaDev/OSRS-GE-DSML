"""
busqueda_significancia.py — responde la pregunta directa del usuario: ¿el
método (clasificador direccional F2P, la MISMA configuración que ya corre
en producción como f2p10_100gp_clasif — ver entrenador.py) predice qué y
cuándo comprar/vender con una seguridad mejor que el azar? A diferencia de
busqueda_calibrada.py (8 ítems elegidos a mano, 2 semanas, sin ningún test
de significancia), acá:

1. Ventana mucho más grande (DIAS_BACKTEST=90 en vez de 14) y solo
   precios_1h (pedido explícito del usuario) — walk-forward real, un
   reentrenamiento por checkpoint horario, exactamente como correría en
   vivo (recolector.job_horario).
2. El universo NO se elige a mano: es EXACTAMENTE el de producción
   (n_items=10, solo_f2p=True, precio_minimo=100, excluir_item_ids=[2353]),
   re-derivado en cada checkpoint vía obtener_top_items_liquidez_hasta — así
   esta prueba mide el método tal como el usuario ya lo está usando, no una
   selección optimista de ítems.
3. Significancia formal por TEST DE PERMUTACIÓN (Monte Carlo), no solo
   win_rate descriptivo: en cada checkpoint, el "azar" elige la MISMA
   cantidad de compras que el modelo predijo 'sube' en ese momento, pero
   sorteadas entre los MISMOS candidatos (mismo universo líquido F2P de ese
   checkpoint) — así una diferencia de resultado solo puede deberse a QUÉ
   ítems eligió el modelo, no a cuántas veces operó o sobre qué universo.
   p_valor = fracción de sorteos al azar que igualan o superan el resultado
   real del modelo (profit_total y win_rate por separado). Es más apropiado
   acá que un binomial test contra 50%: una "victoria" no vale lo mismo en
   gp para todos los trades (montos de posición muy distintos por ítem), así
   que comparar contra un azar CON LA MISMA MECÁNICA DE APUESTA es lo que
   real y limpiamente aísla "¿el modelo elige mejor que elegir cualquiera
   del mismo grupo?".

Resumable a mano: escribe resultados_significancia_predicciones.csv +
resultados_significancia_progreso.csv fila por fila conforme corre (un
checkpoint por fila en progreso) — un corte a mitad de camino no pierde las
horas ya calculadas, correr el script de nuevo detecta lo ya hecho vía el
CSV de progreso y sigue desde ahí. Nunca escribe en
modelos_config del usuario ni toca ningún .pkl productivo (model_name
distinto al productivo, solo dejar historial en model_metrics).
"""
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from base_de_datos import OSRSBaseDatos
from entrenador import entrenar_clasificador_direccional
from replay_historico import calcular_checkpoints
from metricas import calcular_impuesto_ge, dimensionar_oportunidad
from backtest import _buy_limits, _simular_trades_clasificador

DB_PATH = 'data/osrs_ge.db'
TABLA = 'precios_1h'
PASO_SEGUNDOS = 3600

DIAS_BACKTEST = 90          # "ventana mas grande", pedido explícito del usuario (antes: 14 días)
N_ITEMS = 10
SOLO_F2P = True
PRECIO_MINIMO = 100
EXCLUIR_ITEM_IDS = [2353]   # Steel bar — mismo universo EXACTO que f2p10_100gp_clasif (ver entrenador.py)
UMBRAL_PCT = None           # None = usa el default de producción (UMBRAL_CLASIF_PCT=0.5)
MODEL_ID = 'significancia_f2p10_100gp_90d'  # nombre propio, no pisa model_metrics de producción
FRACCION_PARTICIPACION = 0.15

N_PERMUTACIONES = 500
SEED = 42

CSV_PREDICCIONES = 'resultados_significancia_predicciones.csv'
CSV_PROGRESO = 'resultados_significancia_progreso.csv'
CSV_RESUMEN = 'resultados_significancia_resumen.csv'
CSV_NULOS = 'resultados_significancia_nulos.csv'


# ---------------------------------------------------------------------------
# Fase 1: walk-forward real sobre la ventana grande (resumable)
# ---------------------------------------------------------------------------

def _rango_backtest(db, tabla, dias):
    import sqlite3
    conn = sqlite3.connect(db.db_path)
    c = conn.cursor()
    c.execute(f'SELECT MAX(timestamp) FROM {tabla}')
    ultimo = c.fetchone()[0]
    conn.close()
    hasta_ts = (ultimo // PASO_SEGUNDOS) * PASO_SEGUNDOS - PASO_SEGUNDOS
    desde_ts = hasta_ts - dias * 86400
    return desde_ts, hasta_ts


def _checkpoints_pendientes(desde_ts, hasta_ts):
    todos = [ts for ts, tipo in calcular_checkpoints(desde_ts, hasta_ts) if tipo == 'horario']
    hechos = set()
    if Path(CSV_PROGRESO).exists():
        prog = pd.read_csv(CSV_PROGRESO)
        hechos = set(prog['ts'].tolist())
    return [ts for ts in todos if ts not in hechos], todos


def ejecutar_walkforward(db, desde_ts, hasta_ts):
    pendientes, todos = _checkpoints_pendientes(desde_ts, hasta_ts)
    logging.info(
        f"=== Walk-forward significancia: {len(todos)} checkpoints horarios totales, "
        f"{len(pendientes)} pendientes ({len(todos) - len(pendientes)} ya hechos, resumido) ==="
    )
    escribir_header_pred = not Path(CSV_PREDICCIONES).exists()
    escribir_header_prog = not Path(CSV_PROGRESO).exists()

    for i, ts in enumerate(pendientes):
        t0 = time.time()
        estado, n_filas = 'ok', 0
        try:
            kwargs = {}
            if UMBRAL_PCT is not None:  # None debe dejar el default de la función (UMBRAL_CLASIF_PCT), no pisarlo con None
                kwargs['umbral_pct'] = UMBRAL_PCT
            _, test = entrenar_clasificador_direccional(
                db, n_items=N_ITEMS, solo_f2p=SOLO_F2P, precio_minimo=PRECIO_MINIMO,
                excluir_item_ids=EXCLUIR_ITEM_IDS,
                model_name=MODEL_ID, ahora_ts=ts, modo_evaluacion='walkforward', tabla=TABLA, **kwargs,
            )
            if test is not None and not test.empty:
                n_filas = len(test)
                cols = ['item_id', 'timestamp_target', 'price_actual', 'target',
                        'clase_predicha', 'clase_real', 'prob_sube']
                test[cols].to_csv(CSV_PREDICCIONES, mode='a', header=escribir_header_pred, index=False)
                escribir_header_pred = False
        except Exception as e:
            logging.error(f"checkpoint ts={ts} falló: {e}")
            estado = 'error'

        pd.DataFrame([{'ts': ts, 'n_filas': n_filas, 'estado': estado,
                        'duracion_s': round(time.time() - t0, 2)}]).to_csv(
            CSV_PROGRESO, mode='a', header=escribir_header_prog, index=False)
        escribir_header_prog = False

        if (i + 1) % 25 == 0 or (i + 1) == len(pendientes):
            logging.info(f"Walk-forward: {i + 1}/{len(pendientes)} checkpoints nuevos procesados")

    logging.info("=== Walk-forward significancia completo ===")


# ---------------------------------------------------------------------------
# Fase 2: simulación cacheada (para poder correr cientos de permutaciones
# sin repetir cientos de veces las mismas queries a la DB)
# ---------------------------------------------------------------------------

def _preparar_cache(db, test_acumulado, tabla=TABLA):
    item_ids = test_acumulado['item_id'].astype(int).unique().tolist()
    buy_limits = _buy_limits(db, item_ids)
    precios_multi = db.obtener_precios_multi(item_ids, tabla)
    series = {
        iid: grupo.sort_values('timestamp').set_index('timestamp')
        for iid, grupo in precios_multi.groupby('item_id')
    }
    return buy_limits, series


def _simular_desde_cache(candidatos, buy_limits, series, fraccion=FRACCION_PARTICIPACION, paso=PASO_SEGUNDOS):
    """Misma lógica financiera EXACTA que backtest._simular_trades_clasificador
    (mismo impuesto GE, mismo dimensionamiento por volumen/buy_limit) pero
    parametrizada con precios/buy_limits ya cacheados en memoria — evita
    volver a golpear la DB en cada una de las N_PERMUTACIONES corridas."""
    profits = []
    for fila in candidatos.itertuples(index=False):
        item_id = int(fila.item_id)
        t_venta = int(fila.timestamp_target)
        t_compra = t_venta - paso
        avg_low = float(fila.price_actual)

        serie = series.get(item_id)
        buy_limit = buy_limits.get(item_id)
        if serie is None or not buy_limit or not avg_low or t_venta not in serie.index:
            continue
        precio_venta_real = serie.at[t_venta, 'avg_high_price']
        if not precio_venta_real:
            continue

        ventana = serie.loc[t_compra - 23 * paso: t_compra, ['high_volume', 'low_volume']]
        vol_24h = ventana.sum().sum() if not ventana.empty else 0.0
        volumen_1h_promedio = vol_24h / 24

        impuesto = calcular_impuesto_ge(precio_venta_real, item_id)
        margen_neto = precio_venta_real - avg_low - impuesto
        profit = dimensionar_oportunidad(margen_neto, buy_limit, volumen_1h_promedio, 1, fraccion)
        if profit is not None:
            profits.append(profit)
    return profits


def _resumen_de_profits(profits):
    n = len(profits)
    if n == 0:
        return {'n_trades': 0, 'profit_total': 0.0, 'win_rate': None, 'profit_promedio': None}
    arr = np.array(profits)
    return {
        'n_trades': n, 'profit_total': float(arr.sum()),
        'win_rate': float((arr > 0).mean()), 'profit_promedio': float(arr.mean()),
    }


# ---------------------------------------------------------------------------
# Fase 3: test de permutación
# ---------------------------------------------------------------------------

def _wilson_ci(k, n, z=1.96):
    """Intervalo de confianza de Wilson (95% con z=1.96 por default) para una
    proporción — más confiable que el intervalo normal ingenuo cuando n es
    chico o la proporción está cerca de 0/1 (ver Wilson, 1927)."""
    if n == 0:
        return None, None
    p = k / n
    denom = 1 + z ** 2 / n
    centro = p + z ** 2 / (2 * n)
    radio = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2))
    return round((centro - radio) / denom, 4), round((centro + radio) / denom, 4)


def permutacion_test(test_acumulado, buy_limits, series, n_permutaciones=N_PERMUTACIONES, seed=SEED):
    rng = np.random.default_rng(seed)
    grupos_info = []
    for ts, g in test_acumulado.groupby('timestamp_target'):
        k = int((g['clase_predicha'] == 2).sum())
        if k > 0:
            grupos_info.append((g, k))

    filas_nulas = []
    for p in range(n_permutaciones):
        piezas = []
        for g, k in grupos_info:
            elegidos = g.sample(n=k, random_state=int(rng.integers(0, 2**31 - 1)))
            piezas.append(elegidos)
        candidatos_random = pd.concat(piezas, ignore_index=True) if piezas else pd.DataFrame(columns=test_acumulado.columns)
        profits = _simular_desde_cache(candidatos_random, buy_limits, series)
        filas_nulas.append(_resumen_de_profits(profits))
        if (p + 1) % 100 == 0:
            logging.info(f"Permutación {p + 1}/{n_permutaciones}")

    return pd.DataFrame(filas_nulas)


# ---------------------------------------------------------------------------
# Orquestación
# ---------------------------------------------------------------------------

def ejecutar_todo():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s |--| %(levelname)s |--| %(message)s')
    db = OSRSBaseDatos(DB_PATH)
    desde_ts, hasta_ts = _rango_backtest(db, TABLA, DIAS_BACKTEST)
    logging.info(
        f"=== Búsqueda de significancia: universo producción (n_items={N_ITEMS}, solo_f2p={SOLO_F2P}, "
        f"precio_minimo={PRECIO_MINIMO}, excluir={EXCLUIR_ITEM_IDS}), {DIAS_BACKTEST} días, {TABLA} ==="
    )

    ejecutar_walkforward(db, desde_ts, hasta_ts)

    if not Path(CSV_PREDICCIONES).exists():
        logging.error("Sin predicciones acumuladas — el walk-forward no generó ningún checkpoint válido.")
        return

    test_acumulado = pd.read_csv(CSV_PREDICCIONES)
    logging.info(f"Predicciones acumuladas: {len(test_acumulado)} filas, {test_acumulado['timestamp_target'].nunique()} checkpoints")

    trades_modelo, resumen_modelo = _simular_trades_clasificador(db, test_acumulado, FRACCION_PARTICIPACION, TABLA)
    logging.info(f"[modelo real] {resumen_modelo}")

    buy_limits, series = _preparar_cache(db, test_acumulado, TABLA)
    nulos = permutacion_test(test_acumulado, buy_limits, series)
    nulos.to_csv(CSV_NULOS, index=False)

    profit_modelo = resumen_modelo['profit_total']
    winrate_modelo = resumen_modelo['win_rate'] or 0.0
    p_profit = (1 + (nulos['profit_total'] >= profit_modelo).sum()) / (1 + len(nulos))
    p_winrate = (1 + (nulos['win_rate'].fillna(0) >= winrate_modelo).sum()) / (1 + len(nulos))

    n_trades = resumen_modelo['n_trades']
    n_wins = int(round((resumen_modelo['win_rate'] or 0) * n_trades))
    ci_low, ci_high = _wilson_ci(n_wins, n_trades)

    try:
        from scipy.stats import binomtest
        p_binomial = binomtest(n_wins, n_trades, 0.5, alternative='greater').pvalue if n_trades > 0 else None
    except ImportError:
        p_binomial = None

    resumen_final = {
        'dias_backtest': DIAS_BACKTEST, 'n_checkpoints': test_acumulado['timestamp_target'].nunique(),
        'n_trades_modelo': n_trades, 'profit_total_modelo': profit_modelo,
        'win_rate_modelo': resumen_modelo['win_rate'], 'win_rate_ci95_low': ci_low, 'win_rate_ci95_high': ci_high,
        'profit_promedio_modelo': resumen_modelo['profit_promedio'],
        'n_permutaciones': len(nulos),
        'profit_total_nulo_media': nulos['profit_total'].mean(), 'profit_total_nulo_mediana': nulos['profit_total'].median(),
        'win_rate_nulo_media': nulos['win_rate'].mean(),
        'p_valor_permutacion_profit': p_profit, 'p_valor_permutacion_winrate': p_winrate,
        'p_valor_binomial_vs_50pct': p_binomial,
    }
    pd.DataFrame([resumen_final]).to_csv(CSV_RESUMEN, index=False)
    trades_modelo.to_csv('resultados_significancia_trades.csv', index=False)

    logging.info("=== RESUMEN FINAL ===")
    for k, v in resumen_final.items():
        logging.info(f"{k}: {v}")

    return resumen_final


if __name__ == '__main__':
    ejecutar_todo()
