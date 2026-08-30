"""
busqueda_calibrada.py — fase 2 del ensemble (ver busqueda_ensemble.py):
esa corrida encontró que un umbral FIJO (0.8-1.2%) compartido entre ítems de
volatilidad muy distinta deja casi todo el backtest sin señal, porque el
regresor nunca predijo un movimiento tan grande para ítems estables como
Runite ore o Adamantite bar. Acá se resuelven tres preguntas separadas, cada
una atacando una parte de "qué comprar, en qué cantidad, y cuándo vender":

1. UMBRAL CALIBRADO POR ÍTEM (qué comprar): en vez de una constante
   compartida, el umbral de cada ítem es el percentil P (PERCENTIL_UMBRAL,
   default 80) de la distribución real de |log-retorno horario| DE ESE
   ÍTEM, calculado solo con datos ANTERIORES al inicio del backtest (mismo
   criterio walk-forward que el resto del proyecto — calibrar con datos que
   incluyan el propio rango de evaluación sería fuga de información). Ocho
   ítems individuales, elegidos para cubrir un espectro real de volatilidad
   (ver ITEMS) en vez de, como la corrida anterior, terminar con 3 ítems
   que resultaron ser casi todos igual de estables sin saberlo de antemano.

2. DIMENSIONAMIENTO POR CONFIANZA (en qué cantidad): en vez de arriesgar
   la misma fracción fija del volumen de mercado en toda señal 'sube' por
   igual, se usa la probabilidad softmax de esa clase (entrenador.
   entrenar_clasificador_direccional ahora expone `prob_sube`) para escalar
   el tamaño de la posición — una señal con 0.9 de confianza arriesga más
   que una con 0.35. Es la idea de "meta-labeling" de Lopez de Prado
   (Advances in Financial Machine Learning, 2018): un modelo primario
   decide la dirección, un segundo score de confianza decide cuánto
   apostar. Se compara contra el dimensionamiento fijo de siempre.

3. SALIDA POR TRIPLE BARRERA (cuándo vender): en vez de vender siempre
   exactamente 1 hora después (el horizonte fijo de todas las corridas
   anteriores), se prueba la salida con 3 barreras (mismo Lopez de Prado):
   take-profit (el propio umbral calibrado del ítem), stop-loss (una
   fracción del take-profit, ver STOP_LOSS_FRACCION) y timeout
   (TIMEOUT_PASOS horas) — la que se cruce primero, contra precios REALES
   ya ocurridos (nunca predichos). No hace falta entrenar nada nuevo para
   esto: son las mismas señales de (1), solo cambia CUÁNDO se resuelve la
   venta — así que es barato, se corre sobre las mismas corridas.

Ambiente de testeo aparte, mismo criterio que busqueda_ensemble.py /
busqueda_hiperparametros.py: nunca escribe en modelos_config del usuario ni
toca ningún .pkl productivo. Resultados a resultados_calibrada.csv fila por
fila (sobrevive un corte a mitad de camino).
"""
import logging
import sqlite3
import time

import numpy as np
import pandas as pd

from base_de_datos import OSRSBaseDatos
from replay_historico import ejecutar_replay_modelo, ejecutar_replay_clasificador
from busqueda_hiperparametros import (
    _rango_backtest, _precio_en, _senales_regresor, _senales_clasificador,
    backtest_capital_constante, TABLA, CAPITAL_INICIAL, FRACCION_PARTICIPACION,
)

DB_PATH = 'data/osrs_ge.db'
CSV_RESULTADOS = 'resultados_calibrada.csv'
VENTANA_DIAS = 90
PERCENTIL_UMBRAL = 80          # ver punto 1 del docstring
STOP_LOSS_FRACCION = 0.5       # stop-loss = -0.5 * take-profit -- riesgo/beneficio 2:1
TIMEOUT_PASOS = 6              # máximo 6 horas en posición antes de forzar la salida

# Ocho ítems individuales (precio > 1000gp, volumen alto — mismo criterio
# que la corrida anterior), elegidos de resumen_actual real por
# volatilidad_30d_pct para cubrir un espectro real en vez de adivinar:
# los primeros 3 son casi sin movimiento (arrastrados de la corrida
# anterior, permiten comparar directo "antes vs. después de calibrar"),
# los siguientes 3 son de volatilidad media (7-9%, herblore/runecrafting/
# potions, todos extremadamente líquidos: 200k-1M unidades/24h), los
# últimos 2 son de alta volatilidad (materiales de crafting/construcción).
ITEMS = [
    {'item_id': 451,   'nombre': 'Runite ore',        'volatilidad_30d_pct_ref': 0.40},
    {'item_id': 2361,  'nombre': 'Adamantite bar',     'volatilidad_30d_pct_ref': 0.93},
    {'item_id': 536,   'nombre': 'Dragon bones',       'volatilidad_30d_pct_ref': 1.97},
    {'item_id': 573,   'nombre': 'Air orb',            'volatilidad_30d_pct_ref': 7.75},
    {'item_id': 12625, 'nombre': 'Stamina potion(4)',  'volatilidad_30d_pct_ref': 7.11},
    {'item_id': 263,   'nombre': 'Kwuarm',             'volatilidad_30d_pct_ref': 9.29},
    {'item_id': 31475, 'nombre': 'Bolt of canvas',     'volatilidad_30d_pct_ref': 40.11},
    {'item_id': 22603, 'nombre': 'Basalt',             'volatilidad_30d_pct_ref': 106.18},
]


def diagnosticar_saltos_anomalos(db, item_id, tabla, desde_ts, hasta_ts):
    """
    Detector de ítems cuya volatilidad viene de eventos RAROS (iliquidez
    puntual o un dato anómalo en la API) en vez de movimiento repetible que
    un modelo pueda aprender -- encontrado revisando por qué Basalt (ver
    docs/investigacion_calidad_modelos.md) dio +112% de ganancia en el
    backtest de esta misma corrida: su historial tiene un salto de 339% en
    una sola hora, con avg_low_price (4372) mayor que avg_high_price (1933)
    ese mismo período, sobre apenas 7 unidades de volumen -- un precio que
    ningún jugador real podría haber ejecutado a esa escala.

    Un piso de volumen NO alcanza para filtrar esto: el volumen horario
    promedio de Basalt (8277) es más alto que el de Bolt of canvas (1267),
    que sí dio resultados creíbles. Lo que sí separa limpio a los dos es la
    razón entre el salto máximo histórico y el salto TÍPICO (mediana) del
    propio ítem -- ver la tabla en docs/investigacion_calidad_modelos.md:
    los 6 ítems "limpios" de esta corrida están entre 4.7x y 13.2x: Basalt
    da 182x, Bolt of canvas 78x (volatilidad real pero con momentos de
    iliquidez, resultados a tomar con cautela moderada).

    Devuelve (ratio, mediana_pct, maximo_pct) — ratio=None si no hay
    suficiente historial. Diagnóstico, no filtro automático: no está
    enganchado a ninguna selección de ítems en producción todavía, es una
    herramienta para revisar un ítem candidato antes de confiar en su
    backtest.
    """
    conn = sqlite3.connect(db.db_path)
    df = pd.read_sql_query(
        f"SELECT timestamp, avg_low_price FROM {tabla} "
        "WHERE item_id = ? AND timestamp BETWEEN ? AND ? AND avg_low_price IS NOT NULL ORDER BY timestamp",
        conn, params=(item_id, desde_ts, hasta_ts),
    )
    conn.close()
    ret = (df['avg_low_price'].pct_change().abs() * 100).dropna()
    if len(ret) < 30 or ret.median() == 0:
        return None, None, None
    mediana, maximo = float(ret.median()), float(ret.max())
    return round(maximo / mediana, 1), round(mediana, 3), round(maximo, 2)


def _umbral_calibrado(db, item_id, tabla, ventana_dias, hasta_ts, percentil):
    """
    Percentil `percentil` de |log-retorno| por período de ESTE ítem,
    calculado solo con datos ANTERIORES a `hasta_ts` (el inicio del
    backtest) -- calibrar con datos que ya incluyan el rango de evaluación
    sería fuga de información hacia el propio umbral que se está probando.
    None si no hay suficiente historial (< 30 retornos) para un percentil
    confiable.
    """
    conn = sqlite3.connect(db.db_path)
    desde_ts = int(hasta_ts - ventana_dias * 86400)
    df = pd.read_sql_query(
        f"SELECT timestamp, avg_low_price FROM {tabla} "
        "WHERE item_id = ? AND timestamp >= ? AND timestamp < ? AND avg_low_price IS NOT NULL "
        "ORDER BY timestamp",
        conn, params=(item_id, desde_ts, hasta_ts),
    )
    conn.close()
    if len(df) < 30:
        return None
    log_ret = np.log(df['avg_low_price']).diff().dropna()
    return round(float(np.percentile(log_ret.abs(), percentil)) * 100, 4)


def _resolver_ts_salida_triple_barrera(db, tabla, item_id, ts_entrada, precio_entrada, take_profit_pct, stop_loss_pct, timeout_pasos, paso):
    """
    Escanea los precios REALES posteriores a ts_entrada (nunca predichos)
    hasta timeout_pasos períodos, y devuelve el timestamp en que se cruzó
    la primera barrera: take-profit, stop-loss, o el propio timeout si no
    se cruzó ninguna. None si no hay ningún precio posterior disponible
    (borde del rango de datos).
    """
    conn = sqlite3.connect(db.db_path)
    c = conn.cursor()
    c.execute(
        f"SELECT timestamp, avg_high_price FROM {tabla} "
        "WHERE item_id = ? AND timestamp > ? AND timestamp <= ? AND avg_high_price IS NOT NULL "
        "ORDER BY timestamp",
        (item_id, ts_entrada, ts_entrada + timeout_pasos * paso),
    )
    filas = c.fetchall()
    conn.close()
    if not filas:
        return None
    for ts, precio_venta in filas:
        retorno_pct = (precio_venta - precio_entrada) / precio_entrada * 100
        if retorno_pct >= take_profit_pct or retorno_pct <= stop_loss_pct:
            return int(ts)
    return int(filas[-1][0])  # ninguna barrera cruzada -> timeout, sale al último precio real disponible


def _senales_triple_barrera(db, tabla, señales_base, take_profit_pct, stop_loss_fraccion, timeout_pasos):
    """Re-resuelve el ts_salida de cada señal de señales_base (item_id,
    ts_entrada, magnitud) vía triple barrera en vez del horizonte fijo de 1
    paso -- el resto (magnitud para prioridad, precio de entrada/salida) lo
    sigue resolviendo backtest_capital_constante como siempre, así que no
    hace falta tocar el motor de backtest."""
    if señales_base.empty:
        return señales_base
    paso = {'precios_5m': 300, 'precios_1h': 3600, 'precios_6h': 21600}[tabla]
    stop_loss_pct = -take_profit_pct * stop_loss_fraccion
    filas = []
    for _, s in señales_base.iterrows():
        item_id, ts_entrada = int(s['item_id']), int(s['ts_entrada'])
        precio_entrada = _precio_en(db, tabla, item_id, ts_entrada, 'avg_low_price')
        if not precio_entrada:
            continue
        ts_salida = _resolver_ts_salida_triple_barrera(
            db, tabla, item_id, ts_entrada, precio_entrada, take_profit_pct, stop_loss_pct, timeout_pasos, paso,
        )
        if ts_salida is None:
            continue
        filas.append({'item_id': item_id, 'ts_entrada': ts_entrada, 'ts_salida': ts_salida, 'magnitud': s['magnitud']})
    return pd.DataFrame(filas)


def _fraccion_por_confianza(prob_sube):
    """fraccion_fn para backtest_capital_constante: escala la fracción base
    de volumen arriesgado según la probabilidad softmax de 'sube' -- en
    p=1/3 (borde de ser la clase predicha en un problema de 3 clases) se
    arriesga 0.83x lo normal, en p=1.0 se arriesga 1.5x. Nunca negativo ni
    desmedido: acotado a ese rango por construcción de prob_sube en [0,1]."""
    return FRACCION_PARTICIPACION * (0.5 + prob_sube)


def _entrenar_temporal(db, model_id, item_id, tipo, desde_ts, hasta_ts, umbral_pct=None):
    """Modelo temporal de un solo ítem, walk-forward acotado al rango del
    backtest, se borra al terminar -- mismo patrón que busqueda_ensemble.py."""
    db.crear_modelo_config(
        model_id=model_id, nombre=f"[calibrada] {model_id}", tipo=tipo,
        cadencia='manual', modo_seleccion='manual', item_ids=[item_id],
        tabla=TABLA, ventana_dias=VENTANA_DIAS, umbral_pct=umbral_pct,
    )
    t0 = time.time()
    ejecutar_replay_modelo(db, model_id, desde_ts=desde_ts, hasta_ts=hasta_ts)
    duracion = time.time() - t0
    db.eliminar_modelo_config(model_id)
    return duracion


def probar_item(db, item, desde_ts, hasta_ts):
    item_id, nombre = item['item_id'], item['nombre']
    filas = []

    umbral_pct = _umbral_calibrado(db, item_id, TABLA, VENTANA_DIAS, desde_ts, PERCENTIL_UMBRAL)
    if umbral_pct is None or umbral_pct <= 0:
        logging.warning(f"{nombre} ({item_id}): sin historial suficiente para calibrar umbral, se omite.")
        return filas
    logging.info(f"--- {nombre} ({item_id}): umbral calibrado (P{PERCENTIL_UMBRAL}) = {umbral_pct:.3f}% ---")

    # Regresor: un solo entrenamiento, no depende del umbral.
    model_id_reg = f"calibrada_{item_id}_regresor"
    logging.info(f"--- {nombre} ({item_id}): entrenando regresor ---")
    _entrenar_temporal(db, model_id_reg, item_id, 'regresor', desde_ts, hasta_ts)
    señales_reg = _senales_regresor(db, model_id_reg, TABLA, desde_ts, hasta_ts, umbral_subida_pct=umbral_pct)

    # Clasificador con el umbral calibrado.
    model_id_clasif = f"calibrada_{item_id}_clasif"
    logging.info(f"--- {nombre} ({item_id}): entrenando clasificador (umbral calibrado {umbral_pct:.3f}%) ---")
    test_acumulado = ejecutar_replay_clasificador(
        db, desde_ts, hasta_ts, model_name=model_id_clasif, umbral_pct=umbral_pct,
        tabla=TABLA, ventana_dias=VENTANA_DIAS, item_ids=[item_id],
    )
    señales_clasif = _senales_clasificador(test_acumulado)
    señales_ensemble = señales_clasif.merge(
        señales_reg[['item_id', 'ts_entrada']], on=['item_id', 'ts_entrada'],
    ) if not señales_clasif.empty and not señales_reg.empty else pd.DataFrame(columns=señales_clasif.columns)

    # --- Test 1: clasificador / regresor / ensemble, dimensionamiento y salida de siempre (1h fijo, fracción fija) ---
    for metodo, señales in (('clasificador', señales_clasif), ('regresor', señales_reg), ('ensemble', señales_ensemble)):
        trades, resumen = backtest_capital_constante(db, TABLA, señales, CAPITAL_INICIAL)
        filas.append({
            'item_id': item_id, 'nombre': nombre, 'umbral_calibrado_pct': umbral_pct,
            'test': 'umbral_calibrado', 'metodo': metodo, **resumen,
        })
        logging.info(f"[umbral_calibrado/{metodo}] {filas[-1]}")

    # --- Test 2: dimensionamiento por confianza (solo clasificador -- es el único con prob_sube) ---
    if not test_acumulado.empty and 'prob_sube' in test_acumulado.columns:
        candidatos = test_acumulado[test_acumulado['clase_predicha'] == 2].copy()
        señales_confianza = señales_clasif.copy()
        if not señales_confianza.empty and not candidatos.empty:
            mapa_prob = candidatos.set_index(candidatos['timestamp_target'].astype(int))['prob_sube']
            señales_confianza['magnitud'] = (señales_confianza['ts_salida']).map(mapa_prob).fillna(0.5)
        trades, resumen = backtest_capital_constante(
            db, TABLA, señales_confianza, CAPITAL_INICIAL, fraccion_fn=_fraccion_por_confianza,
        )
        filas.append({
            'item_id': item_id, 'nombre': nombre, 'umbral_calibrado_pct': umbral_pct,
            'test': 'dimensionamiento_confianza', 'metodo': 'clasificador', **resumen,
        })
        logging.info(f"[dimensionamiento_confianza] {filas[-1]}")

    # --- Test 3: salida por triple barrera (clasificador y ensemble -- son los que tienen dirección clara) ---
    for metodo, señales_base in (('clasificador', señales_clasif), ('ensemble', señales_ensemble)):
        señales_tb = _senales_triple_barrera(db, TABLA, señales_base, umbral_pct, STOP_LOSS_FRACCION, TIMEOUT_PASOS)
        trades, resumen = backtest_capital_constante(db, TABLA, señales_tb, CAPITAL_INICIAL)
        filas.append({
            'item_id': item_id, 'nombre': nombre, 'umbral_calibrado_pct': umbral_pct,
            'test': 'triple_barrera', 'metodo': metodo, **resumen,
        })
        logging.info(f"[triple_barrera/{metodo}] {filas[-1]}")

    return filas


def ejecutar_busqueda_calibrada():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s |--| %(levelname)s |--| %(message)s')
    db = OSRSBaseDatos(DB_PATH)
    desde_ts, hasta_ts = _rango_backtest(db, TABLA)
    logging.info(f"=== Búsqueda calibrada: {len(ITEMS)} ítems, backtest {desde_ts}-{hasta_ts} ===")

    resultados = []
    for item in ITEMS:
        try:
            filas = probar_item(db, item, desde_ts, hasta_ts)
            resultados.extend(filas)
            pd.DataFrame(resultados).to_csv(CSV_RESULTADOS, index=False)
        except Exception as e:
            logging.error(f"Ítem {item['nombre']} falló: {e}", exc_info=True)

    df = pd.DataFrame(resultados)
    df.to_csv(CSV_RESULTADOS, index=False)
    logging.info(f"\n=== Búsqueda calibrada completa: {len(df)} filas en {CSV_RESULTADOS} ===")
    if not df.empty:
        logging.info("\n" + df.sort_values('ganancia', ascending=False).to_string(index=False))
    return df


if __name__ == '__main__':
    ejecutar_busqueda_calibrada()
