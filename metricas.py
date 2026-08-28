"""
metricas.py — métricas de "screener" para la futura web app de visualización.

Separado a propósito de preprocesamiento.py: ese módulo genera features para
entrenar un modelo (lags, medias móviles, encoding cíclico) y no son legibles
para una persona. Acá, en cambio, cada valor está pensado para MOSTRARSE
directamente a un usuario (margen en gp, % de cambio, etc.).

Convención de precios (según la API de prices.runescape.wiki):
    avg_high_price = precio promedio de COMPRA instantánea (lo que pagarías
                      si quieres comprar el ítem YA).
    avg_low_price  = precio promedio de VENTA instantánea (lo que recibirías
                      si quieres vender el ítem YA).
Estrategia de flip clásica en el Grand Exchange: comprar cerca de
avg_low_price (con una orden de compra, no instantánea) y vender cerca de
avg_high_price (con una orden de venta). Por eso el margen de flip se calcula
como avg_high_price - avg_low_price, no al revés.
"""

import math
import sqlite3
import statistics
import time


# ---------------------------------------------------------------------------
# Impuesto del Grand Exchange (GE tax)
# Fuente: https://oldschool.runescape.wiki/w/Grand_Exchange_tax (verificado
# agosto 2026, vigente desde el 29 de mayo de 2025):
#   - 2% del precio de venta, redondeado hacia ABAJO al entero más cercano
#     (por eso una venta bajo 50 gp no paga impuesto: floor(49 * 0.02) = 0).
#   - Tope de 5,000,000 gp de impuesto por transacción individual, sin
#     importar cuán caro sea el ítem.
#   - Un conjunto fijo de ítems está 100% exento (ver EXEMPT_ITEM_IDS).
# Estas reglas cambian con actualizaciones del juego (el impuesto pasó de 1%
# a 2% en mayo 2025) — si algo empieza a no cuadrar con la realidad, lo
# primero a revisar es si Jagex cambió la tasa, el tope o la lista de exentos.
GE_TAX_RATE = 0.02
GE_TAX_CAP = 5_000_000

# Ítems exentos de impuesto GE (resueltos por nombre contra la tabla `items`
# de esta base de datos en agosto 2026). Categorías: bonds, munición y
# comida de nivel bajo, tablets/joyas de teletransporte, y herramientas de
# fabricación/reparación.
EXEMPT_ITEM_IDS = {
    13190,  # Old school bond
    3008, 3010, 3012, 3014,  # Energy potion (4), (3), (2), (1)
    882, 806, 884, 807, 558, 886, 808,  # Bronze/Iron/Steel arrow-dart, Mind rune
    365, 2309, 1891, 2140, 2142, 347, 379, 355, 2327, 351, 329, 315, 361,  # comida de nivel bajo
    8011, 8010, 28824, 8009, 3853, 28790, 8008, 2552, 8013, 8007,  # teletransporte
    1755, 5325, 1785, 2347, 1733, 233, 5341, 8794, 5329, 5343, 1735, 952, 5331,  # herramientas
}


def calcular_impuesto_ge(precio_venta, item_id):
    """Impuesto GE en gp para una venta a `precio_venta`. 0 si el ítem está
    exento o el precio es inválido."""
    if item_id in EXEMPT_ITEM_IDS or not precio_venta or precio_venta <= 0:
        return 0
    return min(math.floor(precio_venta * GE_TAX_RATE), GE_TAX_CAP)


def calcular_margen(avg_high_price, avg_low_price, item_id):
    """Margen de flip: comprar a avg_low_price, vender a avg_high_price
    (la venta es la que paga impuesto)."""
    impuesto = calcular_impuesto_ge(avg_high_price, item_id)
    margen_bruto = avg_high_price - avg_low_price
    margen_neto = margen_bruto - impuesto
    roi_pct = (margen_neto / avg_low_price * 100) if avg_low_price else None
    return {
        'margen_bruto': margen_bruto,
        'impuesto_ge': impuesto,
        'margen_neto': margen_neto,
        'roi_pct': roi_pct,
    }


def calcular_profit_potencial(margen_neto, buy_limit):
    """Ganancia máxima teórica si compras/vendes el buy_limit completo
    (el límite del GE se resetea cada 4 horas)."""
    if buy_limit is None or margen_neto is None:
        return None
    return margen_neto * buy_limit


def dimensionar_oportunidad(margen_neto, buy_limit, volumen_1h_promedio, horas_horizonte, fraccion_participacion=0.15):
    """
    Ganancia estimada REALISTA de una oportunidad de flip — a diferencia de
    calcular_profit_potencial (que asume que se llena el buy_limit completo,
    optimista si el ítem no tiene volumen de mercado suficiente para
    absorber esa cantidad sin que el precio se mueva en contra), acota las
    unidades a min(buy_limit, volumen esperado en el horizonte *
    fraccion_participacion). `fraccion_participacion` es un supuesto
    conservador y configurable de "qué fracción del volumen horario típico
    se puede capturar sin mover el precio" — no un número validado contra
    microestructura real del Grand Exchange, ajustar si la experiencia real
    de flipping sugiere que es muy optimista o muy conservador.
    """
    if margen_neto is None or buy_limit is None:
        return None
    volumen_capturable = (volumen_1h_promedio or 0) * horas_horizonte * fraccion_participacion
    unidades = min(buy_limit, volumen_capturable)
    return margen_neto * unidades


def margen_neto_proyectado(avg_low_price_actual, avg_high_price_actual, avg_low_price_predicho, item_id):
    """
    Margen neto de comprar AHORA a avg_low_price_actual y vender en el
    horizonte que pronosticó el modelo (prediccion.py/replay_historico.py),
    que solo predice avg_low_price — no avg_high_price ni el spread. Se
    estima el avg_high_price futuro como
    avg_low_price_predicho * (avg_high_price_actual / avg_low_price_actual),
    la misma aproximación de "spread congelado" que ya usa el pronóstico
    recursivo (ver el docstring de prediccion.py): asume que la relación
    compra/venta actual se mantiene. Es una aproximación de corto plazo, más
    floja cuanto mayor el horizonte — usar accuracy_direccional por
    horizonte (evaluacion.py / model_metrics) para calibrar hasta qué
    horizonte esto sigue siendo razonable antes de alertar con él.

    Descuenta el impuesto GE (calcular_impuesto_ge) sobre el precio de venta
    proyectado: el modelo predice el precio crudo, sin impuesto, así que sin
    este descuento la señal sobreestimaría la ganancia real.
    """
    if not avg_low_price_actual or not avg_high_price_actual:
        return None
    ratio_spread = avg_high_price_actual / avg_low_price_actual
    avg_high_price_predicho = avg_low_price_predicho * ratio_spread
    impuesto = calcular_impuesto_ge(avg_high_price_predicho, item_id)
    return avg_high_price_predicho - avg_low_price_actual - impuesto


def filtrar_screener_liquido(resumen_df, volumen_24h_minimo=100, margen_neto_minimo=None):
    """
    Filtra el screener (resumen_actual, o cualquier DataFrame con las mismas
    columnas) a ítems con volumen_24h >= volumen_24h_minimo (y opcionalmente
    margen_neto >= margen_neto_minimo). Sin este filtro, ítems casi sin
    liquidez generan roi_pct absurdos — ej. un ítem de 1 gp con
    volumen_24h=10 mostrando roi_pct=141600% en el historial real de esta
    DB — que es ruido, no una oportunidad real de flip (nadie puede mover
    volumen relevante en un ítem así). Usado por el dashboard (slider del
    screener), alertas.py y backtest.py — un solo filtro compartido, no una
    implementación distinta en cada uno.
    """
    filtrado = resumen_df[resumen_df['volumen_24h'] >= volumen_24h_minimo]
    if margen_neto_minimo is not None:
        filtrado = filtrado[filtrado['margen_neto'] >= margen_neto_minimo]
    return filtrado.reset_index(drop=True)


def pct_cambio(serie_precios):
    """% de cambio entre el primer y el último valor de una serie ordenada
    por tiempo ascendente."""
    if len(serie_precios) < 2 or not serie_precios[0]:
        return None
    return (serie_precios[-1] - serie_precios[0]) / serie_precios[0] * 100


def volatilidad(serie_precios):
    """Coeficiente de variación (desviación estándar / promedio) en %.
    Más alto = precio más inestable; sirve para separar ítems "tranquilos"
    de ítems con swings grandes."""
    if len(serie_precios) < 2:
        return None
    promedio = statistics.fmean(serie_precios)
    if not promedio:
        return None
    return statistics.pstdev(serie_precios) / promedio * 100


def percentil_historico(precio_actual, serie_precios):
    """Ubica precio_actual dentro del rango [min, max] de serie_precios:
    0 = mínimo del periodo, 100 = máximo del periodo. Es la señal clásica
    de "¿está barato o caro respecto a su propio historial reciente?"."""
    if not serie_precios:
        return None
    minimo, maximo = min(serie_precios), max(serie_precios)
    if maximo == minimo:
        return 50.0
    return (precio_actual - minimo) / (maximo - minimo) * 100


def tendencia_volumen(serie_volumen, ventana_reciente=6):
    """Volumen promedio de las últimas `ventana_reciente` horas dividido por
    el promedio del resto de la serie. >1 = el volumen está subiendo
    (posible señal de que algo le está pasando al ítem)."""
    if len(serie_volumen) < ventana_reciente + 1:
        return None
    reciente = statistics.fmean(serie_volumen[-ventana_reciente:])
    resto = statistics.fmean(serie_volumen[:-ventana_reciente])
    if not resto:
        return None
    return reciente / resto


# ---------------------------------------------------------------------------
def calcular_resumen_item(db, item_id, nombre, buy_limit, tabla='precios_1h', horas_historial=720, ahora_ts=None):
    """Calcula el set completo de métricas para un ítem, usando hasta
    `horas_historial` horas más recientes de su historial (default 720h =
    30 días) para percentil/volatilidad/tendencia.

    ahora_ts (opcional): en vez de usar el reloj real, calcula el resumen
    "como si fuera" este momento — usado por el replay histórico
    (replay_historico.py) para simular qué habría mostrado el screener en
    un punto del pasado, sin ver datos posteriores ya backfilleados."""
    # Filtra por fecha en la query SQL (desde_timestamp/hasta_timestamp) en
    # vez de traer todo el historial del ítem y recortar después con
    # .tail() en pandas — con meses de datos acumulados, traer todo en cada
    # refresh del screener se vuelve cada vez más lento.
    ahora = ahora_ts if ahora_ts is not None else int(time.time())
    desde = ahora - horas_historial * 3600
    df = db.obtener_precios_id(item_id, tabla, desde_timestamp=desde, hasta_timestamp=ahora_ts)
    if df.empty:
        return None

    df = df.sort_values('timestamp')
    if len(df) < 2:
        return None

    ultimo = df.iloc[-1]
    high, low = ultimo['avg_high_price'], ultimo['avg_low_price']
    if not high or not low:
        return None

    margen = calcular_margen(int(high), int(low), item_id)
    profit_potencial = calcular_profit_potencial(margen['margen_neto'], buy_limit)

    precios_medios = ((df['avg_high_price'] + df['avg_low_price']) / 2).tolist()
    volumenes = (df['high_volume'] + df['low_volume']).tolist()

    def ultimas(n):
        # ventanas en "horas de historial disponible", no de reloj —
        # si el ítem tiene huecos de datos esto no corrige por eso.
        return precios_medios[-n:] if len(precios_medios) >= n else precios_medios

    return {
        'item_id': int(item_id),
        'name': nombre,
        'ultimo_timestamp': int(ultimo['timestamp']),
        'avg_high_price': int(high),
        'avg_low_price': int(low),
        **margen,
        'profit_potencial_4h': profit_potencial,
        'pct_cambio_24h': pct_cambio(ultimas(24)),
        'pct_cambio_7d': pct_cambio(ultimas(24 * 7)),
        'volatilidad_30d_pct': volatilidad(precios_medios),
        'percentil_30d': percentil_historico(precios_medios[-1], precios_medios),
        'volumen_24h': int(sum(volumenes[-24:])),
        'tendencia_volumen': tendencia_volumen(volumenes),
        'n_horas_historial': len(df),
    }


def calcular_resumen_todos(db, tabla='precios_1h', horas_historial=720, ahora_ts=None):
    """Recorre todos los ítems con datos en `tabla` y calcula su resumen.
    Devuelve una lista de dicts (una fila por ítem con datos suficientes),
    lista para pasarse a OSRSBaseDatos.guardar_resumen().

    ahora_ts (opcional): propagado a calcular_resumen_item para simular un
    momento del pasado (replay histórico); también acota acá el universo de
    ítems candidatos a los que ya tenían datos hasta ese momento — sin esto,
    un ítem con datos solo posteriores a ahora_ts aparecería igual."""
    conn = sqlite3.connect(db.db_path)
    c = conn.cursor()
    query = f'''
        SELECT DISTINCT p.item_id, i.name, i.buy_limit
        FROM {tabla} p JOIN items i ON i.item_id = p.item_id
    '''
    params = []
    if ahora_ts is not None:
        query += ' WHERE p.timestamp <= ?'
        params.append(ahora_ts)
    c.execute(query, params)
    candidatos = c.fetchall()
    conn.close()

    resultados = []
    for item_id, nombre, buy_limit in candidatos:
        fila = calcular_resumen_item(db, item_id, nombre, buy_limit, tabla, horas_historial, ahora_ts=ahora_ts)
        if fila:
            resultados.append(fila)
    return resultados


if __name__ == '__main__':
    import logging
    from base_de_datos import OSRSBaseDatos

    logging.basicConfig(level=logging.INFO, format='%(asctime)s |--| %(levelname)s |--| %(message)s')

    db = OSRSBaseDatos('data/osrs_ge.db')
    logging.info('Calculando resumen para todos los ítems con datos en precios_1h...')
    resumen = calcular_resumen_todos(db)
    n = db.guardar_resumen(resumen)
    logging.info(f'resumen_actual actualizado: {n} ítems')
