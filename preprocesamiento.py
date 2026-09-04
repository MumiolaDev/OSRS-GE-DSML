import sqlite3

import pandas as pd
import numpy as np

# Segundos por período según la tabla de precios — necesario acá (no solo en
# prediccion.py) porque las features ahora dependen de la grilla temporal:
# ver _reindexar_a_grilla y el docstring de construir_features.
PASO_SEGUNDOS_POR_TABLA = {'precios_5m': 300, 'precios_1h': 3600, 'precios_6h': 21600}

# Versión del esquema de features. Se guarda en el bundle .pkl
# (entrenador.py) y se verifica al inferir (prediccion.py): un modelo
# entrenado con la versión 1 (features en NIVELES, ver más abajo) alimentado
# con features de la versión 2 no falla — reindex(columns=...) simplemente
# rellena con NaN las columnas que no existen — sino que predice ruido sin
# ningún error visible. Subir este número cada vez que cambie el conjunto o
# el significado de las columnas que produce construir_features.
FEATURES_VERSION = 2


def _reindexar_a_grilla(df, paso_segundos):
    """
    Deja la serie de un ítem sobre la grilla temporal REGULAR (un punto cada
    `paso_segundos`), insertando filas con NaN donde el dato no existe.

    Por qué es imprescindible: la API no devuelve un bucket cuando el ítem
    no se transó por las dos puntas en ese período (ver
    osrs_ge_api._procesar_data_historicals), y el recolector puede haber
    estado apagado. Sin reindexar, `shift(1)` significa "la fila anterior
    que exista", que puede ser de hace diez horas, y el target
    `shift(-1) - actual` deja de ser el movimiento de UN período para pasar
    a ser el de un lapso arbitrario y desconocido. El modelo aprende
    entonces una mezcla de horizontes distintos, y el backtest —que asume
    `timestamp_target - timestamp == paso`— empareja compras y ventas mal.

    Con la grilla regular, cualquier lag/media móvil/target que toque un
    hueco queda en NaN y lo elimina el dropna() del llamador: se pierden
    esas filas, que es exactamente lo correcto, en vez de contaminarlas.

    Los timestamps de la API son siempre múltiplos exactos del paso, así que
    la grilla generada desde el mínimo cae siempre sobre los datos reales.
    """
    if df.empty:
        return df
    # La PK (item_id, timestamp) ya garantiza unicidad en la DB, pero el
    # pronóstico recursivo de prediccion.py concatena filas sintéticas que
    # pueden caer sobre un timestamp ya presente; reindex() sobre un índice
    # con duplicados lanza excepción. Gana la última (la sintética).
    df = df.drop_duplicates(subset='timestamp', keep='last')
    ts_min, ts_max = int(df['timestamp'].min()), int(df['timestamp'].max())
    grilla = np.arange(ts_min, ts_max + 1, paso_segundos)
    completo = df.set_index('timestamp').reindex(grilla)
    completo.index.name = 'timestamp'
    if 'item_id' in completo.columns:
        # Constante de la serie: no tiene sentido dejarla NaN en los huecos
        # (se usa como categórica, no como dato del período).
        completo['item_id'] = completo['item_id'].ffill().bfill()
    return completo.reset_index()


def construir_features(df, target_col='avg_low_price', lags=5, ma_windows=[3, 6], paso_segundos=3600):
    """
    Calcula, sobre `df` (columnas: timestamp, avg_high_price, avg_low_price,
    high_volume, low_volume), las features del modelo — sin generar el
    target ni descartar filas por historia insuficiente. Es el cálculo que
    comparten:
    - `preprocess_item` (entrenamiento: le agrega el target y descarta la
      última fila, que no tiene "período siguiente" contra el cual entrenar)
    - `prediccion.py` (inferencia hacia adelante: necesita justo la última
      fila, la que `preprocess_item` descartaría, porque ahí no hay todavía
      un precio real siguiente con el cual comparar).
    Mantener esto en un solo lugar evita que el cálculo de features en
    entrenamiento y en inferencia se desincronice con el tiempo (train/serve
    skew).

    TODAS las features son RELATIVAS (retornos, distancias a una media,
    ratios), ninguna es un nivel absoluto. El motivo es concreto y medido:
    el target es un log-retorno, `log(p_t+1) - log(p_t)`, y un árbol de
    decisión no puede restar dos features. Dándole `price_lag_1 = 5.43` y
    `price_lag_2 = 5.41` (la versión 1 de este archivo) el modelo no puede
    formar "subió 0.02"; solo puede memorizar umbrales de precio por ítem,
    que no generalizan a ningún otro nivel de precio. Peor todavía: esa
    versión excluía `log_price` de las features, así que el modelo tenía los
    cinco lags pero NO el precio actual — tenía que predecir un movimiento
    medido desde un punto de referencia que no podía ver.

    Medido sobre 24 ítems líquidos y 365 horas reales de la API, mismos
    hiperparámetros y tres cortes temporales distintos: la accuracy
    direccional sobre movimientos reales pasa de 54.9% (IC95% 52.7-57.1) con
    features en niveles a 68.8% (IC95% 66.7-70.8) con estas. El MAE en
    cambio no mejora — ninguna de las dos le gana al baseline trivial "el
    precio no cambia" (ver baseline.py), la ganancia está en la dirección,
    no en el nivel.

    `paso_segundos`: granularidad de la serie, para reindexarla a la grilla
    regular antes de calcular nada (ver _reindexar_a_grilla). Tiene que
    coincidir con la tabla de la que salió `df` — entrenador.py lo deriva de
    la tabla del modelo y lo guarda en el bundle.

    Devuelve el DataFrame ordenado por timestamp con las columnas de
    features agregadas al final, sin dropna ni selección de columnas.
    """
    df = df.sort_values('timestamp').copy()

    # Precios inválidos (<=0) no tienen logaritmo definido. Se descartan
    # ANTES de reindexar para que queden como hueco explícito de la grilla,
    # no como una fila que corre el resto de la serie un lugar.
    df = df[df[target_col] > 0]
    df = _reindexar_a_grilla(df, paso_segundos)
    if df.empty:
        return df

    # Variables de tiempo: hora del día y día de la semana, encoding cíclico.
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
    df['hour'] = df['datetime'].dt.hour
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['dow'] = df['datetime'].dt.dayofweek
    df['dow_sin'] = np.sin(2 * np.pi * df['dow'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['dow'] / 7)

    # Intermedios en espacio logarítmico. NO son features (quedan en
    # _COLUMNAS_NO_FEATURE): son niveles absolutos, solo sirven de base para
    # las diferencias de más abajo.
    df['log_price'] = np.log(df[target_col])
    df['total_volume'] = df['high_volume'] + df['low_volume']
    df['log_volume'] = np.log1p(df['total_volume'])

    # Spread relativo log(high/low): comparable entre ítems (a diferencia de
    # high - low en gp) y ya relativo de por sí, así que acá SÍ es feature
    # directamente. NaN si avg_high_price es inválido, lo limpia el dropna()
    # del llamador.
    high_valido = df['avg_high_price'].where(df['avg_high_price'] > 0)
    df['log_spread'] = np.log(high_valido) - df['log_price']

    # Desbalance de volumen: qué fracción del volumen del período se transó
    # contra la punta alta (compras instantáneas) en vez de la baja. Es un
    # proxy directo de presión compradora/vendedora dentro del período y
    # estaba disponible desde siempre en los datos crudos, pero se perdía al
    # sumar high_volume + low_volume en un solo total. 0.5 = equilibrado.
    # NaN (no 0.5) cuando no hubo volumen o la fila es un hueco de la
    # grilla: rellenarlo con un valor inventado contaminaría las medias
    # móviles de las filas vecinas, que sí son reales.
    df['desbalance_volumen'] = (df['high_volume'] / df['total_volume']).where(df['total_volume'] > 0)

    # Retornos pasados: log(p_t) - log(p_t-k), el movimiento acumulado de
    # los últimos k períodos hasta AHORA. Reemplazan a los lags en nivel.
    for lag in range(1, lags + 1):
        df[f'ret_lag_{lag}'] = df['log_price'] - df['log_price'].shift(lag)
        df[f'vol_ret_lag_{lag}'] = df['log_volume'] - df['log_volume'].shift(lag)
        df[f'spread_dif_lag_{lag}'] = df['log_spread'] - df['log_spread'].shift(lag)

    # Distancia a la media móvil (señal clásica de reversión a la media) y
    # volatilidad reciente, ambas en unidades comparables entre ítems.
    retorno_1 = df['log_price'] - df['log_price'].shift(1)
    for w in ma_windows:
        df[f'dist_ma_price_{w}'] = df['log_price'] - df['log_price'].rolling(w).mean()
        df[f'dist_ma_volume_{w}'] = df['log_volume'] - df['log_volume'].rolling(w).mean()
        df[f'volatilidad_{w}'] = retorno_1.rolling(w).std()
        df[f'dist_ma_desbalance_{w}'] = (
            df['desbalance_volumen'] - df['desbalance_volumen'].rolling(w).mean()
        )

    return df


# Columnas crudas/intermedias que nunca son features del modelo (se calculan
# para llegar a las features, pero no se le pasan al modelo tal cual).
# `log_price`/`log_volume` están acá justamente porque son NIVELES: darle a
# un árbol el log-precio absoluto solo le permite memorizar el rango de
# precios de cada ítem (ver el docstring de construir_features).
_COLUMNAS_NO_FEATURE = [
    'timestamp', 'datetime', 'hour', 'dow',
    'avg_high_price', 'avg_low_price', 'high_volume', 'low_volume', 'total_volume',
    'log_price', 'log_volume',
]


def columnas_feature(df, target_col='avg_low_price'):
    """Nombres de columnas de `df` que son features del modelo (excluye
    crudas/intermedias y `target_col`, que puede no estar en la lista fija
    de arriba si es distinta de avg_low_price)."""
    excluir = set(_COLUMNAS_NO_FEATURE) | {target_col}
    return [col for col in df.columns if col not in excluir]


def preprocess_item(df, target_col='avg_low_price', lags=5, ma_windows=[3, 6], paso_segundos=3600):
    """
    Genera el set de entrenamiento para un ítem a partir de su serie
    temporal (features de `construir_features` + target).

    El target es el log-retorno del siguiente período
    (log(precio_t+1) - log(precio_t)), no el precio crudo. Esto es lo que
    hace comparables las features/target entre ítems de escalas de precio
    muy distintas (ver build_training_set) — un modelo por ítem no lo
    necesitaría, pero uno global sí.

    Gracias al reindexado a la grilla regular de construir_features, acá
    "período siguiente" significa literalmente `timestamp + paso_segundos`:
    si ese período no tiene dato, el target queda NaN y la fila se descarta,
    en vez de emparejarse con el próximo dato disponible (que podía estar
    diez horas después). Eso vale también para los lags/medias móviles.

    Devuelve un DataFrame con las features, más 'timestamp_target',
    'price_actual' y 'price_target' (para reconstruir precio en gp después
    de predecir en espacio de retorno) y la columna 'target'.
    """
    if len(df) < max(lags, max(ma_windows)) + 2:
        return pd.DataFrame()  # no hay suficientes datos

    df = construir_features(df, target_col, lags, ma_windows, paso_segundos)
    if len(df) < max(lags, max(ma_windows)) + 2:
        return pd.DataFrame()

    # Nombres de las columnas de features, calculados ANTES de agregar las
    # de reconstrucción de abajo (si no, columnas_feature() las incluiría
    # también a esas y quedarían duplicadas en el `df[...]` final).
    feature_cols = columnas_feature(df, target_col)

    # Columnas de reconstrucción: precio actual (conocido en el momento de
    # predecir) y precio/timestamp del período siguiente (lo que se quiere
    # predecir / contra lo que se evalúa).
    df['price_actual'] = df[target_col]
    df['price_target'] = df[target_col].shift(-1)
    df['timestamp_target'] = df['timestamp'].shift(-1)

    # Target: log-retorno del siguiente período.
    df['target'] = df['log_price'].shift(-1) - df['log_price']

    # Eliminar filas con NaN (historia insuficiente para lags/MA, hueco de
    # datos en la ventana, o sin período siguiente para el target)
    df = df.dropna(subset=feature_cols + ['timestamp_target', 'price_actual', 'price_target', 'target'])
    df = df.reset_index(drop=True)

    return df[feature_cols + ['timestamp_target', 'price_actual', 'price_target', 'target']]


def build_training_set(db, item_ids, tabla='precios_1h', hasta_timestamp=None, desde_timestamp=None, **kwargs):
    """
    Arma el dataset de entrenamiento del modelo global: recorre `item_ids`,
    genera features por ítem con preprocess_item, les agrega el item_id y
    las features estáticas del ítem (buy_limit, members desde la tabla
    `items`), y concatena todo en un solo DataFrame ordenado por tiempo.

    hasta_timestamp (opcional): acota cada ítem a datos con
    timestamp <= hasta_timestamp — es lo que hace posible entrenar "como si
    fuera" un momento del pasado (replay_historico.py), en vez de siempre
    usar todo lo que ya esté cargado en la tabla.

    desde_timestamp (opcional): acota cada ítem a datos con
    timestamp >= desde_timestamp — es la "ventana" de entrenamiento
    (entrenador.py la deriva de modelos_config.ventana_dias). Sin esto
    (default None), se usa todo el historial disponible hasta
    hasta_timestamp, el comportamiento de siempre. Nota: como los lags/
    medias móviles necesitan unas pocas filas anteriores a cada punto para
    calcularse, las primeras filas dentro de la ventana quedan sin
    suficiente historia y se descartan (dropna() en preprocess_item) — con
    ventanas chicas (ej. 24h a 5m) esto recorta una fracción notoria del
    total, no solo un puñado de filas.

    kwargs se pasan a preprocess_item (target_col, lags, ma_windows). El
    `paso_segundos` NO se pasa a mano: se deriva de `tabla`, para que no
    pueda quedar desalineado con la granularidad real de los datos.
    """
    # Una sola query para todos los ítems (obtener_precios_multi) en vez de
    # una conexión+query por ítem — con ~200 ítems, abrir 200 conexiones
    # SQLite por separado era el cuello de botella real de esta función,
    # sobre todo repetido en cada checkpoint del replay histórico
    # (replay_historico.py).
    precios_todos = db.obtener_precios_multi(item_ids, tabla, desde_timestamp=desde_timestamp, hasta_timestamp=hasta_timestamp)
    if precios_todos.empty:
        return pd.DataFrame()

    paso_segundos = PASO_SEGUNDOS_POR_TABLA.get(tabla, 3600)

    frames = []
    for item_id, df in precios_todos.groupby('item_id'):
        feat = preprocess_item(df, paso_segundos=paso_segundos, **kwargs)
        if feat.empty:
            continue
        feat = feat.copy()
        feat['item_id'] = item_id
        frames.append(feat)

    if not frames:
        return pd.DataFrame()

    dataset = pd.concat(frames, ignore_index=True)

    # Features estáticas del ítem, en una sola consulta para todos los ítems.
    conn = sqlite3.connect(db.db_path)
    placeholders = ','.join('?' * len(item_ids))
    items_info = pd.read_sql_query(
        f'SELECT item_id, buy_limit, members FROM items WHERE item_id IN ({placeholders})',
        conn, params=item_ids,
    )
    conn.close()

    dataset = dataset.merge(items_info, on='item_id', how='left')
    # dtype 'category' para que XGBoost use su soporte nativo de categóricas
    # (enable_categorical=True) en vez de tratar item_id como numérico ordinal.
    dataset['item_id'] = dataset['item_id'].astype('category')
    dataset['members'] = dataset['members'].fillna(0).astype('category')

    return dataset.sort_values('timestamp_target').reset_index(drop=True)
