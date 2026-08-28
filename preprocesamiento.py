import sqlite3

import pandas as pd
import numpy as np


def construir_features(df, target_col='avg_low_price', lags=5, ma_windows=[3, 6]):
    """
    Calcula, sobre `df` (columnas: timestamp, avg_high_price, avg_low_price,
    high_volume, low_volume), los lags/medias móviles/encoding cíclico en
    espacio logarítmico — sin generar el target ni descartar filas por
    historia insuficiente. Es el cálculo que comparten:
    - `preprocess_item` (entrenamiento: le agrega el target y descarta la
      última fila, que no tiene "período siguiente" contra el cual entrenar)
    - `prediccion.py` (inferencia hacia adelante: necesita justo la última
      fila, la que `preprocess_item` descartaría, porque ahí no hay todavía
      un precio real siguiente con el cual comparar).
    Mantener esto en un solo lugar evita que el cálculo de features en
    entrenamiento y en inferencia se desincronice con el tiempo (train/serve
    skew).

    Devuelve el DataFrame ordenado por timestamp con las columnas de
    features agregadas al final, sin dropna ni selección de columnas.
    """
    df = df.sort_values('timestamp').copy()

    # Precios inválidos (<=0) no tienen logaritmo definido.
    df = df[df[target_col] > 0].reset_index(drop=True)

    # Variables de tiempo: hora del día y día de la semana, encoding cíclico.
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
    df['hour'] = df['datetime'].dt.hour
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['dow'] = df['datetime'].dt.dayofweek
    df['dow_sin'] = np.sin(2 * np.pi * df['dow'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['dow'] / 7)

    # Precio, volumen y spread en espacio logarítmico.
    df['log_price'] = np.log(df[target_col])
    df['total_volume'] = df['high_volume'] + df['low_volume']
    df['log_volume'] = np.log1p(df['total_volume'])
    # log(high/low): spread relativo, comparable entre ítems (a diferencia
    # de high - low en gp). NaN si avg_high_price es inválido, se limpia
    # con el dropna() del llamador.
    high_valido = df['avg_high_price'].where(df['avg_high_price'] > 0)
    df['log_spread'] = np.log(high_valido) - df['log_price']

    # Lags
    for lag in range(1, lags + 1):
        df[f'price_lag_{lag}'] = df['log_price'].shift(lag)
        df[f'volume_lag_{lag}'] = df['log_volume'].shift(lag)
        df[f'spread_lag_{lag}'] = df['log_spread'].shift(lag)

    # Medias móviles
    for w in ma_windows:
        df[f'ma_price_{w}'] = df['log_price'].rolling(w).mean()
        df[f'ma_volume_{w}'] = df['log_volume'].rolling(w).mean()

    return df


# Columnas crudas/intermedias que nunca son features del modelo (se calculan
# para llegar a las features, pero no se le pasan al modelo tal cual).
_COLUMNAS_NO_FEATURE = [
    'timestamp', 'datetime', 'hour', 'dow',
    'avg_high_price', 'avg_low_price', 'high_volume', 'low_volume', 'total_volume',
    'log_price', 'log_volume', 'log_spread',
]


def columnas_feature(df, target_col='avg_low_price'):
    """Nombres de columnas de `df` que son features del modelo (excluye
    crudas/intermedias y `target_col`, que puede no estar en la lista fija
    de arriba si es distinta de avg_low_price)."""
    excluir = set(_COLUMNAS_NO_FEATURE) | {target_col}
    return [col for col in df.columns if col not in excluir]


def preprocess_item(df, target_col='avg_low_price', lags=5, ma_windows=[3, 6]):
    """
    Genera el set de entrenamiento para un ítem a partir de su serie
    temporal (features de `construir_features` + target).

    El target es el log-retorno del siguiente período
    (log(precio_t+1) - log(precio_t)), no el precio crudo. Esto es lo que
    hace comparables las features/target entre ítems de escalas de precio
    muy distintas (ver build_training_set) — un modelo por ítem no lo
    necesitaría, pero uno global sí.

    Devuelve un DataFrame con las features, más 'timestamp_target',
    'price_actual' y 'price_target' (para reconstruir precio en gp después
    de predecir en espacio de retorno) y la columna 'target'.
    """
    if len(df) < max(lags, max(ma_windows)) + 2:
        return pd.DataFrame()  # no hay suficientes datos

    df = construir_features(df, target_col, lags, ma_windows)
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

    # Eliminar filas con NaN (historia insuficiente para lags/MA, o sin
    # período siguiente para el target)
    df = df.dropna().reset_index(drop=True)

    return df[feature_cols + ['timestamp_target', 'price_actual', 'price_target', 'target']]


def build_training_set(db, item_ids, tabla='precios_1h', hasta_timestamp=None, **kwargs):
    """
    Arma el dataset de entrenamiento del modelo global: recorre `item_ids`,
    genera features por ítem con preprocess_item, les agrega el item_id y
    las features estáticas del ítem (buy_limit, members desde la tabla
    `items`), y concatena todo en un solo DataFrame ordenado por tiempo.

    hasta_timestamp (opcional): acota cada ítem a datos con
    timestamp <= hasta_timestamp — es lo que hace posible entrenar "como si
    fuera" un momento del pasado (replay_historico.py), en vez de siempre
    usar todo lo que ya esté cargado en la tabla.

    kwargs se pasan a preprocess_item (target_col, lags, ma_windows).
    """
    # Una sola query para todos los ítems (obtener_precios_multi) en vez de
    # una conexión+query por ítem — con ~200 ítems, abrir 200 conexiones
    # SQLite por separado era el cuello de botella real de esta función,
    # sobre todo repetido en cada checkpoint del replay histórico
    # (replay_historico.py).
    precios_todos = db.obtener_precios_multi(item_ids, tabla, hasta_timestamp=hasta_timestamp)

    frames = []
    for item_id, df in precios_todos.groupby('item_id'):
        feat = preprocess_item(df, **kwargs)
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
