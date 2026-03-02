import pandas as pd
import numpy as np

def preprocess_item(df, target_col='avg_low_price', lags=5, ma_windows=[3,6]):
    """
    Genera features para un ítem a partir de su serie temporal.
    df debe tener columnas: timestamp, avg_high_price, avg_low_price, high_volume, low_volume
    Devuelve un DataFrame con las features y la columna 'target' (próximo precio).
    """
    df = df.sort_values('timestamp').copy()
    if len(df) < max(lags, max(ma_windows)) + 2:
        return pd.DataFrame()  # no hay suficientes datos

    # Variables de tiempo
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
    df['hour'] = df['datetime'].dt.hour
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

    # Volumen total y spread
    df['total_volume'] = df['high_volume'] + df['low_volume']
    df['spread'] = df['avg_high_price'] - df['avg_low_price']

    # Lags
    for lag in range(1, lags+1):
        df[f'price_lag_{lag}'] = df[target_col].shift(lag)
        df[f'volume_lag_{lag}'] = df['total_volume'].shift(lag)
        df[f'spread_lag_{lag}'] = df['spread'].shift(lag)

    # Medias móviles
    for w in ma_windows:
        df[f'ma_price_{w}'] = df[target_col].rolling(w).mean()
        df[f'ma_volume_{w}'] = df['total_volume'].rolling(w).mean()

    # Target: precio en el próximo período
    df['target'] = df[target_col].shift(-1)

    # Eliminar filas con NaN
    df = df.dropna().reset_index(drop=True)

    # Seleccionar solo columnas de features (excluir las originales)
    exclude = ['timestamp', 'datetime', 'hour', target_col, 'avg_high_price', 'high_volume', 'low_volume', 'total_volume', 'spread', 'target']
    feature_cols = [col for col in df.columns if col not in exclude]
    return df[feature_cols + ['target']]