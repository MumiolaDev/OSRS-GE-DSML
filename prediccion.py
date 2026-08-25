"""
prediccion.py — pronóstico hacia adelante con el modelo ya entrenado.

Distinto de lo que ya hace entrenador.py: ese guarda en `predicciones` las
predicciones sobre el set de TEST (datos históricos, donde el precio real
ya se conoce — sirve para medir qué tan bueno es el modelo). Acá el precio
real todavía no existe: es el pronóstico de verdad, para saber de antemano
hacia dónde puede ir un ítem antes de que pase.

No hay entrenamiento online ni un solo modelo por horizonte: se reusa el
mismo modelo de 1 paso (`entrenador.py`) de forma recursiva — se predice
t+1, se agrega como fila sintética a la serie, y con eso se predice t+2, y
así hasta `n_pasos`. El volumen/spread de esas filas sintéticas se
mantienen igual al último dato real (el modelo no predice volumen), así
que el error compuesto crece con el horizonte — esto no es una serie de
verdad, es una extrapolación de lo que el modelo cree que puede pasar; se
recomienda no confiar en horizontes largos (más de ~6-12 pasos) sin
validarlo contra lo que efectivamente ocurre.
"""

import logging
import sqlite3

import joblib
import numpy as np
import pandas as pd

from preprocesamiento import construir_features, columnas_feature

MODEL_PATH_DEFAULT = 'models/model_global_v1.pkl'


def cargar_modelo(model_path=MODEL_PATH_DEFAULT):
    """
    Carga el bundle guardado por entrenador.py: no es solo el modelo, sino
    también los metadatos (feature_cols, categorías de item_id/members)
    necesarios para reproducir exactamente la misma codificación categórica
    que se usó al entrenar — ver el comentario en entrenador.py sobre por
    qué esto es imprescindible con XGBoost + enable_categorical.
    """
    return joblib.load(model_path)


def _preparar_fila_prediccion(df, bundle, item_id, buy_limit, members):
    """
    Arma la fila de features del ÚLTIMO punto conocido de `df` — a
    diferencia de preprocess_item (entrenamiento), acá no se descarta esa
    última fila, porque es justo la que no tiene todavía un "período
    siguiente" real: es lo que queremos predecir.
    Devuelve (X, precio_actual, timestamp_actual) o None si no hay
    suficiente historia para calcular lags/medias móviles.
    """
    target_col = bundle['target_col']
    feats = construir_features(df, target_col, bundle['lags'], bundle['ma_windows'])
    feature_cols_base = columnas_feature(feats, target_col)
    feats = feats.dropna(subset=feature_cols_base)
    if feats.empty:
        return None

    ultimo = feats.iloc[[-1]].copy()
    ultimo['item_id'] = pd.Categorical([item_id], categories=bundle['item_id_categories'])
    ultimo['members'] = pd.Categorical([members], categories=bundle['members_categories'])
    ultimo['buy_limit'] = float(buy_limit) if buy_limit is not None else np.nan

    X = ultimo.reindex(columns=bundle['feature_cols'])
    precio_actual = float(ultimo[target_col].iloc[0])
    ts_actual = int(ultimo['timestamp'].iloc[0])
    return X, precio_actual, ts_actual


def pronosticar_item(db, item_id, bundle=None, n_pasos=6, tabla='precios_1h'):
    """
    Pronóstico recursivo a `n_pasos` períodos hacia adelante (horas, si
    tabla='precios_1h') para un ítem. Devuelve un DataFrame con columnas
    paso, timestamp, predicted_price — vacío si no hay suficiente historia
    o el ítem no existe.
    """
    if bundle is None:
        bundle = cargar_modelo()

    df = db.obtener_precios_id(item_id, tabla)
    if df.empty:
        logging.warning(f"Ítem {item_id}: sin datos en {tabla}, no se puede pronosticar")
        return pd.DataFrame()
    df = df.sort_values('timestamp').reset_index(drop=True)

    conn = sqlite3.connect(db.db_path)
    fila_item = conn.execute(
        'SELECT buy_limit, members FROM items WHERE item_id = ?', (item_id,)
    ).fetchone()
    conn.close()
    if fila_item is None:
        logging.warning(f"Ítem {item_id}: no está en la tabla items")
        return pd.DataFrame()
    buy_limit, members = fila_item

    target_col = bundle['target_col']
    modelo = bundle['model']
    paso_segundos = {'precios_1h': 3600, 'precios_6h': 6 * 3600, 'precios_5m': 300}.get(tabla, 3600)

    resultados = []
    for paso in range(1, n_pasos + 1):
        preparado = _preparar_fila_prediccion(df, bundle, item_id, buy_limit, members)
        if preparado is None:
            logging.warning(f"Ítem {item_id}: historia insuficiente para el paso {paso}, se corta acá")
            break
        X, precio_actual, ts_actual = preparado

        retorno_pred = float(modelo.predict(X)[0])
        precio_pred = precio_actual * np.exp(retorno_pred)
        ts_pred = ts_actual + paso_segundos

        resultados.append({'paso': paso, 'timestamp': ts_pred, 'predicted_price': precio_pred})

        # Fila sintética para poder calcular lags/medias móviles del paso
        # siguiente reciclando construir_features tal cual. Volumen y
        # spread se mantienen igual al último dato real conocido — no se
        # predicen, ver docstring del módulo.
        ultima_fila_real = df.iloc[[-1]].copy()
        ratio_spread = ultima_fila_real['avg_high_price'].iloc[0] / ultima_fila_real['avg_low_price'].iloc[0]
        nueva_fila = ultima_fila_real.copy()
        nueva_fila['timestamp'] = ts_pred
        nueva_fila[target_col] = precio_pred
        if target_col == 'avg_low_price':
            nueva_fila['avg_high_price'] = precio_pred * ratio_spread
        else:
            nueva_fila['avg_low_price'] = precio_pred / ratio_spread
        df = pd.concat([df, nueva_fila], ignore_index=True)

    return pd.DataFrame(resultados)


if __name__ == '__main__':
    from base_de_datos import OSRSBaseDatos

    logging.basicConfig(level=logging.INFO, format='%(asctime)s |--| %(levelname)s |--| %(message)s')
    db = OSRSBaseDatos('data/osrs_ge.db')
    bundle = cargar_modelo()
    for item_id in [385]:  # Shark, ejemplo
        pronostico = pronosticar_item(db, item_id, bundle, n_pasos=6)
        logging.info(f"Pronóstico ítem {item_id}:\n{pronostico}")
