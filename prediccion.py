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

`pronosticar_clase_item` es el equivalente para el clasificador direccional
(entrenador.entrenar_clasificador_direccional, variante productiva
f2p10_clasif) — a diferencia de `pronosticar_item`, es de horizonte fijo a
1 paso y sin recursión (el clasificador no tiene noción de horizontes
mayores), así que no hay loop ni filas sintéticas.
"""

import logging
import os
import sqlite3

import joblib
import numpy as np
import pandas as pd

from preprocesamiento import (
    construir_features, columnas_feature, FEATURES_VERSION, PASO_SEGUNDOS_POR_TABLA,
)

# PASO_SEGUNDOS_POR_TABLA se importa de preprocesamiento.py (donde ahora
# también lo necesita el reindexado a la grilla regular) en vez de tener la
# misma tabla de segundos-por-intervalo repetida en dos archivos.


def cargar_modelo(model_name=None, model_path=None):
    """
    Carga el bundle guardado por entrenador.py: no es solo el modelo, sino
    también los metadatos (feature_cols, categorías de item_id/members)
    necesarios para reproducir exactamente la misma codificación categórica
    que se usó al entrenar — ver el comentario en entrenador.py sobre por
    qué esto es imprescindible con XGBoost + enable_categorical.

    Se puede pedir por `model_name` (cualquier model_id de modelos_config,
    no solo 'global_horario'/'global_diario' — ver más abajo) o por
    `model_path` directo. Sin ninguno de los dos, carga 'global_horario'
    — el modelo de cadencia rápida, el que consumen las alertas casi en
    tiempo real (alertas.py).

    Resuelve la ruta con el mismo criterio que el lado que GUARDA el
    bundle (entrenador.entrenar_modelo_global/entrenar_clasificador_direccional):
    entrenador.MODEL_PATHS.get(model_name, .../f"model_{model_name}.pkl")
    — antes esto indexaba MODEL_PATHS directo (`MODEL_PATHS[model_name]`),
    que solo tiene 4 entradas fijas (los 2 regresores + 2 variantes del
    clasificador F2P); cualquier modelo creado por el usuario desde la app
    de escritorio (ver escritorio/paginas/pagina_modelos.py) SÍ se guarda
    bien (ese lado ya usaba `.get()` con fallback), pero cargar_modelo
    tiraba KeyError al intentar leerlo de vuelta — bug real, encontrado
    revisando pagina_oportunidades.py ("no se pudo cargar el modelo
    'custom_...'": ese `str(KeyError(...))` es justo el nombre del modelo
    entre comillas, no un mensaje de "no entrenado todavía").
    """
    if model_path is None:
        from entrenador import MODEL_NAME_HORARIO, MODEL_PATHS, MODEL_DIR
        nombre = model_name or MODEL_NAME_HORARIO
        model_path = MODEL_PATHS.get(nombre, os.path.join(MODEL_DIR, f"model_{nombre}.pkl"))
    bundle = joblib.load(model_path)
    _verificar_version_features(bundle, model_path)
    return bundle


def _verificar_version_features(bundle, origen):
    """
    Rechaza un bundle entrenado con un esquema de features distinto del que
    produce preprocesamiento.py hoy (ver FEATURES_VERSION).

    Hace falta un chequeo explícito porque el modo de falla es silencioso:
    _preparar_fila_prediccion arma la fila con `reindex(columns=
    bundle['feature_cols'])`, que rellena con NaN toda columna que el
    bundle esperaba y ya no existe. XGBoost acepta NaN sin protestar
    (los trata como valores faltantes), así que un modelo viejo sigue
    "prediciendo" — ruido, sin ningún error ni warning. Mejor romper con un
    mensaje claro y que el usuario reentrene.
    """
    version = bundle.get('features_version', 1)
    if version != FEATURES_VERSION:
        raise ValueError(
            f"El modelo guardado en {origen} se entrenó con features_version={version} y el "
            f"código actual genera la versión {FEATURES_VERSION}. Reentrenalo "
            "(\"Entrenar ahora\" en Mis modelos, o esperar al próximo job horario) antes de "
            "usarlo: predecir con features de otra versión devuelve ruido en silencio."
        )


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
    # paso_segundos del bundle: las features se calculan sobre la grilla
    # temporal regular de la tabla con la que se entrenó (ver
    # preprocesamiento._reindexar_a_grilla). Con bundles nuevos siempre está;
    # el default es solo por si falta.
    paso_segundos = bundle.get('paso_segundos', PASO_SEGUNDOS_POR_TABLA.get(bundle.get('tabla'), 3600))
    feats = construir_features(df, target_col, bundle['lags'], bundle['ma_windows'], paso_segundos)
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


def _obtener_item_info(db, item_id):
    """
    (buy_limit, members) del ítem desde la tabla `items`, o None si no está
    registrado. Compartido por pronosticar_item y pronosticar_clase_item —
    ambos necesitan estas dos columnas estáticas para armar la fila de
    features (ver _preparar_fila_prediccion), antes ese lookup estaba
    duplicado en los dos.
    """
    conn = sqlite3.connect(db.db_path)
    fila_item = conn.execute(
        'SELECT buy_limit, members FROM items WHERE item_id = ?', (item_id,)
    ).fetchone()
    conn.close()
    return fila_item


def pronosticar_item(db, item_id, bundle=None, n_pasos=6, tabla=None, hasta_timestamp=None):
    """
    Pronóstico recursivo a `n_pasos` períodos hacia adelante (horas, si
    tabla='precios_1h') para un ítem. Devuelve un DataFrame con columnas
    paso, timestamp, predicted_price — vacío si no hay suficiente historia
    o el ítem no existe.

    tabla (opcional): si no se pasa, se usa bundle['tabla'] (la
    granularidad con la que se entrenó ese modelo específico, ver
    entrenador.py) — con bundles viejos sin esa clave, 'precios_1h' como
    antes. Pasar `tabla` a mano solo tiene sentido para forzar una
    granularidad distinta de la de entrenamiento, algo que en general
    produce features con una escala temporal distinta a la que el modelo
    aprendió (train/serve skew) — no hacerlo salvo que sepas por qué.

    hasta_timestamp (opcional): en vez de partir del último dato real
    disponible, parte del último dato <= hasta_timestamp. Usado por
    evaluacion.evaluar_horizontes() para simular "qué habría pronosticado
    el modelo en este momento del pasado" y compararlo contra lo que
    después efectivamente pasó (ya conocido en la DB) — a diferencia de un
    pronóstico real hacia adelante, donde el futuro todavía no existe.
    """
    if bundle is None:
        bundle = cargar_modelo()
    if tabla is None:
        tabla = bundle.get('tabla', 'precios_1h')

    df = db.obtener_precios_id(item_id, tabla, hasta_timestamp=hasta_timestamp)
    if df.empty:
        logging.warning(f"Ítem {item_id}: sin datos en {tabla}, no se puede pronosticar")
        return pd.DataFrame()
    df = df.sort_values('timestamp').reset_index(drop=True)

    fila_item = _obtener_item_info(db, item_id)
    if fila_item is None:
        logging.warning(f"Ítem {item_id}: no está en la tabla items")
        return pd.DataFrame()
    buy_limit, members = fila_item

    target_col = bundle['target_col']
    modelo = bundle['model']
    paso_segundos = PASO_SEGUNDOS_POR_TABLA.get(tabla, 3600)

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

        # El punto de partida puede no ser la última fila de `df`: si esa
        # fila cae justo después de un hueco, sus lags/medias móviles quedan
        # en NaN y _preparar_fila_prediccion usa una anterior. Truncar acá
        # deja la serie terminando en ts_actual, así la fila sintética que
        # se agrega abajo extiende la serie en vez de pisar una observación
        # real (que además dispararía timestamps duplicados en la grilla).
        df = df[df['timestamp'] <= ts_actual]

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


def pronosticar_clase_item(db, item_id, bundle=None, tabla=None, hasta_timestamp=None):
    """
    Inferencia hacia adelante del clasificador direccional (entrenador.
    entrenar_clasificador_direccional): UNA sola predicción a horizonte fijo
    de 1 paso — a diferencia de pronosticar_item (regresor), que es
    recursivo a n_pasos, el clasificador no tiene noción de horizontes > 1
    (ver backtest.simular_clasificador* y CLAUDE.md), así que no hay loop
    ni fila sintética que encadenar.

    tabla (opcional): ver pronosticar_item — si no se pasa, se usa
    bundle['tabla'] (con qué granularidad se entrenó ese modelo).

    Devuelve un dict: {item_id, clase_predicha (0/1/2), label ('baja'/
    'estable'/'sube', vía bundle['clase_labels']), probabilidades (dict
    label->prob, de predict_proba), precio_actual, timestamp_dato,
    timestamp_prediccion} — o None si no hay suficiente historia o el ítem
    no existe.

    bundle=None carga por default el modelo productivo f2p10_100gp_clasif
    (import perezoso de entrenador, mismo patrón que cargar_modelo) — pasar
    un bundle explícito (ej. desde el dashboard, ya cacheado) evita
    recargarlo del disco en cada llamada.
    """
    if bundle is None:
        from entrenador import MODEL_NAME_CLASIF_F2P_100GP
        bundle = cargar_modelo(model_name=MODEL_NAME_CLASIF_F2P_100GP)
    if tabla is None:
        tabla = bundle.get('tabla', 'precios_1h')

    df = db.obtener_precios_id(item_id, tabla, hasta_timestamp=hasta_timestamp)
    if df.empty:
        logging.warning(f"Ítem {item_id}: sin datos en {tabla}, no se puede pronosticar clase")
        return None
    df = df.sort_values('timestamp').reset_index(drop=True)

    fila_item = _obtener_item_info(db, item_id)
    if fila_item is None:
        logging.warning(f"Ítem {item_id}: no está en la tabla items")
        return None
    buy_limit, members = fila_item

    preparado = _preparar_fila_prediccion(df, bundle, item_id, buy_limit, members)
    if preparado is None:
        logging.warning(f"Ítem {item_id}: historia insuficiente para pronosticar clase")
        return None
    X, precio_actual, ts_actual = preparado

    modelo = bundle['model']
    clase_pred = int(modelo.predict(X)[0])
    probs = modelo.predict_proba(X)[0]
    clase_labels = bundle.get('clase_labels', {0: 'baja', 1: 'estable', 2: 'sube'})

    paso_segundos = PASO_SEGUNDOS_POR_TABLA.get(tabla, 3600)
    return {
        'item_id': item_id,
        'clase_predicha': clase_pred,
        'label': clase_labels[clase_pred],
        'probabilidades': {clase_labels[i]: float(p) for i, p in enumerate(probs)},
        'precio_actual': precio_actual,
        'timestamp_dato': ts_actual,
        'timestamp_prediccion': ts_actual + paso_segundos,
    }


if __name__ == '__main__':
    from base_de_datos import OSRSBaseDatos

    logging.basicConfig(level=logging.INFO, format='%(asctime)s |--| %(levelname)s |--| %(message)s')
    db = OSRSBaseDatos('data/osrs_ge.db')
    bundle = cargar_modelo()
    for item_id in [385]:  # Shark, ejemplo
        pronostico = pronosticar_item(db, item_id, bundle, n_pasos=6)
        logging.info(f"Pronóstico ítem {item_id}:\n{pronostico}")
