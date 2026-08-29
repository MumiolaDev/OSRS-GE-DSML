import logging
import os
import time

import joblib
import numpy as np
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error

from base_de_datos import OSRSBaseDatos
from preprocesamiento import build_training_set

MODEL_DIR = "models"

# Dos modelos, misma arquitectura y código de entrenamiento, distinta
# cadencia de reentrenamiento (ver recolector.py job_horario/job_diario):
# 'global_horario' es la señal rápida que consumen las alertas casi en
# tiempo real, 'global_diario' es la referencia estable de calidad de largo
# plazo. model_metrics.model_name y predicciones.model_version ya soportan
# taggear ambos sin pisarse, así que no hace falta tocar el esquema para
# esto — solo dejar de usar una constante fija (MODEL_VERSION) y parametrizar.
MODEL_NAME_HORARIO = "global_horario"
MODEL_NAME_DIARIO = "global_diario"

# Clasificador direccional (entrenar_clasificador_direccional): declarado acá
# arriba, antes de MODEL_PATHS, porque las variantes productivas necesitan
# una entrada en ese dict. global_horario_clasif (200 ítems F2P+members)
# sigue siendo solo un experimento de comparación en model_metrics — no
# tiene .pkl, no lo busques en MODEL_PATHS. f2p10_clasif (10 ítems F2P sin
# filtro de precio) fue la primera variante promovida a producción, pero
# quedó reemplazada por f2p10_100gp_clasif (mismos 10 ítems, pero con
# precio_minimo=100 y Steel bar excluido) tras el backtest walk-forward de
# 90 días: sin el filtro de precio la señal empataba con comprar a ciegas
# (ver backtest.simular_clasificador_walkforward); con el filtro, la señal
# convierte una estrategia perdedora (-67.6M gp comprando a ciegas en el
# mismo rango) en ganadora (+7.3M gp). f2p10_clasif ya no se reentrena
# desde recolector.job_horario, pero se deja el nombre/.pkl viejo sin
# borrar (no rompe nada, solo queda desactualizado). Nombres distintos
# para no pisarse en model_metrics (INSERT OR REPLACE por
# model_name+train_timestamp).
MODEL_NAME_CLASIFICADOR = "global_horario_clasif"
MODEL_NAME_CLASIF_F2P = "f2p10_clasif"
MODEL_NAME_CLASIF_F2P_100GP = "f2p10_100gp_clasif"
UMBRAL_CLASIF_PCT = 0.5  # log-retorno %: |retorno| > esto para no contar como "estable"
# 0=baja, 1=estable, 2=sube (ver _clasificar_retorno) — centralizado acá para
# que prediccion.py/dashboard.py no repitan los mismos números mágicos.
CLASE_LABELS = {0: 'baja', 1: 'estable', 2: 'sube'}

MODEL_PATHS = {
    MODEL_NAME_HORARIO: os.path.join(MODEL_DIR, "model_horario_v1.pkl"),
    MODEL_NAME_DIARIO: os.path.join(MODEL_DIR, "model_diario_v1.pkl"),
    MODEL_NAME_CLASIF_F2P: os.path.join(MODEL_DIR, "model_f2p10_clasif_v1.pkl"),
    MODEL_NAME_CLASIF_F2P_100GP: os.path.join(MODEL_DIR, "model_f2p10_100gp_clasif_v1.pkl"),
}

N_ITEMS_LIQUIDOS = 200
TEST_FRACTION = 0.2
N_PASOS_HORIZONTE = 6
N_MUESTRAS_HORIZONTE = 200

# Columnas del dataset que no son features (reconstrucción/target), el resto
# (lags, medias móviles, encoding de tiempo, item_id, buy_limit, members) se
# usa como entrada del modelo.
NON_FEATURE_COLS = ['timestamp_target', 'price_actual', 'price_target', 'target']


def entrenar_modelo_global(
    db,
    n_items=N_ITEMS_LIQUIDOS,
    model_name=MODEL_NAME_HORARIO,
    model_path=None,
    ahora_ts=None,
    guardar_en_disco=True,
    modo_evaluacion='holdout',
    calcular_metricas_horizonte=False,
    solo_f2p=False,
    item_ids=None,
    tabla='precios_1h',
    ventana_dias=None,
):
    """
    Entrena un modelo XGBoost sobre un universo de ítems, prediciendo el
    log-retorno del siguiente período (ver preprocesamiento.py para el
    porqué de trabajar en espacio de retorno y no precio crudo).

    item_ids (opcional): lista explícita de ítems a usar en vez de derivar
    el universo desde obtener_top_items_liquidez[_hasta] con n_items/
    solo_f2p — es lo que permite que un modelo definido por el usuario
    (base_de_datos.modelos_config con modo_seleccion='manual', ver
    recolector._entrenar_desde_config) entrene sobre exactamente los ítems
    que eligió a mano, no sobre un ranking de liquidez. Cuando se pasa,
    n_items/solo_f2p se ignoran por completo — el default (None) mantiene
    el comportamiento de siempre.

    tabla ('precios_5m'|'precios_1h'|'precios_6h', default 'precios_1h' —
    el comportamiento de siempre): granularidad de precios sobre la que se
    entrena. Se guarda en el bundle (bundle['tabla']) para que
    prediccion.py sepa contra qué tabla pedir el historial al pronosticar
    — sin esto, un modelo entrenado sobre precios_6h pero consultado con
    la tabla equivocada en inferencia tendría fuga de escala temporal
    (train/serve skew) sin ningún error visible, los lags/medias móviles
    significarían una cosa distinta a la que el modelo aprendió.

    ventana_dias (opcional, default None = todo el historial disponible,
    el comportamiento de siempre): acota el dataset de entrenamiento a los
    últimos `ventana_dias` días relativos a `ahora_ts` (o a ahora mismo, en
    vivo) — ver preprocesamiento.build_training_set. Pensado para modelos
    de ítems puntuales (ver base_de_datos.modelos_config), donde entrenar
    con años de historial de un solo ítem no necesariamente es mejor que
    una ventana reciente más representativa del régimen actual del mercado
    — no hay un valor "correcto" universal, es una decisión del usuario al
    crear el modelo.

    model_name/model_path: distingue 'global_horario' de 'global_diario'
    (ver MODEL_PATHS arriba) — mismo código, dos cadencias de reentrenamiento
    que conviven sin pisarse en la DB ni en disco.

    ahora_ts (opcional): si se pasa, entrena "como si fuera" ese momento del
    pasado — usa obtener_top_items_liquidez_hasta() en vez de la tabla
    resumen_actual (que es el estado EN VIVO que lee el dashboard) y acota
    build_training_set con hasta_timestamp=ahora_ts. Es lo que usa
    replay_historico.py para el walk-forward; en modo en vivo (ahora_ts=None,
    default) el comportamiento es idéntico al de antes de este cambio.

    modo_evaluacion:
    - 'holdout' (default, uso en vivo): split temporal 80/20 sobre todo el
      dataset combinado, igual que siempre — no evita fuga de futuro de un
      ítem hacia el pasado de otro vía el modelo compartido.
    - 'walkforward' (uso del replay): entrena con todo el dataset menos el
      último período (timestamp_target == max) y evalúa solo ahí. Repetir un
      split 80/20 completo en cada uno de potencialmente miles de
      checkpoints de replay sería caro y redundante (los test sets de
      checkpoints consecutivos casi se solapan) — evaluar un solo período es
      exactamente "qué habría predicho el modelo en ese momento para el
      período siguiente", y barato (~n_items filas, no ~20% de la historia).

    guardar_en_disco: si es False, no escribe el .pkl — usado en replay,
    para no pisar el modelo productivo en cada uno de miles de checkpoints
    históricos (el .pkl productivo se actualiza al final del replay, con
    una corrida en vivo real).

    calcular_metricas_horizonte: si es True, además evalúa el modelo a
    horizontes de 2..N_PASOS_HORIZONTE pasos (ver evaluacion.py) sobre una
    muestra acotada del test set — caro (multiplica el costo de inferencia
    por N_PASOS_HORIZONTE), por eso solo se activa desde job_diario (una vez
    al día), nunca en job_horario ni en el replay.

    Devuelve True si se entrenó y se guardaron métricas/predicciones, False
    si se cortó antes por falta de ítems/datos suficientes (sin excepción,
    ver los logging.error de cada caso) — lo usa
    recolector._entrenar_desde_config para no marcar
    modelos_config.ultimo_entrenamiento_ts en una corrida que en realidad
    no entrenó nada (ej. un modelo recién creado sobre un ítem sin
    suficiente historial todavía).
    """
    model_path = model_path or MODEL_PATHS.get(model_name, os.path.join(MODEL_DIR, f"model_{model_name}.pkl"))

    if item_ids is not None:
        logging.info(f"[{model_name}] Ítems elegidos manualmente: {len(item_ids)}")
    else:
        if ahora_ts is None:
            item_ids = db.obtener_top_items_liquidez(n_items, solo_f2p=solo_f2p)
        else:
            item_ids = db.obtener_top_items_liquidez_hasta(ahora_ts, n_items, solo_f2p=solo_f2p)
        logging.info(f"[{model_name}] Ítems líquidos seleccionados: {len(item_ids)}")
    if not item_ids:
        logging.error(
            f"[{model_name}] Sin ítems disponibles — correr metricas.py antes de entrenar en "
            "modo en vivo (o esperar más historial acumulado en modo replay)."
        )
        return False

    desde_ts = None
    if ventana_dias is not None:
        referencia = ahora_ts if ahora_ts is not None else int(time.time())
        desde_ts = int(referencia - ventana_dias * 86400)

    dataset = build_training_set(db, item_ids, tabla=tabla, hasta_timestamp=ahora_ts, desde_timestamp=desde_ts)
    if dataset.empty:
        logging.error(f"[{model_name}] No se pudo construir el dataset de entrenamiento (sin datos suficientes).")
        return False
    logging.info(f"[{model_name}] Dataset combinado: {len(dataset)} filas, {dataset['item_id'].nunique()} ítems con features")

    if modo_evaluacion == 'walkforward':
        corte = dataset['timestamp_target'].max()
        train = dataset[dataset['timestamp_target'] < corte]
        test = dataset[dataset['timestamp_target'] == corte].copy()
        if train.empty or test.empty:
            logging.error(f"[{model_name}] Dataset insuficiente para walk-forward (train={len(train)}, test={len(test)}).")
            return False
    else:
        corte = dataset['timestamp_target'].quantile(1 - TEST_FRACTION)
        train = dataset[dataset['timestamp_target'] <= corte]
        test = dataset[dataset['timestamp_target'] > corte].copy()
    logging.info(
        f"[{model_name}] Split temporal ({modo_evaluacion}) en timestamp {int(corte)}: "
        f"train={len(train)} filas, test={len(test)} filas"
    )

    feature_cols = [c for c in dataset.columns if c not in NON_FEATURE_COLS]
    X_train, y_train = train[feature_cols], train['target']
    X_test, y_test = test[feature_cols], test['target']

    model = XGBRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=6,
        enable_categorical=True, tree_method='hist',
        random_state=42, verbosity=0,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    test['pred_target'] = y_pred
    # Reconstrucción a gp: precio_predicho = precio_actual * exp(retorno_predicho)
    test['pred_price'] = test['price_actual'].to_numpy() * np.exp(y_pred)

    mae_retorno = mean_absolute_error(y_test, y_pred)
    rmse_retorno = np.sqrt(mean_squared_error(y_test, y_pred))
    mae_gp = mean_absolute_error(test['price_target'], test['pred_price'])
    # Accuracy direccional: de los movimientos que el modelo predijo, ¿en
    # qué fracción acertó el signo (sube/baja), más allá de si acertó la
    # magnitud? Es la pregunta que de verdad importa para decidir si vale
    # comprar o vender, y que hasta ahora no se medía en ningún lado.
    acc_direccional = float((np.sign(test['pred_target']) == np.sign(test['target'])).mean())
    logging.info(
        f"[{model_name}] MAE retorno (log): {mae_retorno:.5f} | RMSE retorno (log): {rmse_retorno:.5f} | "
        f"MAE reconstruido: {mae_gp:.2f} gp | accuracy direccional: {acc_direccional:.3f}"
    )

    # Se guarda un bundle (no solo el modelo): XGBoost, con enable_categorical,
    # codifica item_id/members como los códigos enteros de su dtype 'category'
    # en el momento del fit — si prediccion.py arma esas columnas con una
    # categoría distinta (ej. un solo ítem, código 0) en vez de con exactamente
    # las mismas categorías vistas al entrenar, los splits categóricos del
    # árbol comparan contra el ítem equivocado sin ningún error visible.
    # Guardar `dataset['item_id'].cat.categories` (y lo mismo para members)
    # es lo que le permite a prediccion.py reproducir esa codificación exacta.
    bundle = {
        'model': model,
        'feature_cols': feature_cols,
        'item_id_categories': dataset['item_id'].cat.categories,
        'members_categories': dataset['members'].cat.categories,
        'target_col': 'avg_low_price',
        'tabla': tabla,
        'lags': 5,
        'ma_windows': [3, 6],
    }
    if guardar_en_disco:
        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(bundle, model_path)
        logging.info(f"[{model_name}] Modelo guardado en {model_path}")

    # En vivo, train_ts es el máximo timestamp_target del dataset (como
    # siempre). En replay (ahora_ts dado), se usa el checkpoint mismo, no el
    # máximo del dataset filtrado — que puede quedar antes de ahora_ts si
    # hay un hueco de datos justo en ese momento. Sin esto, existe_checkpoint()
    # no podría reconocer de forma confiable "este checkpoint ya corrió" al
    # reanudar un replay interrumpido (replay_historico.py).
    train_ts = int(ahora_ts) if ahora_ts is not None else int(dataset['timestamp_target'].max())

    # Una fila agregada (item_id=None) + una fila por ítem del test set, para
    # detectar ítems con error desproporcionado dentro del modelo global.
    # horizonte_horas=1: esto siempre es la métrica de 1 paso (ver
    # calcular_metricas_horizonte más abajo para 2..N pasos).
    metricas_filas = [
        (None, train_ts, model_name, float(mae_retorno), float(rmse_retorno), 1, acc_direccional, modo_evaluacion)
    ]
    for item_id, grupo in test.groupby('item_id', observed=True):
        mae_item = mean_absolute_error(grupo['price_target'], grupo['pred_price'])
        rmse_item = np.sqrt(mean_squared_error(grupo['price_target'], grupo['pred_price']))
        acc_item = float((np.sign(grupo['pred_target']) == np.sign(grupo['target'])).mean())
        metricas_filas.append(
            (int(item_id), train_ts, model_name, float(mae_item), float(rmse_item), 1, acc_item, modo_evaluacion)
        )
    db.guardar_metricas_modelo(metricas_filas)
    logging.info(f"[{model_name}] Métricas guardadas en model_metrics ({len(metricas_filas)} filas)")

    predicciones_filas = [
        (int(item_id), int(ts), float(pred), float(actual), float(actual - pred), model_name, modo_evaluacion)
        for item_id, ts, pred, actual in zip(
            test['item_id'], test['timestamp_target'], test['pred_price'], test['price_target'],
        )
    ]
    db.guardar_predicciones(predicciones_filas)
    logging.info(f"[{model_name}] Predicciones guardadas en predicciones ({len(predicciones_filas)} filas)")

    if calcular_metricas_horizonte:
        _evaluar_y_guardar_horizontes(db, bundle, test, model_name, train_ts, modo_evaluacion)

    return True


def _evaluar_y_guardar_horizontes(db, bundle, test, model_name, train_ts, modo_evaluacion):
    """
    Muestrea N_MUESTRAS_HORIZONTE puntos (item_id, timestamp) del test set y
    mide, vía evaluacion.evaluar_horizontes(), el MAE/RMSE/accuracy
    direccional a 2..N_PASOS_HORIZONTE pasos — aparte porque es
    significativamente más caro que la métrica de 1 paso (multiplica el
    costo de inferencia por N_PASOS_HORIZONTE) y solo se llama desde
    job_diario. Solo guarda el agregado (item_id=None) por horizonte, no
    detalle por ítem, para no multiplicar también el volumen de filas.
    """
    try:
        from evaluacion import evaluar_horizontes
    except ImportError as e:
        logging.error(f"[{model_name}] No se pudo importar evaluacion.py, se omiten las métricas de horizonte: {e}")
        return

    candidatos = test[['item_id', 'timestamp_target']].drop_duplicates()
    if candidatos.empty:
        return
    muestra = candidatos.sample(n=min(N_MUESTRAS_HORIZONTE, len(candidatos)), random_state=42)
    puntos_eval = list(zip(muestra['item_id'].astype(int), muestra['timestamp_target'].astype(int)))

    resultado = evaluar_horizontes(db, bundle, puntos_eval, n_pasos=N_PASOS_HORIZONTE, tabla='precios_1h')
    if resultado.empty:
        logging.warning(f"[{model_name}] Sin datos suficientes para evaluar métricas de horizonte.")
        return

    filas_horizonte = []
    for horizonte, grupo in resultado.groupby('horizonte'):
        mae_h = float(grupo['error_abs'].mean())
        rmse_h = float(np.sqrt((grupo['error_abs'] ** 2).mean()))
        acc_h = float(grupo['acierto_direccional'].mean())
        filas_horizonte.append((None, train_ts, model_name, mae_h, rmse_h, int(horizonte), acc_h, modo_evaluacion))

    db.guardar_metricas_modelo(filas_horizonte)
    logging.info(
        f"[{model_name}] Métricas por horizonte guardadas ({len(filas_horizonte)} horizontes, "
        f"sobre {len(puntos_eval)} puntos muestreados)"
    )


def _clasificar_retorno(target, umbral_pct=UMBRAL_CLASIF_PCT):
    """
    Convierte el log-retorno continuo (el mismo target que usa el
    regresor) en 3 clases: 0=baja, 1=estable, 2=sube, según un umbral en %
    — ej. con umbral_pct=0.5, un retorno de +0.8% clasifica 'sube', uno de
    +0.1% clasifica 'estable'. Un umbral muy chico satura el dataset de
    'sube'/'baja' con ruido; uno muy grande lo satura de 'estable' y deja
    pocos ejemplos para aprender movimientos reales — 0.5% es un punto de
    partida razonable dado el MAE típico medido (~3-9% en espacio log), no
    un valor validado exhaustivamente.
    """
    umbral = umbral_pct / 100
    return np.where(target > umbral, 2, np.where(target < -umbral, 0, 1))


def entrenar_clasificador_direccional(
    db,
    n_items=N_ITEMS_LIQUIDOS,
    umbral_pct=UMBRAL_CLASIF_PCT,
    model_name=MODEL_NAME_CLASIFICADOR,
    model_path=None,
    ahora_ts=None,
    guardar_en_disco=False,
    modo_evaluacion='holdout',
    solo_f2p=False,
    precio_minimo=None,
    excluir_item_ids=None,
    item_ids=None,
    tabla='precios_1h',
    ventana_dias=None,
):
    """
    Entrena un XGBClassifier de 3 clases (baja/estable/sube) sobre el mismo
    dataset/features/split que entrenar_modelo_global — a diferencia de
    derivar la dirección del signo de una regresión (que optimiza error
    cuadrático, no accuracy de signo — los movimientos chicos cerca de cero,
    que son la mayoría, pesan igual que uno grande en esa pérdida), este
    modelo optimiza directamente la clase correcta.

    item_ids (opcional): ver entrenar_modelo_global — lista explícita de
    ítems, en vez de derivar el universo desde obtener_top_items_liquidez
    con n_items/solo_f2p/precio_minimo/excluir_item_ids (que se ignoran
    por completo cuando se pasa).

    tabla/ventana_dias: ver entrenar_modelo_global — misma granularidad de
    precios y misma ventana de historial configurables, mismo motivo
    (bundle['tabla'] evita el train/serve skew de inferir con la tabla
    equivocada).

    solo_f2p: restringe el universo de ítems a free-to-play (ver
    obtener_top_items_liquidez) — ej. n_items=10, solo_f2p=True entrena
    solo sobre las runas/materiales F2P más líquidos, un universo mucho más
    chico y homogéneo que los 200 ítems (F2P+members) por default.

    precio_minimo: descarta ítems con avg_low_price por debajo de ese piso
    (ver obtener_top_items_liquidez) — sin esto, el universo F2P más líquido
    queda dominado por runas de 2-5 gp cuyo margen neto por unidad es tan
    chico que el impuesto GE se lo come en la mitad de los trades incluso
    acertando la dirección (validado en backtest: seguir la señal ahí no le
    gana a comprar a ciegas).

    excluir_item_ids: ver obtener_top_items_liquidez — descarta ítems
    puntuales aunque califiquen por liquidez/precio (ej. Steel bar, que en
    el backtest walk-forward de 90 días perdió plata con las dos
    estrategias, señal y compra ciega).

    Guarda en model_metrics la accuracy "direccional pura" — excluyendo los
    casos donde la clase real o la predicha fue 'estable', para que sea
    comparable con la accuracy_direccional del regresor (que no tiene
    noción de 'estable', solo signo) — con mae/rmse en NULL (no aplican a
    un clasificador). No persiste filas en `predicciones` (no tiene
    "predicted_price" continuo, esa tabla queda intacta).

    guardar_en_disco: si es True, SÍ persiste un bundle .pkl en model_path
    (o MODEL_PATHS[model_name] por default) — mismo patrón que
    entrenar_modelo_global (ver su docstring). Esto es lo que usa
    recolector.job_horario para reentrenar y publicar en producción
    MODEL_NAME_CLASIF_F2P_100GP (los 10 ítems F2P más líquidos con
    precio_minimo=100 y Steel bar excluido — ver el comentario junto a esa
    constante) cada hora. Sigue sin alimentar alertas.py (fuera de alcance
    de esa promoción) — el consumidor en vivo es
    prediccion.pronosticar_clase_item() + el tab del dashboard.
    MODEL_NAME_CLASIFICADOR (200 ítems F2P+members) sigue siendo solo
    comparación en model_metrics, no se le pasa guardar_en_disco=True en
    ningún caller productivo. Default False a propósito: backtest.py y
    replay_historico.py llaman esta función potencialmente miles de veces
    por checkpoint y no deben pisar el .pkl productivo en cada una.

    ahora_ts/modo_evaluacion: mismo significado que en entrenar_modelo_global
    — permite compararlo también en modo walkforward vía replay_historico.py
    si hiciera falta, aunque hoy no está enganchado al loop de replay.

    Devuelve (modelo, test) — `test` es el DataFrame del test set (NUNCA
    visto en entrenamiento) con una columna 'clase_predicha' agregada,
    listo para que backtest.py simule trades sobre datos genuinamente
    held-out sin tener que reentrenar ni volver a correr el split. None si
    no se pudo entrenar.
    """
    model_path = model_path or MODEL_PATHS.get(model_name, os.path.join(MODEL_DIR, f"model_{model_name}.pkl"))

    if item_ids is not None:
        logging.info(f"[{model_name}] Ítems elegidos manualmente: {len(item_ids)}")
    else:
        if ahora_ts is None:
            item_ids = db.obtener_top_items_liquidez(n_items, solo_f2p=solo_f2p, precio_minimo=precio_minimo, excluir_item_ids=excluir_item_ids)
        else:
            item_ids = db.obtener_top_items_liquidez_hasta(ahora_ts, n_items, solo_f2p=solo_f2p, precio_minimo=precio_minimo, excluir_item_ids=excluir_item_ids)
        logging.info(f"[{model_name}] Ítems líquidos seleccionados: {len(item_ids)}")
    if not item_ids:
        logging.error(f"[{model_name}] Sin ítems disponibles.")
        return None, None

    desde_ts = None
    if ventana_dias is not None:
        referencia = ahora_ts if ahora_ts is not None else int(time.time())
        desde_ts = int(referencia - ventana_dias * 86400)

    dataset = build_training_set(db, item_ids, tabla=tabla, hasta_timestamp=ahora_ts, desde_timestamp=desde_ts)
    if dataset.empty:
        logging.error(f"[{model_name}] No se pudo construir el dataset de entrenamiento.")
        return None, None

    if modo_evaluacion == 'walkforward':
        corte = dataset['timestamp_target'].max()
        train = dataset[dataset['timestamp_target'] < corte]
        test = dataset[dataset['timestamp_target'] == corte].copy()
    else:
        corte = dataset['timestamp_target'].quantile(1 - TEST_FRACTION)
        train = dataset[dataset['timestamp_target'] <= corte]
        test = dataset[dataset['timestamp_target'] > corte].copy()
    if train.empty or test.empty:
        logging.error(f"[{model_name}] Dataset insuficiente (train={len(train)}, test={len(test)}).")
        return None, None

    feature_cols = [c for c in dataset.columns if c not in NON_FEATURE_COLS]
    y_train_clase = _clasificar_retorno(train['target'].to_numpy(), umbral_pct)
    y_test_clase = _clasificar_retorno(test['target'].to_numpy(), umbral_pct)

    modelo = XGBClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=6,
        enable_categorical=True, tree_method='hist',
        objective='multi:softprob', num_class=3,
        random_state=42, verbosity=0,
    )
    modelo.fit(train[feature_cols], y_train_clase)
    y_pred_clase = modelo.predict(test[feature_cols])
    test['clase_predicha'] = y_pred_clase  # 0=baja, 1=estable, 2=sube — para backtest.py
    test['clase_real'] = y_test_clase  # idem, para poder medir accuracy sin recalcular después

    accuracy_multiclase = float((y_pred_clase == y_test_clase).mean())
    # Direccional "pura": solo entre los casos donde ni la clase real ni la
    # predicha fue 'estable' — comparable con la accuracy_direccional del
    # regresor, que solo mide signo.
    mascara_no_estable = (y_test_clase != 1) & (y_pred_clase != 1)
    n_no_estable = int(mascara_no_estable.sum())
    acc_direccional_pura = (
        float((y_pred_clase[mascara_no_estable] == y_test_clase[mascara_no_estable]).mean())
        if n_no_estable > 0 else None
    )

    dist = {label: int((y_test_clase == code).sum()) for code, label in CLASE_LABELS.items()}
    logging.info(
        f"[{model_name}] accuracy multi-clase: {accuracy_multiclase:.3f} | "
        f"accuracy direccional pura (excl. 'estable', n={n_no_estable}): "
        f"{'N/A' if acc_direccional_pura is None else f'{acc_direccional_pura:.3f}'} | "
        f"distribución test: {dist}"
    )

    train_ts = int(ahora_ts) if ahora_ts is not None else int(dataset['timestamp_target'].max())
    db.guardar_metricas_modelo([
        (None, train_ts, model_name, None, None, 1, acc_direccional_pura, modo_evaluacion)
    ])
    logging.info(f"[{model_name}] Métrica guardada en model_metrics")

    # Persistencia opcional del bundle — mismo patrón que entrenar_modelo_global
    # (ver su comentario sobre por qué hace falta guardar item_id_categories/
    # members_categories además del modelo, no solo el modelo, con XGBoost +
    # enable_categorical). guardar_en_disco=False por default para no pisar
    # el .pkl productivo desde backtest.py/replay_historico.py, que llaman
    # esta función solo por (modelo, test), potencialmente muchas veces.
    if guardar_en_disco:
        bundle = {
            'model': modelo,
            'feature_cols': feature_cols,
            'item_id_categories': dataset['item_id'].cat.categories,
            'members_categories': dataset['members'].cat.categories,
            'target_col': 'avg_low_price',
            'tabla': tabla,
            'lags': 5,
            'ma_windows': [3, 6],
            'umbral_pct': umbral_pct,
            'clase_labels': CLASE_LABELS,
        }
        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(bundle, model_path)
        logging.info(f"[{model_name}] Modelo guardado en {model_path}")

    return modelo, test


def _kwargs_desde_modelo_config(cfg):
    """
    Traduce una fila de modelos_config (ver
    base_de_datos.OSRSBaseDatos._fila_a_modelo_config) a los kwargs
    comunes que entrenar_modelo_global/entrenar_clasificador_direccional
    esperan — compartido entre recolector._entrenar_desde_config
    (reentrenamiento en vivo, programado) y
    replay_historico.ejecutar_replay_modelo (walk-forward inicial al crear
    un modelo), para no tener el mismo mapeo modo_seleccion/tipo
    duplicado en dos archivos (antes solo vivía en recolector.py).

    No incluye ahora_ts/guardar_en_disco/modo_evaluacion/
    calcular_metricas_horizonte — esos dependen de PARA QUÉ se está
    entrenando (en vivo vs. un checkpoint del pasado), los agrega el
    llamador vía entrenar_desde_config(db, cfg, **overrides).
    """
    kwargs = dict(
        model_name=cfg['model_id'], tabla=cfg.get('tabla') or 'precios_1h',
        ventana_dias=cfg.get('ventana_dias'),
    )
    if cfg['modo_seleccion'] == 'manual':
        kwargs['item_ids'] = cfg['item_ids']
    else:
        kwargs['n_items'] = cfg['n_items']
        kwargs['solo_f2p'] = cfg['solo_f2p']
        if cfg['tipo'] == 'clasificador':
            kwargs['precio_minimo'] = cfg['precio_minimo']
            kwargs['excluir_item_ids'] = cfg['excluir_item_ids']
    if cfg['tipo'] == 'clasificador' and cfg['umbral_pct'] is not None:
        kwargs['umbral_pct'] = cfg['umbral_pct']
    return kwargs


def entrenar_desde_config(db, cfg, **overrides):
    """
    Llama entrenar_modelo_global o entrenar_clasificador_direccional según
    cfg['tipo'], con los kwargs derivados de _kwargs_desde_modelo_config +
    `overrides` (típicamente ahora_ts, guardar_en_disco, modo_evaluacion,
    calcular_metricas_horizonte).

    Devuelve exactamente lo que devuelve la función subyacente sin
    envolverlo — True/False para el regresor, (modelo, test) para el
    clasificador — el llamador decide cómo interpretar el éxito en cada
    caso (ver recolector._entrenar_desde_config y
    replay_historico.ejecutar_replay_modelo).
    """
    kwargs = _kwargs_desde_modelo_config(cfg)
    kwargs.update(overrides)
    if cfg['tipo'] == 'clasificador':
        return entrenar_clasificador_direccional(db, **kwargs)
    return entrenar_modelo_global(db, **kwargs)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s |--| %(levelname)s |--| %(message)s')
    db = OSRSBaseDatos('data/osrs_ge.db')
    entrenar_modelo_global(db, model_name=MODEL_NAME_HORARIO)
