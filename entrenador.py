import logging
import os
import time

import joblib
import numpy as np
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error

from base_de_datos import OSRSBaseDatos
from metricas import GE_TAX_RATE
from preprocesamiento import build_training_set, FEATURES_VERSION, PASO_SEGUNDOS_POR_TABLA

MODEL_DIR = "models"

# Nombres de referencia para model_name/model_version (model_metrics.model_name
# y predicciones.model_version ya soportan taggear cualquier cantidad de
# variantes sin pisarse) — NO son modelos sembrados por default: no hay
# ningún modelo hasta que el usuario cree uno desde "Mis modelos" en la app
# de escritorio (ver escritorio/paginas/pagina_modelos.py, que hoy crea
# siempre un par regresor+clasificador con cadencia horaria y tabla
# precios_1h). Estos nombres solo importan si el usuario elige exactamente
# uno de ellos (o si un modelo apunta a la ruta fija de MODEL_PATHS más
# abajo) — el resto del pipeline (job_horario/job_diario, dashboard.py) es
# genérico sobre cualquier model_name.
MODEL_NAME_HORARIO = "global_horario"
MODEL_NAME_DIARIO = "global_diario"

# Nombres de referencia para el clasificador direccional
# (entrenar_clasificador_direccional) — mismo criterio que arriba, ninguno
# sembrado por default. f2p10_100gp_clasif es el nombre usado en la
# investigación de calidad de modelos de 2026-08-30/31 (10 ítems F2P más
# líquidos, precio_minimo=100, excluir_item_ids=[2353, 449, 453] — Steel
# bar, Adamantite ore, Coal): un test de significancia por permutación
# sobre 90 días/5.639 trades mostró que esa configuración gana
# significativamente más que el azar (p=0.002) comprando/vendiendo a 1h,
# pero con la ganancia muy concentrada en un solo ítem (Cosmic rune, 91%
# del total) y con Coal/Adamantite ore restando plata de forma consistente
# (por eso están excluidos) — si se recrea un modelo así desde la app, ese
# es el criterio de selección validado, aunque hoy la app solo permite
# elegir ítems a mano (modo_seleccion='liquidez' no está expuesto en la UI,
# ver pagina_modelos.py), así que replicarlo exacto requiere pasar esos
# mismos parámetros vía crear_modelo_config directamente, no desde la UI.
MODEL_NAME_CLASIFICADOR = "global_horario_clasif"
MODEL_NAME_CLASIF_F2P = "f2p10_clasif"
MODEL_NAME_CLASIF_F2P_100GP = "f2p10_100gp_clasif"

# Costo mínimo de un round-trip que NO captura el spread: el impuesto del
# Grand Exchange sobre la venta (metricas.GE_TAX_RATE). Un movimiento
# predicho por debajo de esto no alcanza para cubrir el impuesto, así que
# comprar por esa señal es perder plata salvo que además se capture el
# spread (comprar con orden en la punta baja y vender en la alta, con las
# dos órdenes efectivamente completadas — ver el docstring de backtest.py).
UMBRAL_COSTO_PCT = GE_TAX_RATE * 100

# log-retorno %: |retorno| > esto para no contar como "estable".
# Subido de 0.5 a 1.0 después de medir la distribución real: el movimiento
# horario MEDIANO de los ítems líquidos está entre 0.2% y 0.9% según el ítem,
# así que con 0.5% la mitad de las clases 'sube'/'baja' eran ruido de
# redondeo del precio entero, y ninguna de ellas cubría el 2% de impuesto.
# 1.0% es un compromiso: sigue por debajo del costo (ver UMBRAL_COSTO_PCT y
# el warning de _avisar_umbral_vs_costo) pero deja suficientes ejemplos de
# cada clase para que el clasificador aprenda algo — subirlo hasta 2% deja
# entre 1% y 10% de las horas como movimiento, y el modelo colapsa a
# predecir 'estable' siempre. Que la señal cubra el costo es una decisión
# del punto de consumo (cruzarla con margen_neto del screener), no del
# etiquetado.
UMBRAL_CLASIF_PCT = 1.0
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

# Dirección asociada a cada clase del clasificador (ver CLASE_LABELS):
# 'estable' es 0, o sea "no me juego por ninguna dirección".
_DIRECCION_POR_CLASE = {0: -1, 1: 0, 2: 1}


def accuracy_direccional(target_real, direccion_predicha, umbral_pct=UMBRAL_CLASIF_PCT):
    """
    Fracción de aciertos de DIRECCIÓN sobre los períodos en que el precio
    efectivamente se movió (|log-retorno real| > umbral_pct%).

    Definición única y compartida entre el regresor y el clasificador — sin
    esto las dos métricas se guardaban en la misma columna de model_metrics
    calculadas sobre poblaciones distintas, y no eran comparables:

    - El regresor usaba `sign(pred) == sign(real)` sobre TODAS las filas. En
      los períodos sin movimiento el retorno real es exactamente 0 (entre el
      7% y el 35% de las horas según el ítem, medido sobre datos reales:
      los precios son enteros y muchos ítems no se mueven en una hora), su
      signo es 0, y el modelo nunca predice exactamente 0 — así que TODOS
      esos empates contaban como error. Medido: la misma corrida daba 31.7%
      con esa fórmula y 55.9% con esta.
    - El clasificador excluía los casos 'estable' tanto reales como
      predichos, lo que además de cambiar la población le regalaba los casos
      en que se abstenía.

    Acá los períodos sin movimiento real quedan fuera de la cuenta (no hay
    dirección que acertar), pero abstenerse cuando SÍ hubo movimiento cuenta
    como error: el modelo tuvo la oportunidad y no la vio.

    `direccion_predicha` puede ser un retorno continuo (regresor, se usa su
    signo) o ya un -1/0/+1. Devuelve (accuracy, n_evaluado); accuracy es
    None si no hubo ningún movimiento por encima del umbral.
    """
    target_real = np.asarray(target_real, dtype=float)
    direccion_predicha = np.asarray(direccion_predicha, dtype=float)
    umbral = umbral_pct / 100

    hubo_movimiento = np.abs(target_real) > umbral
    n = int(hubo_movimiento.sum())
    if n == 0:
        return None, 0
    aciertos = np.sign(direccion_predicha[hubo_movimiento]) == np.sign(target_real[hubo_movimiento])
    return float(aciertos.mean()), n


def _direcciones_desde_clases(clases):
    """Convierte las clases 0/1/2 del clasificador en -1/0/+1 para poder
    pasárselas a accuracy_direccional() con la misma semántica que el
    retorno continuo del regresor."""
    return np.array([_DIRECCION_POR_CLASE[int(c)] for c in clases], dtype=float)


def _avisar_umbral_vs_costo(model_name, umbral_pct):
    """
    Loguea un warning si el umbral con el que se etiquetan las clases del
    clasificador queda por debajo del costo del impuesto GE — sus señales
    'sube' no cubren el costo de operar salvo que además se capture el
    spread. No cambia el entrenamiento: es una decisión del usuario, pero
    tiene que ser visible y no un default silencioso (ver UMBRAL_COSTO_PCT).
    """
    if umbral_pct < UMBRAL_COSTO_PCT:
        logging.warning(
            f"[{model_name}] umbral_pct={umbral_pct}% < {UMBRAL_COSTO_PCT}% de impuesto GE: "
            "una señal 'sube' de esa magnitud no cubre el impuesto de la venta por sí sola — "
            "solo es rentable si además se captura el spread (cruzarla con margen_neto del "
            "screener antes de operar)."
        )


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
    precio_minimo=None,
    excluir_item_ids=None,
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
    solo_f2p/precio_minimo/excluir_item_ids — es lo que permite que un
    modelo definido por el usuario (base_de_datos.modelos_config con
    modo_seleccion='manual', ver recolector._entrenar_desde_config) entrene
    sobre exactamente los ítems que eligió a mano, no sobre un ranking de
    liquidez. Cuando se pasa, el resto de los filtros de selección se
    ignoran por completo — el default (None) mantiene el comportamiento de
    siempre.

    precio_minimo/excluir_item_ids: ver
    entrenar_clasificador_direccional/obtener_top_items_liquidez — mismos
    filtros de selección por liquidez, ahora también disponibles acá (antes
    solo existían en el clasificador; un modelo modo_seleccion='liquidez'
    con estos filtros configurados los ignoraba silenciosamente si era de
    tipo 'regresor', ver _kwargs_desde_modelo_config más abajo).

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
            item_ids = db.obtener_top_items_liquidez(n_items, solo_f2p=solo_f2p, precio_minimo=precio_minimo, excluir_item_ids=excluir_item_ids)
        else:
            item_ids = db.obtener_top_items_liquidez_hasta(ahora_ts, n_items, solo_f2p=solo_f2p, precio_minimo=precio_minimo, excluir_item_ids=excluir_item_ids)
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
    rmse_gp = np.sqrt(mean_squared_error(test['price_target'], test['pred_price']))
    # Accuracy direccional: de los períodos en que el precio SE MOVIÓ, ¿en
    # qué fracción acertó el signo, más allá de la magnitud? Es la pregunta
    # que de verdad importa para decidir si vale comprar o vender. Ver
    # accuracy_direccional() sobre por qué los períodos sin movimiento no
    # entran en la cuenta (antes contaban todos como error).
    acc_direccional, n_direccional = accuracy_direccional(test['target'], test['pred_target'])
    # Piso de referencia obligado: el error del baseline trivial "el precio
    # no cambia" sobre el mismo test set. Si el modelo no le gana a esto,
    # está agregando ruido y conviene saberlo en el log, no descubrirlo
    # después en el dashboard (ver baseline.py).
    mae_flat = float(np.abs(y_test).mean())
    logging.info(
        f"[{model_name}] MAE retorno (log): {mae_retorno:.5f} (baseline 'no cambia': {mae_flat:.5f}"
        f"{' — el modelo NO le gana' if mae_retorno >= mae_flat else ''}) | "
        f"MAE reconstruido: {mae_gp:.2f} gp | accuracy direccional: "
        f"{'N/A' if acc_direccional is None else f'{acc_direccional:.3f}'} sobre {n_direccional} movimientos"
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
        # Ver preprocesamiento.FEATURES_VERSION: prediccion.py lo verifica
        # antes de inferir. Un bundle viejo con features en niveles
        # alimentado con las nuevas no falla — reindex() rellena con NaN lo
        # que no encuentra — sino que devuelve ruido en silencio.
        'features_version': FEATURES_VERSION,
        'paso_segundos': PASO_SEGUNDOS_POR_TABLA.get(tabla, 3600),
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
    # mae/rmse en gp también en la fila agregada (antes iban en espacio
    # log-retorno mientras las filas por ítem iban en gp: dos unidades en la
    # misma columna). El error en espacio log vive ahora en
    # mae_retorno/rmse_retorno — ver base_de_datos.guardar_metricas_modelo.
    metricas_filas = [
        (None, train_ts, model_name, float(mae_gp), float(rmse_gp), 1, acc_direccional,
         modo_evaluacion, n_direccional, float(mae_retorno), float(rmse_retorno))
    ]
    for item_id, grupo in test.groupby('item_id', observed=True):
        mae_item = mean_absolute_error(grupo['price_target'], grupo['pred_price'])
        rmse_item = np.sqrt(mean_squared_error(grupo['price_target'], grupo['pred_price']))
        acc_item, n_item = accuracy_direccional(grupo['target'], grupo['pred_target'])
        metricas_filas.append(
            (int(item_id), train_ts, model_name, float(mae_item), float(rmse_item), 1, acc_item,
             modo_evaluacion, n_item,
             float(mean_absolute_error(grupo['target'], grupo['pred_target'])),
             float(np.sqrt(mean_squared_error(grupo['target'], grupo['pred_target']))))
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

    # La tabla sale del bundle, NO fija en 'precios_1h': un modelo entrenado
    # sobre precios_5m o precios_6h evaluado contra el historial horario
    # mide otra cosa (los "pasos" del pronóstico recursivo son de otra
    # duración) y guardaba esa métrica como si fuera del modelo.
    tabla = bundle.get('tabla', 'precios_1h')
    resultado = evaluar_horizontes(db, bundle, puntos_eval, n_pasos=N_PASOS_HORIZONTE, tabla=tabla)
    if resultado.empty:
        logging.warning(f"[{model_name}] Sin datos suficientes para evaluar métricas de horizonte.")
        return

    filas_horizonte = []
    for horizonte, grupo in resultado.groupby('horizonte'):
        mae_h = float(grupo['error_abs'].mean())
        rmse_h = float(np.sqrt((grupo['error_abs'] ** 2).mean()))
        # Solo los puntos donde el precio efectivamente se movió, igual
        # criterio que accuracy_direccional() — evaluacion.py ya marca los
        # empates con acierto_direccional=None para que queden fuera.
        con_movimiento = grupo[grupo['acierto_direccional'].notna()]
        acc_h = float(con_movimiento['acierto_direccional'].mean()) if not con_movimiento.empty else None
        filas_horizonte.append(
            (None, train_ts, model_name, mae_h, rmse_h, int(horizonte), acc_h,
             modo_evaluacion, int(len(con_movimiento)), None, None)
        )

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
    visto en entrenamiento) con columnas 'clase_predicha' y 'prob_sube'
    (probabilidad softmax de la clase 'sube', para dimensionamiento por
    confianza — ver busqueda_calibrada.py) agregadas, listo para que
    backtest.py simule trades sobre datos genuinamente held-out sin tener
    que reentrenar ni volver a correr el split. None si no se pudo
    entrenar.
    """
    model_path = model_path or MODEL_PATHS.get(model_name, os.path.join(MODEL_DIR, f"model_{model_name}.pkl"))
    _avisar_umbral_vs_costo(model_name, umbral_pct)

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
    proba = modelo.predict_proba(test[feature_cols])
    test['clase_predicha'] = y_pred_clase  # 0=baja, 1=estable, 2=sube — para backtest.py
    test['clase_real'] = y_test_clase  # idem, para poder medir accuracy sin recalcular después
    # Probabilidad de la clase 'sube' (softmax, no solo el argmax) — a
    # diferencia de clase_predicha (sí/no), esto es una medida continua de
    # confianza que un backtest puede usar para dimensionar la posición
    # (más confianza -> más plata en esa señal) en vez de apostar el mismo
    # monto fijo a toda señal 'sube', sea 0.34 o 0.95 de probabilidad. No
    # cambia ningún consumidor existente (columna nueva, se ignora si no se
    # usa) — ver busqueda_calibrada.py para el primer uso real.
    test['prob_sube'] = proba[:, 2]

    accuracy_multiclase = float((y_pred_clase == y_test_clase).mean())
    # MISMA definición que el regresor (entrenador.accuracy_direccional):
    # sobre los períodos en que el precio realmente se movió más que el
    # umbral, ¿acertó la dirección? Antes esta métrica excluía además los
    # casos en que el modelo predecía 'estable', o sea que abstenerse no
    # costaba nada — y se guardaba en la misma columna que la del regresor,
    # calculada sobre otra población. Parte de la "ventaja del clasificador
    # sobre el regresor" que estaba documentada venía de esa diferencia de
    # definición, no del modelo.
    acc_direccional_pura, n_movimientos = accuracy_direccional(
        test['target'].to_numpy(), _direcciones_desde_clases(y_pred_clase), umbral_pct,
    )

    # Precisión de la señal accionable: de las veces que dijo 'sube',
    # ¿cuántas subieron de verdad? Es lo que decide si conviene operarla
    # (la accuracy global mezcla eso con los aciertos de 'baja', que en este
    # pipeline no se operan: no se puede vender en corto en el GE).
    dijo_sube = y_pred_clase == 2
    precision_sube = (
        float((y_test_clase[dijo_sube] == 2).mean()) if int(dijo_sube.sum()) > 0 else None
    )

    dist = {label: int((y_test_clase == code).sum()) for code, label in CLASE_LABELS.items()}
    logging.info(
        f"[{model_name}] accuracy multi-clase: {accuracy_multiclase:.3f} | "
        f"accuracy direccional (movimientos > {umbral_pct}%, n={n_movimientos}): "
        f"{'N/A' if acc_direccional_pura is None else f'{acc_direccional_pura:.3f}'} | "
        f"precisión de 'sube' (n={int(dijo_sube.sum())}): "
        f"{'N/A' if precision_sube is None else f'{precision_sube:.3f}'} | "
        f"distribución test: {dist}"
    )

    train_ts = int(ahora_ts) if ahora_ts is not None else int(dataset['timestamp_target'].max())
    db.guardar_metricas_modelo([
        (None, train_ts, model_name, None, None, 1, acc_direccional_pura,
         modo_evaluacion, n_movimientos, None, None)
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
            'features_version': FEATURES_VERSION,
            'paso_segundos': PASO_SEGUNDOS_POR_TABLA.get(tabla, 3600),
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
        # precio_minimo/excluir_item_ids ahora los soportan tanto el
        # regresor como el clasificador (ver entrenar_modelo_global) — antes
        # esto solo se pasaba para 'clasificador' y un modelo 'regresor' en
        # modo liquidez con estos filtros configurados los ignoraba sin
        # avisar (bug encontrado en la primera campaña de
        # busqueda_hiperparametros.py: 3 filas entrenaron sobre runas/
        # Feather pese a precio_minimo=100).
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
