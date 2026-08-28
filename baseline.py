"""
baseline.py — modelos "ingenuos" de referencia, sin ningún entrenamiento,
para saber si XGBoost (entrenador.py) realmente agrega algo sobre reglas
triviales. Se evalúan sobre EXACTAMENTE el mismo dataset/split que
entrenador.py (misma build_training_set, mismo corte de holdout) y se
guardan en model_metrics con el mismo esquema — así se comparan directo
en el dashboard (tab "Calidad del modelo") contra global_horario/global_diario,
sin tener que armar nada nuevo del lado de la visualización.

Dos reglas:
- 'baseline_flat': predice que el precio no cambia (retorno=0) — el
  benchmark clásico de "los precios siguen un camino aleatorio, la mejor
  predicción de mañana es el precio de hoy". No tiene signo (predice
  exactamente cero), así que no se le mide accuracy_direccional (queda
  NULL) — solo sirve de piso para MAE/RMSE.
- 'baseline_momentum': predice que el retorno del período siguiente va a
  ser igual al del período anterior (persistencia/momentum) — sí tiene
  signo, así que es directamente comparable en accuracy_direccional contra
  el modelo real.

Sin este piso no hay forma de saber si un accuracy_direccional de ~0.42
(lo que viene dando el modelo real) es genuinamente malo o si ninguna regla
razonable le gana mucho al azar en estos ítems a resolución de 1 hora.
"""

import logging

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

from base_de_datos import OSRSBaseDatos
from entrenador import N_ITEMS_LIQUIDOS, TEST_FRACTION
from preprocesamiento import build_training_set

MODEL_NAME_FLAT = "baseline_flat"
MODEL_NAME_MOMENTUM = "baseline_momentum"


def evaluar_baselines(db, n_items=N_ITEMS_LIQUIDOS, tabla='precios_1h'):
    """
    Arma el mismo dataset que entrenar_modelo_global (mismos ítems líquidos,
    mismas features, mismo split temporal 80/20) y calcula, sobre el test
    set, las métricas de las dos reglas triviales — sin fitear nada: el
    "predicho" sale directo de columnas que ya están en el dataset
    (price_lag_1 = log-precio del período anterior, price_actual = precio
    del período actual).

    Guarda ambas en model_metrics (horizonte_horas=1, modo_evaluacion=
    'holdout', igual que el 1-paso de entrenador.py) y devuelve un dict con
    las métricas agregadas de cada una para poder loguearlas/compararlas
    de una.
    """
    item_ids = db.obtener_top_items_liquidez(n_items)
    if not item_ids:
        logging.error("baseline.py: resumen_actual vacía — correr metricas.py antes.")
        return {}

    dataset = build_training_set(db, item_ids, tabla=tabla)
    if dataset.empty:
        logging.error("baseline.py: no se pudo construir el dataset (sin datos suficientes).")
        return {}

    corte = dataset['timestamp_target'].quantile(1 - TEST_FRACTION)
    test = dataset[dataset['timestamp_target'] > corte].copy()
    if test.empty:
        logging.error("baseline.py: test set vacío con el split configurado.")
        return {}

    train_ts = int(dataset['timestamp_target'].max())

    # retorno_anterior = log(price_actual) - price_lag_1 (price_lag_1 ya
    # está en espacio log, ver preprocesamiento.construir_features).
    log_precio_actual = np.log(test['price_actual'])
    retorno_anterior = log_precio_actual - test['price_lag_1']

    resultados = {}

    # --- baseline_flat: predicted_target = 0 ---
    mae_flat = mean_absolute_error(test['target'], np.zeros(len(test)))
    rmse_flat = np.sqrt(mean_squared_error(test['target'], np.zeros(len(test))))
    resultados[MODEL_NAME_FLAT] = {'mae': mae_flat, 'rmse': rmse_flat, 'accuracy_direccional': None}
    logging.info(f"[{MODEL_NAME_FLAT}] MAE retorno (log): {mae_flat:.5f} | RMSE retorno (log): {rmse_flat:.5f}")

    # --- baseline_momentum: predicted_target = retorno_anterior ---
    mae_mom = mean_absolute_error(test['target'], retorno_anterior)
    rmse_mom = np.sqrt(mean_squared_error(test['target'], retorno_anterior))
    acc_mom = float((np.sign(retorno_anterior) == np.sign(test['target'])).mean())
    resultados[MODEL_NAME_MOMENTUM] = {'mae': mae_mom, 'rmse': rmse_mom, 'accuracy_direccional': acc_mom}
    logging.info(
        f"[{MODEL_NAME_MOMENTUM}] MAE retorno (log): {mae_mom:.5f} | RMSE retorno (log): {rmse_mom:.5f} | "
        f"accuracy direccional: {acc_mom:.3f}"
    )

    filas = [
        (None, train_ts, nombre, float(m['mae']), float(m['rmse']), 1, m['accuracy_direccional'], 'holdout')
        for nombre, m in resultados.items()
    ]
    db.guardar_metricas_modelo(filas)
    logging.info(f"baseline.py: {len(filas)} fila(s) agregadas guardadas en model_metrics")

    return resultados


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s |--| %(levelname)s |--| %(message)s')
    db = OSRSBaseDatos('data/osrs_ge.db')
    evaluar_baselines(db)
