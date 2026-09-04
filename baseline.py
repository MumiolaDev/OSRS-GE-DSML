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
from entrenador import N_ITEMS_LIQUIDOS, TEST_FRACTION, accuracy_direccional
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

    # ret_lag_1 ya ES el log-retorno del período anterior
    # (log(p_t) - log(p_t-1), ver preprocesamiento.construir_features): desde
    # que las features son relativas no hay que reconstruirlo restando un
    # nivel absoluto, que es lo que hacía antes con price_lag_1.
    retorno_anterior = test['ret_lag_1']
    precio_actual = test['price_actual']

    def _metricas(pred_retorno, nombre):
        """MAE/RMSE en gp (la unidad de model_metrics.mae) y en espacio
        log-retorno (mae_retorno/rmse_retorno), más la accuracy direccional
        con la MISMA definición que entrenador.py — sin esto el baseline no
        es comparable contra el modelo real, que es su única razón de ser."""
        pred_precio = precio_actual * np.exp(pred_retorno)
        acc, n = accuracy_direccional(test['target'], pred_retorno)
        return {
            'mae': float(mean_absolute_error(test['price_target'], pred_precio)),
            'rmse': float(np.sqrt(mean_squared_error(test['price_target'], pred_precio))),
            'mae_retorno': float(mean_absolute_error(test['target'], pred_retorno)),
            'rmse_retorno': float(np.sqrt(mean_squared_error(test['target'], pred_retorno))),
            'accuracy_direccional': acc,
            'n_evaluado': n,
            'nombre': nombre,
        }

    resultados = {}

    # --- baseline_flat: predicted_target = 0 ---
    # Sin signo (predice exactamente cero), así que su accuracy direccional
    # queda en None a propósito: solo sirve de piso de MAE/RMSE.
    flat = _metricas(np.zeros(len(test)), MODEL_NAME_FLAT)
    flat['accuracy_direccional'], flat['n_evaluado'] = None, 0
    resultados[MODEL_NAME_FLAT] = flat
    logging.info(
        f"[{MODEL_NAME_FLAT}] MAE retorno (log): {flat['mae_retorno']:.5f} | MAE: {flat['mae']:.2f} gp"
    )

    # --- baseline_momentum: predicted_target = retorno_anterior ---
    momentum = _metricas(retorno_anterior, MODEL_NAME_MOMENTUM)
    resultados[MODEL_NAME_MOMENTUM] = momentum
    acc_momentum = momentum['accuracy_direccional']
    logging.info(
        f"[{MODEL_NAME_MOMENTUM}] MAE retorno (log): {momentum['mae_retorno']:.5f} | "
        f"MAE: {momentum['mae']:.2f} gp | accuracy direccional: "
        f"{'N/A' if acc_momentum is None else f'{acc_momentum:.3f}'} "
        f"sobre {momentum['n_evaluado']} movimientos"
    )

    filas = [
        (None, train_ts, nombre, m['mae'], m['rmse'], 1, m['accuracy_direccional'], 'holdout',
         m['n_evaluado'], m['mae_retorno'], m['rmse_retorno'])
        for nombre, m in resultados.items()
    ]
    db.guardar_metricas_modelo(filas)
    logging.info(f"baseline.py: {len(filas)} fila(s) agregadas guardadas en model_metrics")

    return resultados


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s |--| %(levelname)s |--| %(message)s')
    db = OSRSBaseDatos('data/osrs_ge.db')
    evaluar_baselines(db)
