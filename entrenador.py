import logging
import os

import joblib
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from base_de_datos import OSRSBaseDatos
from preprocesamiento import build_training_set

MODEL_DIR = "models"
MODEL_VERSION = "global_v1"
N_ITEMS_LIQUIDOS = 200
TEST_FRACTION = 0.2

# Columnas del dataset que no son features (reconstrucción/target), el resto
# (lags, medias móviles, encoding de tiempo, item_id, buy_limit, members) se
# usa como entrada del modelo.
NON_FEATURE_COLS = ['timestamp_target', 'price_actual', 'price_target', 'target']


def entrenar_modelo_global(db, n_items=N_ITEMS_LIQUIDOS):
    """
    Entrena un único modelo XGBoost sobre los `n_items` ítems más líquidos,
    prediciendo el log-retorno del siguiente período (ver preprocesamiento.py
    para el porqué de trabajar en espacio de retorno y no precio crudo).

    Split temporal global (no 80/20 por ítem): todas las filas anteriores al
    corte van a train, las posteriores a test, sobre el dataset combinado —
    evita que información futura de un ítem se filtre, vía el modelo
    compartido, hacia el pasado de otro.
    """
    item_ids = db.obtener_top_items_liquidez(n_items)
    logging.info(f"Ítems líquidos seleccionados: {len(item_ids)}")
    if not item_ids:
        logging.error("resumen_actual está vacía — correr metricas.py antes de entrenar.")
        return

    dataset = build_training_set(db, item_ids)
    if dataset.empty:
        logging.error("No se pudo construir el dataset de entrenamiento (sin datos suficientes).")
        return
    logging.info(f"Dataset combinado: {len(dataset)} filas, {dataset['item_id'].nunique()} ítems con features")

    corte = dataset['timestamp_target'].quantile(1 - TEST_FRACTION)
    train = dataset[dataset['timestamp_target'] <= corte]
    test = dataset[dataset['timestamp_target'] > corte].copy()
    logging.info(f"Split temporal en timestamp {int(corte)}: train={len(train)} filas, test={len(test)} filas")

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
    mae_retorno = mean_absolute_error(y_test, y_pred)
    rmse_retorno = np.sqrt(mean_squared_error(y_test, y_pred))

    # Reconstrucción a gp: precio_predicho = precio_actual * exp(retorno_predicho)
    test['pred_price'] = test['price_actual'].to_numpy() * np.exp(y_pred)
    mae_gp = mean_absolute_error(test['price_target'], test['pred_price'])
    logging.info(
        f"MAE retorno (log): {mae_retorno:.5f} | RMSE retorno (log): {rmse_retorno:.5f} | "
        f"MAE reconstruido: {mae_gp:.2f} gp"
    )

    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, f"model_{MODEL_VERSION}.pkl")
    joblib.dump(model, model_path)
    logging.info(f"Modelo guardado en {model_path}")

    train_ts = int(dataset['timestamp_target'].max())

    # Una fila agregada (item_id=None) + una fila por ítem del test set, para
    # detectar ítems con error desproporcionado dentro del modelo global.
    metricas_filas = [(None, train_ts, MODEL_VERSION, float(mae_retorno), float(rmse_retorno))]
    for item_id, grupo in test.groupby('item_id', observed=True):
        mae_item = mean_absolute_error(grupo['price_target'], grupo['pred_price'])
        rmse_item = np.sqrt(mean_squared_error(grupo['price_target'], grupo['pred_price']))
        metricas_filas.append((int(item_id), train_ts, MODEL_VERSION, float(mae_item), float(rmse_item)))
    db.guardar_metricas_modelo(metricas_filas)
    logging.info(f"Métricas guardadas en model_metrics ({len(metricas_filas)} filas)")

    predicciones_filas = [
        (int(item_id), int(ts), float(pred), float(actual), float(actual - pred), MODEL_VERSION)
        for item_id, ts, pred, actual in zip(
            test['item_id'], test['timestamp_target'], test['pred_price'], test['price_target'],
        )
    ]
    db.guardar_predicciones(predicciones_filas)
    logging.info(f"Predicciones guardadas en predicciones ({len(predicciones_filas)} filas)")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s |--| %(levelname)s |--| %(message)s')
    db = OSRSBaseDatos('data/osrs_ge.db')
    entrenar_modelo_global(db)
