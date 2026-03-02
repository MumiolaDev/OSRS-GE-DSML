import time
import logging
import joblib
import os
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
from base_de_datos import OSRSBaseDatos
from preprocesamiento import preprocess_item

#MODEL_DIR = "models"
#os.makedirs(MODEL_DIR, exist_ok=True)
#logging.basicConfig(level=logging.INFO)

# def train_for_item(item_id, db, table='prices_5m',):
#     logging.info(f"Entrenando modelo para item {item_id}")
#     df = OSRSBaseDatos.obtener_precios_id(db, item_id, table)
    
#     if len(df) < 25:  # mínimo de datos para entrenar
#         logging.warning(f"Item {item_id}: pocos datos ({len(df)}), se omite")
#         return

#     df_feat = preprocess_item(df)
#     if df_feat.empty:
#         logging.warning(f"Item {item_id}: no se pudieron generar features")
#         return

#     X = df_feat.drop('target', axis=1)
#     y = df_feat['target']
#     # División temporal (80% train, 20% test)
#     split = int(len(X) * 0.8)
#     X_train, X_test = X.iloc[:split], X.iloc[split:]
#     y_train, y_test = y.iloc[:split], y.iloc[split:]

#     model = XGBRegressor(n_estimators=200, learning_rate=0.05, random_state=42, verbosity=0)
#     model.fit(X_train, y_train)

#     y_pred = model.predict(X_test)
#     mae = mean_absolute_error(y_test, y_pred)
#     rmse = np.sqrt(mean_squared_error(y_test, y_pred))
#     logging.info(f"Item {item_id} - MAE: {mae:.2f}, RMSE: {rmse:.2f}")

#     # Guardar modelo
#     joblib.dump(model, f"{MODEL_DIR}/model_{item_id}.pkl")

    # Guardar métricas
    #insert_metric(item_id, int(time.time()), 'XGBoost', mae, rmse)



#if __name__ == "__main__":
    