"""
dashboard.py — panel de monitoreo, solo lectura sobre data/osrs_ge.db.

No escribe nunca en la DB (WAL permite lectores concurrentes con el
recolector escribiendo). No reemplaza la futura app web del README — es un
panel interno para verificar que el pipeline 24/7 (recolección,
reentrenamiento, retención) está funcionando solo, sin tener que abrir la
DB a mano.

Se corre con: streamlit run dashboard.py
Refresh manual (recargar la página) — con datos horarios no se justifica
todavía un auto-refresh en vivo.
"""

import sqlite3

import pandas as pd
import streamlit as st

DB_PATH = 'data/osrs_ge.db'


def query(sql, params=()):
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(sql, conn, params=params)
    conn.close()
    return df


st.set_page_config(page_title="OSRS GE — Monitoreo", layout="wide")
st.title("OSRS GE Predictor — Monitoreo")

tab_screener, tab_modelo, tab_predicciones = st.tabs(
    ["Screener", "Calidad del modelo", "Predicción vs realidad"]
)

with tab_screener:
    st.caption("Estado actual de resumen_actual (calculado por metricas.py).")
    resumen = query("SELECT * FROM resumen_actual")
    if resumen.empty:
        st.info("resumen_actual está vacía — correr metricas.py.")
    else:
        orden_por = st.selectbox(
            "Ordenar por", ["margen_neto", "roi_pct", "volumen_24h", "profit_potencial_4h"]
        )
        st.dataframe(
            resumen.sort_values(orden_por, ascending=False).reset_index(drop=True),
            use_container_width=True,
        )

with tab_modelo:
    st.caption("MAE/RMSE agregado (item_id NULL) de cada corrida de entrenamiento en model_metrics.")
    metricas_modelo = query(
        "SELECT train_timestamp, model_name, mae, rmse FROM model_metrics "
        "WHERE item_id IS NULL ORDER BY train_timestamp"
    )
    if metricas_modelo.empty:
        st.info("Todavía no hay corridas de entrenamiento registradas — correr entrenador.py.")
    else:
        metricas_modelo['fecha'] = pd.to_datetime(metricas_modelo['train_timestamp'], unit='s')
        st.line_chart(metricas_modelo.set_index('fecha')[['mae', 'rmse']])
        st.dataframe(metricas_modelo, use_container_width=True)

with tab_predicciones:
    st.caption("Predicción vs precio real, por ítem, sobre el set de test de la última corrida.")
    items_con_pred = query(
        "SELECT DISTINCT p.item_id, i.name FROM predicciones p "
        "JOIN items i ON i.item_id = p.item_id ORDER BY i.name"
    )
    if items_con_pred.empty:
        st.info("Todavía no hay predicciones registradas — correr entrenador.py.")
    else:
        nombre_a_id = dict(zip(items_con_pred['name'], items_con_pred['item_id']))
        nombre_elegido = st.selectbox("Ítem", list(nombre_a_id.keys()))
        item_id = nombre_a_id[nombre_elegido]

        pred_item = query(
            "SELECT timestamp, predicted_price, actual_price FROM predicciones "
            "WHERE item_id = ? ORDER BY timestamp",
            params=(int(item_id),),
        )
        pred_item['fecha'] = pd.to_datetime(pred_item['timestamp'], unit='s')
        st.line_chart(pred_item.set_index('fecha')[['predicted_price', 'actual_price']])

        mae_item = query(
            "SELECT mae, rmse, train_timestamp FROM model_metrics "
            "WHERE item_id = ? ORDER BY train_timestamp DESC LIMIT 1",
            params=(int(item_id),),
        )
        if not mae_item.empty:
            st.metric("MAE (gp, último entrenamiento)", f"{mae_item['mae'].iloc[0]:.2f}")
