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

from base_de_datos import OSRSBaseDatos
from prediccion import cargar_modelo, pronosticar_item

DB_PATH = 'data/osrs_ge.db'


def query(sql, params=()):
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(sql, conn, params=params)
    conn.close()
    return df


@st.cache_resource
def _modelo_cacheado():
    """Cachea el bundle del modelo entre reruns de Streamlit — es el único
    estado que se lee una vez y no cambia hasta el próximo reentrenamiento;
    recargarlo del disco en cada interacción del usuario sería innecesario."""
    return cargar_modelo()


st.set_page_config(page_title="OSRS GE — Monitoreo", layout="wide")
st.title("OSRS GE Predictor — Monitoreo")

tab_screener, tab_modelo, tab_predicciones, tab_pronostico = st.tabs(
    ["Screener", "Calidad del modelo", "Predicción vs realidad", "Pronóstico a futuro"]
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

with tab_pronostico:
    st.caption(
        "Pronóstico hacia adelante (prediccion.py): a diferencia de la pestaña anterior, "
        "acá el precio real todavía no existe — es una extrapolación recursiva del modelo "
        "(se predice t+1, se usa como si fuera dato real para predecir t+2, y así sucesivamente). "
        "El volumen/spread de los pasos futuros se mantiene igual al último dato real conocido "
        "porque el modelo no predice volumen — por eso el error crece con el horizonte y el "
        "pronóstico tiende a verse más 'liso' que el precio real. Tratarlo como una señal de "
        "tendencia de corto plazo, no como un valor puntual exacto."
    )
    items_liquidos = query(
        "SELECT DISTINCT p.item_id, i.name FROM predicciones p "
        "JOIN items i ON i.item_id = p.item_id ORDER BY i.name"
    )
    if items_liquidos.empty:
        st.info("Todavía no hay ítems con modelo entrenado — correr entrenador.py.")
    else:
        nombre_a_id_pron = dict(zip(items_liquidos['name'], items_liquidos['item_id']))
        col1, col2 = st.columns([3, 1])
        with col1:
            nombre_pron = st.selectbox("Ítem", list(nombre_a_id_pron.keys()), key="item_pronostico")
        with col2:
            n_pasos = st.number_input("Horas a futuro", min_value=1, max_value=24, value=6)
        item_id_pron = int(nombre_a_id_pron[nombre_pron])

        db = OSRSBaseDatos(DB_PATH)
        bundle = _modelo_cacheado()
        pronostico = pronosticar_item(db, item_id_pron, bundle, n_pasos=int(n_pasos))

        if pronostico.empty:
            st.warning("No se pudo generar el pronóstico (historia insuficiente para este ítem).")
        else:
            hist = query(
                "SELECT timestamp, avg_low_price FROM precios_1h WHERE item_id = ? "
                "ORDER BY timestamp DESC LIMIT 96",
                params=(item_id_pron,),
            ).sort_values('timestamp')

            hist_serie = hist.set_index(pd.to_datetime(hist['timestamp'], unit='s'))['avg_low_price']
            hist_serie.name = 'precio_real'
            pron_serie = pronostico.set_index(pd.to_datetime(pronostico['timestamp'], unit='s'))['predicted_price']
            pron_serie.name = 'pronostico'
            # Conecta el pronóstico con el último punto real, para que la
            # línea no aparezca cortada entre pasado y futuro.
            if not hist_serie.empty:
                pron_serie = pd.concat([pd.Series([hist_serie.iloc[-1]], index=[hist_serie.index[-1]], name='pronostico'), pron_serie])

            combinado = pd.concat([hist_serie, pron_serie], axis=1)
            st.line_chart(combinado)
            st.dataframe(pronostico[['paso', 'timestamp', 'predicted_price']], use_container_width=True)
