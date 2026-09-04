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
from baseline import MODEL_NAME_FLAT, MODEL_NAME_MOMENTUM
from metricas import filtrar_screener_liquido
from prediccion import cargar_modelo, pronosticar_item, pronosticar_clase_item

DB_PATH = 'data/osrs_ge.db'


def query(sql, params=()):
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(sql, conn, params=params)
    conn.close()
    return df


def _modelos_del_usuario(tipo=None):
    """
    model_id de los modelos que el usuario creó en modelos_config,
    opcionalmente filtrados por tipo ('regresor'|'clasificador').

    Antes las listas de modelos de este dashboard eran constantes fijas
    ('global_horario', 'global_diario', las variantes F2P del clasificador).
    Esos modelos dejaron de existir cuando modelos_config pasó a arrancar
    vacía: los selectores mostraban nombres de modelos que nadie tiene y
    todas las pestañas caían en "no hay modelo entrenado" sin importar
    cuántos modelos reales hubiera creado el usuario.
    """
    filtro = ' WHERE tipo = ?' if tipo else ''
    params = (tipo,) if tipo else ()
    try:
        df = query(f'SELECT model_id FROM modelos_config{filtro} ORDER BY creado_en', params)
    except Exception:
        return []
    return df['model_id'].tolist()


# Modelos con precio continuo en `predicciones` — únicos válidos para las
# tabs Predicción vs realidad y Pronóstico a futuro (asumen predicted_price/
# actual_price en gp, no aplican al clasificador, que predice una clase).
MODELOS_DISPONIBLES = _modelos_del_usuario('regresor')
# Los de arriba + el clasificador + los baselines de baseline.py: todos
# comparables en model_metrics (tab Calidad del modelo) porque comparten la
# métrica de accuracy direccional, aunque el clasificador y los baselines no
# tengan predicciones continuas ni pronóstico en vivo.
MODELOS_COMPARABLES = MODELOS_DISPONIBLES + _modelos_del_usuario('clasificador') + [
    MODEL_NAME_FLAT, MODEL_NAME_MOMENTUM,
]


@st.cache_resource
def _modelo_cacheado(model_name):
    """Cachea el bundle del modelo entre reruns de Streamlit — es el único
    estado que se lee una vez y no cambia hasta el próximo reentrenamiento;
    recargarlo del disco en cada interacción del usuario sería innecesario.
    Cacheado por model_name (Streamlit ya cachea por argumentos), así que
    horario y diario conviven sin pisarse."""
    return cargar_modelo(model_name=model_name)


st.set_page_config(page_title="OSRS GE — Monitoreo", layout="wide")
st.title("OSRS GE Predictor — Monitoreo")

tab_screener, tab_modelo, tab_predicciones, tab_pronostico, tab_clasificador = st.tabs(
    ["Screener", "Calidad del modelo", "Predicción vs realidad", "Pronóstico a futuro", "Señal direccional (F2P)"]
)

with tab_screener:
    st.caption(
        "Estado actual de resumen_actual (calculado por metricas.py). Filtrado por volumen "
        "mínimo: sin esto, ítems casi sin liquidez muestran roi_pct absurdos (ej. un ítem de "
        "1 gp con volumen_24h=10 mostrando 141600% de ROI) — ruido, no una oportunidad real."
    )
    resumen = query("SELECT * FROM resumen_actual")
    if resumen.empty:
        st.info("resumen_actual está vacía — correr metricas.py.")
    else:
        col1, col2 = st.columns([1, 3])
        with col1:
            volumen_min = st.slider("Volumen 24h mínimo", min_value=0, max_value=1000, value=100, step=10)
        with col2:
            orden_por = st.selectbox(
                "Ordenar por", ["margen_neto", "roi_pct", "volumen_24h", "profit_potencial_4h"]
            )
        resumen_filtrado = filtrar_screener_liquido(resumen, volumen_24h_minimo=volumen_min)
        st.caption(f"{len(resumen_filtrado)} de {len(resumen)} ítems pasan el filtro de liquidez.")
        st.dataframe(
            resumen_filtrado.sort_values(orden_por, ascending=False).reset_index(drop=True),
            use_container_width=True,
        )

with tab_modelo:
    st.caption(
        "MAE/RMSE/accuracy direccional agregado (item_id NULL) de cada corrida de "
        "entrenamiento en model_metrics. 'Modo de evaluación' separa el holdout 80/20 en "
        "vivo del walk-forward del replay histórico (replay_historico.py) — no son "
        "comparables en el mismo gráfico. horizonte=1 hora es la métrica de 1 paso "
        "(siempre disponible); horizontes mayores (evaluacion.py) solo se calculan una vez "
        "al día, en job_diario, y responden '¿hasta qué horizonte confío en una alerta?'. "
        "baseline_flat/baseline_momentum (baseline.py) son reglas triviales sin entrenar "
        "nada — el piso honesto contra el que hay que comparar antes de confiar en el modelo."
    )
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        model_sel_modelo = st.selectbox("Modelo", MODELOS_COMPARABLES, key="model_tab_modelo")
    with col_b:
        modo_sel_modelo = st.selectbox("Modo de evaluación", ["holdout", "walkforward"], key="modo_tab_modelo")
    with col_c:
        horizonte_sel = st.number_input("Horizonte (horas)", min_value=1, max_value=6, value=1, key="horizonte_tab_modelo")

    metricas_modelo = query(
        "SELECT train_timestamp, mae, rmse, accuracy_direccional FROM model_metrics "
        "WHERE item_id IS NULL AND model_name = ? AND modo_evaluacion = ? AND horizonte_horas = ? "
        "ORDER BY train_timestamp",
        params=(model_sel_modelo, modo_sel_modelo, int(horizonte_sel)),
    )
    if metricas_modelo.empty:
        st.info("Sin corridas registradas todavía para esta combinación de modelo/modo/horizonte.")
    else:
        metricas_modelo['fecha'] = pd.to_datetime(metricas_modelo['train_timestamp'], unit='s')
        st.line_chart(metricas_modelo.set_index('fecha')[['mae', 'rmse']])
        st.caption("Accuracy direccional: fracción de veces que el modelo acertó si el precio subía o bajaba (0.5 = azar).")
        st.line_chart(metricas_modelo.set_index('fecha')[['accuracy_direccional']])
        st.dataframe(metricas_modelo, use_container_width=True)

with tab_predicciones:
    st.caption(
        "Predicción vs precio real, por ítem y modelo, sobre el set de test holdout de la "
        "última corrida en vivo — no incluye las corridas walk-forward del replay histórico "
        "(replay_historico.py), que alimentan el backtest (backtest.py) en vez del dashboard."
    )
    model_sel_pred = st.selectbox("Modelo", MODELOS_DISPONIBLES, key="model_tab_predicciones")
    items_con_pred = query(
        "SELECT DISTINCT p.item_id, i.name FROM predicciones p "
        "JOIN items i ON i.item_id = p.item_id "
        "WHERE p.model_version = ? AND p.modo_evaluacion = 'holdout' ORDER BY i.name",
        params=(model_sel_pred,),
    )
    if items_con_pred.empty:
        st.info("Todavía no hay predicciones registradas para este modelo — correr entrenador.py.")
    else:
        nombre_a_id = dict(zip(items_con_pred['name'], items_con_pred['item_id']))
        nombre_elegido = st.selectbox("Ítem", list(nombre_a_id.keys()))
        item_id = nombre_a_id[nombre_elegido]

        pred_item = query(
            "SELECT timestamp, predicted_price, actual_price FROM predicciones "
            "WHERE item_id = ? AND model_version = ? AND modo_evaluacion = 'holdout' ORDER BY timestamp",
            params=(int(item_id), model_sel_pred),
        )
        pred_item['fecha'] = pd.to_datetime(pred_item['timestamp'], unit='s')
        st.line_chart(pred_item.set_index('fecha')[['predicted_price', 'actual_price']])

        mae_item = query(
            "SELECT mae, rmse, train_timestamp FROM model_metrics "
            "WHERE item_id = ? AND model_name = ? ORDER BY train_timestamp DESC LIMIT 1",
            params=(int(item_id), model_sel_pred),
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
    model_sel_pron = st.selectbox("Modelo", MODELOS_DISPONIBLES, key="model_tab_pronostico")
    items_liquidos = query(
        "SELECT DISTINCT p.item_id, i.name FROM predicciones p "
        "JOIN items i ON i.item_id = p.item_id "
        "WHERE p.model_version = ? AND p.modo_evaluacion = 'holdout' ORDER BY i.name",
        params=(model_sel_pron,),
    )
    if items_liquidos.empty:
        st.info("Todavía no hay ítems con este modelo entrenado — correr entrenador.py.")
    else:
        nombre_a_id_pron = dict(zip(items_liquidos['name'], items_liquidos['item_id']))
        col1, col2 = st.columns([3, 1])
        with col1:
            nombre_pron = st.selectbox("Ítem", list(nombre_a_id_pron.keys()), key="item_pronostico")
        with col2:
            n_pasos = st.number_input("Horas a futuro", min_value=1, max_value=24, value=6)
        item_id_pron = int(nombre_a_id_pron[nombre_pron])

        db = OSRSBaseDatos(DB_PATH)
        bundle = _modelo_cacheado(model_sel_pron)
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

with tab_clasificador:
    st.caption(
        "Señal del clasificador direccional (entrenador.entrenar_clasificador_direccional): "
        "para cada ítem del modelo elegido, la clase más probable del próximo período — "
        "baja/estable/sube — en vez de un precio continuo reconstruido. A diferencia de "
        "'Predicción vs realidad' y 'Pronóstico a futuro' (modelo continuo, horizonte "
        "recursivo a varios pasos), acá el horizonte es fijo a 1 paso: la inferencia se "
        "calcula en vivo acá mismo (no hay tabla de predicciones categóricas en la DB, a "
        "propósito, para no migrar el esquema) sobre el último dato conocido de cada ítem."
    )
    db_clasif = OSRSBaseDatos(DB_PATH)
    clasificadores = _modelos_del_usuario('clasificador')
    bundle_clasif = None
    if not clasificadores:
        st.info(
            "No hay ningún modelo clasificador en modelos_config — creá uno desde \"Mis "
            "modelos\" en la app de escritorio (cada modelo nuevo crea un par "
            "regresor+clasificador)."
        )
    else:
        model_sel_clasif = st.selectbox("Modelo", clasificadores, key="model_tab_clasificador")
        try:
            bundle_clasif = _modelo_cacheado(model_sel_clasif)
        except FileNotFoundError:
            st.info(
                f"'{model_sel_clasif}' todavía no tiene un .pkl entrenado — esperar al próximo "
                "job_horario o usar \"Entrenar ahora\" en la app."
            )
        except ValueError as e:  # features_version desactualizada, ver prediccion.py
            st.warning(str(e))

    if bundle_clasif is not None:
        cfg_clasif = db_clasif.obtener_modelo_config(model_sel_clasif)
        if cfg_clasif['modo_seleccion'] == 'manual':
            item_ids_f2p = cfg_clasif['item_ids'] or []
        else:
            item_ids_f2p = db_clasif.obtener_top_items_liquidez(
                cfg_clasif['n_items'], solo_f2p=cfg_clasif['solo_f2p'],
                precio_minimo=cfg_clasif['precio_minimo'],
                excluir_item_ids=cfg_clasif['excluir_item_ids'],
            )
        filas = []
        for iid in item_ids_f2p:
            resultado = pronosticar_clase_item(db_clasif, iid, bundle=bundle_clasif)
            if resultado is None:
                continue
            filas.append({
                'item_id': iid,
                'señal': resultado['label'],
                'prob_sube': resultado['probabilidades'].get('sube'),
                'prob_estable': resultado['probabilidades'].get('estable'),
                'prob_baja': resultado['probabilidades'].get('baja'),
                'precio_actual': resultado['precio_actual'],
            })

        if not filas:
            st.warning("Sin historia suficiente para generar señales todavía (ítems recién agregados o huecos de datos).")
        else:
            señales = pd.DataFrame(filas)
            resumen_f2p = query(
                "SELECT item_id, name, margen_neto, roi_pct, volumen_24h FROM resumen_actual "
                "WHERE item_id IN ({})".format(",".join("?" * len(item_ids_f2p))),
                params=tuple(int(i) for i in item_ids_f2p),
            )
            señales = señales.merge(resumen_f2p, on='item_id', how='left')
            columnas_orden = [
                'name', 'señal', 'prob_sube', 'prob_estable', 'prob_baja',
                'precio_actual', 'margen_neto', 'roi_pct', 'volumen_24h',
            ]
            st.caption(
                "Ordenado por prob_sube descendente — cruzado con margen_neto/roi_pct de "
                "resumen_actual (metricas.py) para que la señal sea accionable, no solo "
                "informativa: 'sube' con margen_neto negativo no es necesariamente una "
                "oportunidad."
            )
            st.dataframe(
                señales[[c for c in columnas_orden if c in señales.columns]]
                    .sort_values('prob_sube', ascending=False).reset_index(drop=True),
                use_container_width=True,
            )
