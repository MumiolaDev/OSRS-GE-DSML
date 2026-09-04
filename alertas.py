"""
alertas.py — notifica por Telegram cuando aparecen buenas oportunidades de
flip en resumen_actual.

Requiere que el usuario haya creado su propio bot de Telegram (@BotFather,
/newbot) y haya obtenido su chat_id (ver docs/despliegue_24_7.md para los
pasos completos), y que TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID estén seteadas
como variables de entorno del proceso — nunca hardcodeadas ni commiteadas.

Se engancha al final de recolector.job_horario: reusa el resumen_actual y
el modelo que ese job acaba de refrescar/reentrenar, sin necesitar un
scheduler propio.
"""

import logging
import sqlite3
import time

import pandas as pd
import requests

import configuracion
from metricas import filtrar_screener_liquido, margen_neto_proyectado
from prediccion import cargar_modelo, pronosticar_item

TELEGRAM_API_URL = "https://api.telegram.org/bot{token}/sendMessage"

# Un dato de precio más viejo que esto no describe el mercado "de ahora":
# alertar con él es mandar una oportunidad que puede haber desaparecido hace
# rato. Con recolección horaria sana, `ultimo_timestamp` nunca debería
# quedar a más de un par de horas (ver recolector._ultimo_bucket_cerrado).
ANTIGUEDAD_MAXIMA_HORAS = 6


def _modelo_regresor_activo(db):
    """
    model_id del regresor activo más reciente de modelos_config, o None si el
    usuario todavía no creó ninguno.

    Antes esto era la constante MODEL_NAME_HORARIO ('global_horario'), que
    dejó de existir cuando modelos_config pasó a arrancar vacía: el resultado
    era un warning "no se pudo cargar el modelo 'global_horario'" en CADA
    corrida de job_horario, y las alertas nunca llevaban señal del modelo por
    más modelos que el usuario hubiera creado.
    """
    regresores = [
        m for m in db.listar_modelos_config(estado='activo') if m['tipo'] == 'regresor'
    ]
    if not regresores:
        return None
    return max(regresores, key=lambda m: m['creado_en'] or 0)['model_id']


def enviar_mensaje_telegram(texto, token=None, chat_id=None):
    """
    POST a la Bot API de Telegram (requests, ya es dependencia del proyecto
    — no hace falta un SDK nuevo para un envío unidireccional simple).
    token/chat_id, si no se pasan, salen de configuracion.obtener() —
    variable de entorno TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID si está seteada
    (despliegue en modo servidor/NSSM, ver docs/despliegue_24_7.md) o si no
    config.json (editado desde la app de escritorio, ver
    escritorio/paginas/pagina_configuracion.py). Si faltan las
    credenciales en cualquiera de los dos lados, loguea un warning y no
    rompe el job que la llama: las alertas son un extra, no deben tumbar
    job_horario si todavía no están configuradas.
    """
    token = token or configuracion.obtener('telegram_bot_token')
    chat_id = chat_id or configuracion.obtener('telegram_chat_id')
    if not token or not chat_id:
        logging.warning(
            "alertas.py: faltan las credenciales de Telegram (TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID "
            "como variables de entorno, o token/chat_id en config.json) — se omite el envío. "
            "Ver docs/despliegue_24_7.md o la pestaña Configuración de la app."
        )
        return False

    try:
        resp = requests.post(
            TELEGRAM_API_URL.format(token=token),
            json={'chat_id': chat_id, 'text': texto, 'parse_mode': 'HTML', 'disable_web_page_preview': True},
            timeout=10,
        )
        resp.raise_for_status()
        return True
    except Exception as e:
        logging.error(f"alertas.py: error enviando mensaje a Telegram: {e}")
        return False


def _formatear_mensaje(candidatos):
    """candidatos: lista de dicts con name, margen_neto, roi_pct, buy_limit,
    margen_proyectado_modelo (puede ser None), horizonte_horas."""
    lineas = [f"🔔 <b>{len(candidatos)} oportunidad(es) de flip</b>"]
    for c in candidatos:
        señal = ""
        if c.get('margen_proyectado_modelo') is not None:
            señal = f" | modelo ({c['horizonte_horas']}h): {c['margen_proyectado_modelo']:.0f} gp netos proyectados"
        lineas.append(
            f"• <b>{c['name']}</b> — margen neto {c['margen_neto']:.0f} gp, "
            f"ROI {c['roi_pct']:.1f}%, buy limit {c['buy_limit']}{señal}"
        )
    return "\n".join(lineas)


def evaluar_alertas(
    db,
    umbral_roi_pct=5.0,
    volumen_24h_minimo=100,
    cooldown_horas=6,
    cambio_significativo_pct=20.0,
    top_n=10,
    model_name=None,
    horizonte_horas=3,
):
    """
    Filtra resumen_actual con filtrar_screener_liquido() + roi_pct >=
    umbral_roi_pct, ordena por margen_neto y toma el top_n. Si hay un
    modelo disponible, cruza cada candidato con margen_neto_proyectado()
    (metricas.py) para anotar la señal del modelo en el mensaje — no filtra
    por ella, solo informa; el filtro principal sigue siendo el screener.
    Usar el modelo como filtro duro queda como mejora futura, una vez
    calibrada su accuracy_direccional por horizonte (ver evaluacion.py) —
    hoy no hay garantía de que sea mejor que el azar a todos los horizontes.

    Por ítem, alerta solo si nunca se avisó, si pasó cooldown_horas desde la
    última alerta, o si roi_pct/margen_neto cambió más de
    cambio_significativo_pct desde entonces (evita spam en cada corrida de
    job_horario sin perderse cambios grandes dentro del cooldown). Agrupa
    todos los candidatos que pasan el filtro en UN solo mensaje Telegram,
    no uno por ítem.
    """
    model_name = model_name or _modelo_regresor_activo(db)

    conn = db.conectar()
    resumen = pd.read_sql_query(
        'SELECT r.*, i.buy_limit FROM resumen_actual r JOIN items i ON i.item_id = r.item_id', conn,
    )
    conn.close()
    if resumen.empty:
        logging.info("alertas.py: resumen_actual vacía, nada que evaluar.")
        return

    candidatos_df = filtrar_screener_liquido(
        resumen, volumen_24h_minimo=volumen_24h_minimo,
        antiguedad_maxima_horas=ANTIGUEDAD_MAXIMA_HORAS,
    )
    candidatos_df = candidatos_df[candidatos_df['roi_pct'] >= umbral_roi_pct]
    candidatos_df = candidatos_df.sort_values('margen_neto', ascending=False).head(top_n)

    if candidatos_df.empty:
        logging.info("alertas.py: sin candidatos que pasen los umbrales configurados.")
        return

    bundle = None
    if model_name is None:
        logging.info(
            "alertas.py: no hay ningún modelo regresor activo en modelos_config — se alerta solo "
            "con el screener, sin señal de modelo."
        )
    else:
        try:
            bundle = cargar_modelo(model_name=model_name)
        except FileNotFoundError:
            logging.info(
                f"alertas.py: '{model_name}' todavía no tiene un .pkl entrenado, se omite la señal del modelo."
            )
        except Exception as e:
            logging.warning(f"alertas.py: no se pudo cargar el modelo '{model_name}', se omite la señal del modelo: {e}")

    ahora = int(time.time())
    candidatos_a_enviar = []
    for _, fila in candidatos_df.iterrows():
        item_id = int(fila['item_id'])

        ultima = db.obtener_ultima_alerta(item_id)
        if ultima is not None:
            ts_ultima, roi_anterior, margen_anterior = ultima
            pasado_cooldown = (ahora - ts_ultima) >= cooldown_horas * 3600
            cambio_roi = abs(fila['roi_pct'] - (roi_anterior or 0)) >= cambio_significativo_pct
            cambio_margen = bool(
                margen_anterior and abs(fila['margen_neto'] - margen_anterior) / abs(margen_anterior) * 100 >= cambio_significativo_pct
            )
            if not pasado_cooldown and not cambio_roi and not cambio_margen:
                continue

        margen_proyectado_modelo = None
        if bundle is not None:
            try:
                # Sin `tabla=`: la granularidad sale del bundle (con cuál se
                # entrenó ese modelo), forzar 'precios_1h' acá le daba
                # features de otra escala temporal a un modelo de 5m/6h.
                pronostico = pronosticar_item(db, item_id, bundle, n_pasos=horizonte_horas)
                if not pronostico.empty:
                    pred_price = pronostico.iloc[-1]['predicted_price']
                    margen_proyectado_modelo = margen_neto_proyectado(
                        fila['avg_low_price'], fila['avg_high_price'], pred_price, item_id,
                    )
            except Exception as e:
                logging.warning(f"alertas.py: error pronosticando ítem {item_id}, se omite la señal del modelo: {e}")

        candidatos_a_enviar.append({
            'item_id': item_id,
            'name': fila['name'],
            'margen_neto': fila['margen_neto'],
            'roi_pct': fila['roi_pct'],
            'buy_limit': int(fila['buy_limit']) if pd.notna(fila['buy_limit']) else None,
            'margen_proyectado_modelo': margen_proyectado_modelo,
            'horizonte_horas': horizonte_horas,
        })

    if not candidatos_a_enviar:
        logging.info("alertas.py: candidatos filtrados, pero todos en cooldown sin cambios significativos.")
        return

    texto = _formatear_mensaje(candidatos_a_enviar)
    enviado = enviar_mensaje_telegram(texto)
    if enviado:
        for c in candidatos_a_enviar:
            db.guardar_alerta(c['item_id'], ahora, c['roi_pct'], c['margen_neto'])
        logging.info(f"alertas.py: {len(candidatos_a_enviar)} oportunidad(es) enviadas por Telegram.")


if __name__ == '__main__':
    from base_de_datos import OSRSBaseDatos

    logging.basicConfig(level=logging.INFO, format='%(asctime)s |--| %(levelname)s |--| %(message)s')
    db = OSRSBaseDatos('data/osrs_ge.db')
    evaluar_alertas(db)
