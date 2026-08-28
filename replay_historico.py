"""
replay_historico.py — reentrena en los momentos históricos exactos en los
que job_horario/job_diario habrían corrido, para un rango del pasado que
recién se rellenó (recolector.rellenar_huecos_al_inicio).

Por qué existe: sin esto, un hueco de datos rellenado de golpe (ej. el
recolector estuvo apagado varias semanas) solo se reentrena UNA vez al
final, con todo el historial disponible — se pierde por completo lo que
habría pasado en vivo: 200 reentrenamientos horarios uno detrás de otro,
cada uno viendo solo los datos disponibles hasta ese momento. Este módulo
"repite" esa secuencia real (walk-forward), lo que además deja un historial
real de model_metrics/predicciones (modo_evaluacion='walkforward') que
sirve de insumo directo para backtest.py, en vez de tener que simularlo por
separado.

Es intencionalmente caro (puede ser cientos o miles de reentrenamientos
para un hueco grande) — se prioriza fidelidad sobre velocidad, decisión
explícita frente a la alternativa de solo hacer catch-up con el modelo más
reciente. Es reanudable: cada checkpoint se salta si ya corrió
(OSRSBaseDatos.existe_checkpoint), así que interrumpir el proceso (ej.
reiniciar el recolector) y volver a llamar ejecutar_replay() sobre el mismo
rango no duplica trabajo ni filas.
"""

import logging

from entrenador import entrenar_modelo_global, MODEL_NAME_HORARIO, MODEL_NAME_DIARIO

MINUTO_OFFSET_HORARIO = 5  # job_horario corre a :05, ver recolector.py
HORA_DIARIA = 3            # job_diario corre a las 03:00 UTC, ver recolector.py


def calcular_checkpoints(desde_ts, hasta_ts, minuto_offset_horario=MINUTO_OFFSET_HORARIO, hora_diaria=HORA_DIARIA):
    """
    Devuelve, ordenados cronológicamente, los timestamps en los que
    job_horario (cada hora, minuto `minuto_offset_horario`) y job_diario
    (una vez al día, a las `hora_diaria`:00 UTC) habrían corrido entre
    `desde_ts` y `hasta_ts` — replica exactamente la cadencia real del
    scheduler (recolector.py __main__), para que el replay sea fiel a qué
    checkpoints habrían existido si el proceso hubiese estado vivo.

    Los timestamps Unix ya son UTC por definición, así que no hace falta
    ninguna conversión de zona horaria acá (a diferencia del scheduler en
    vivo, que sí necesita decirle a `schedule` que interprete ":05"/"03:00"
    como hora UTC).

    Devuelve una lista de tuplas (timestamp, 'horario'|'diario').
    """
    checkpoints = []

    paso_horario = 3600
    inicio_horario = (desde_ts // paso_horario) * paso_horario + minuto_offset_horario * 60
    if inicio_horario < desde_ts:
        inicio_horario += paso_horario
    ts = inicio_horario
    while ts <= hasta_ts:
        checkpoints.append((ts, 'horario'))
        ts += paso_horario

    paso_diario = 86400
    inicio_diario = (desde_ts // paso_diario) * paso_diario + hora_diaria * 3600
    if inicio_diario < desde_ts:
        inicio_diario += paso_diario
    ts = inicio_diario
    while ts <= hasta_ts:
        checkpoints.append((ts, 'diario'))
        ts += paso_diario

    checkpoints.sort(key=lambda par: par[0])
    return checkpoints


def ejecutar_replay(db, desde_ts, hasta_ts):
    """
    Recorre calcular_checkpoints(desde_ts, hasta_ts) en orden cronológico y,
    para cada uno que no haya corrido todavía (db.existe_checkpoint), llama
    entrenar_modelo_global() con modo_evaluacion='walkforward' y
    guardar_en_disco=False — el .pkl productivo no debe pisarse en cada uno
    de potencialmente miles de checkpoints históricos; se actualiza aparte,
    al final, con una corrida en vivo real (ver recolector.job_horario/
    job_diario, llamados después de esto por quien orqueste el replay).

    No toca resumen_actual (entrenar_modelo_global en modo replay usa
    obtener_top_items_liquidez_hasta, no la tabla resumen_actual — esa es el
    estado EN VIVO que lee el dashboard).

    Loguea y sigue ante el fallo de un checkpoint individual: con
    potencialmente miles de checkpoints, uno malo (ej. un hueco de datos
    puntual) no debe abortar todo el replay.

    Devuelve (n_corridos, n_saltados, n_fallidos).
    """
    checkpoints = calcular_checkpoints(desde_ts, hasta_ts)
    logging.info(f"Replay histórico: {len(checkpoints)} checkpoints entre {desde_ts} y {hasta_ts}")

    n_corridos = n_saltados = n_fallidos = 0
    for ts, tipo in checkpoints:
        model_name = MODEL_NAME_HORARIO if tipo == 'horario' else MODEL_NAME_DIARIO

        if db.existe_checkpoint(model_name, ts, 'walkforward'):
            n_saltados += 1
            continue

        try:
            entrenar_modelo_global(
                db, model_name=model_name, ahora_ts=ts,
                guardar_en_disco=False, modo_evaluacion='walkforward',
            )
            n_corridos += 1
        except Exception as e:
            logging.error(f"Replay: error en checkpoint {tipo} ts={ts}: {e}")
            n_fallidos += 1

        if (n_corridos + n_saltados + n_fallidos) % 50 == 0:
            logging.info(
                f"Replay en curso: {n_corridos} corridos, {n_saltados} ya existentes, "
                f"{n_fallidos} fallidos, de {len(checkpoints)} checkpoints totales"
            )

    logging.info(
        f"Replay histórico completo ({desde_ts}-{hasta_ts}): {n_corridos} corridos, "
        f"{n_saltados} ya existentes, {n_fallidos} fallidos"
    )
    return n_corridos, n_saltados, n_fallidos


def ejecutar_replay_clasificador(db, desde_ts, hasta_ts, n_items=10, solo_f2p=True, model_name=None, umbral_pct=None, precio_minimo=None, excluir_item_ids=None):
    """
    Como ejecutar_replay(), pero para el clasificador direccional
    (entrenador.entrenar_clasificador_direccional) en vez del regresor —
    todavía no hay una cadencia 'diaria' del clasificador, así que solo usa
    los checkpoints 'horario'. A diferencia de ejecutar_replay (que guarda
    una métrica agregada por checkpoint y sigue), acá se junta el DataFrame
    de test COMPLETO de cada checkpoint en uno solo — eso es lo que hace
    esto un walk-forward de verdad: cada fila es una predicción hecha por
    un modelo que en ese momento nunca vio ese período ni nada posterior,
    a diferencia de un único split holdout (una sola "foto" del tiempo,
    con solo un modelo evaluado una vez).

    precio_minimo: ver entrenador.entrenar_clasificador_direccional — se
    aplica en cada checkpoint vía obtener_top_items_liquidez_hasta, así el
    universo de ítems excluye los mismos baratos en todo el rango.

    excluir_item_ids: ver entrenador.entrenar_clasificador_direccional —
    idem, aplicado en cada checkpoint.

    Devuelve un DataFrame con todas las filas de test acumuladas (columnas
    item_id, timestamp_target, price_actual, target, clase_predicha,
    clase_real, entre otras) — insumo directo de
    backtest.simular_clasificador_walkforward().
    """
    from entrenador import entrenar_clasificador_direccional, MODEL_NAME_CLASIF_F2P, MODEL_NAME_CLASIFICADOR
    import pandas as pd

    model_name = model_name or (MODEL_NAME_CLASIF_F2P if solo_f2p else MODEL_NAME_CLASIFICADOR)
    kwargs = {}
    if umbral_pct is not None:
        kwargs['umbral_pct'] = umbral_pct
    if precio_minimo is not None:
        kwargs['precio_minimo'] = precio_minimo
    if excluir_item_ids is not None:
        kwargs['excluir_item_ids'] = excluir_item_ids

    checkpoints = [ts for ts, tipo in calcular_checkpoints(desde_ts, hasta_ts) if tipo == 'horario']
    logging.info(f"Replay clasificador [{model_name}]: {len(checkpoints)} checkpoints horarios entre {desde_ts} y {hasta_ts}")

    resultados = []
    n_ok = n_vacios = n_fallidos = 0
    for ts in checkpoints:
        try:
            _, test = entrenar_clasificador_direccional(
                db, n_items=n_items, solo_f2p=solo_f2p, model_name=model_name,
                ahora_ts=ts, modo_evaluacion='walkforward', **kwargs,
            )
            if test is not None and not test.empty:
                resultados.append(test)
                n_ok += 1
            else:
                n_vacios += 1
        except Exception as e:
            logging.error(f"Replay clasificador: error en checkpoint {ts}: {e}")
            n_fallidos += 1

        if (n_ok + n_vacios + n_fallidos) % 50 == 0:
            logging.info(
                f"Replay clasificador en curso: {n_ok} ok, {n_vacios} vacíos, {n_fallidos} fallidos, "
                f"de {len(checkpoints)} checkpoints totales"
            )

    logging.info(f"Replay clasificador completo: {n_ok} ok, {n_vacios} vacíos, {n_fallidos} fallidos")
    if not resultados:
        return pd.DataFrame()
    return pd.concat(resultados, ignore_index=True)
