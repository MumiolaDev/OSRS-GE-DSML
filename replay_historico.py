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


def ejecutar_replay(db, desde_ts, hasta_ts, on_progreso=None, debe_detener=None):
    """
    on_progreso (opcional): callback `f(fase, actual, total)`, llamado con
    fase='replay' en cada checkpoint procesado (corrido, saltado o
    fallido) — sin dependencia de ningún framework, misma idea que
    recolector.backfill_faltantes. Es lo que le permite a
    escritorio/hilo_recolector.py mostrar progreso real durante un replay
    largo en vez de dejar la UI sin ninguna señal por potencialmente
    minutos u horas.

    debe_detener (opcional): callback `f() -> bool`, chequeado antes de
    cada checkpoint — si devuelve True, corta el loop ahí mismo (un
    replay largo, de potencialmente miles de checkpoints, no debía tener
    forma de interrumpirse antes de esto: pedir "detener" desde la app de
    escritorio no tenía ningún efecto hasta que terminaba solo).

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
        if debe_detener is not None and debe_detener():
            logging.info(
                f"Replay histórico: interrumpido por pedido externo "
                f"({n_corridos + n_saltados + n_fallidos}/{len(checkpoints)})"
            )
            break

        model_name = MODEL_NAME_HORARIO if tipo == 'horario' else MODEL_NAME_DIARIO

        if db.existe_checkpoint(model_name, ts, 'walkforward'):
            n_saltados += 1
            # El `continue` saltea el try/except de abajo, pero on_progreso
            # y el log periódico tienen que dispararse igual acá -- si no,
            # un replay que reanuda uno ya casi terminado (la mayoría de
            # los checkpoints ya corridos) deja la barra de progreso
            # congelada porque casi todas las iteraciones pasan por acá.
            if on_progreso is not None:
                on_progreso('replay', n_corridos + n_saltados + n_fallidos, len(checkpoints))
            if (n_corridos + n_saltados + n_fallidos) % 50 == 0:
                logging.info(
                    f"Replay en curso: {n_corridos} corridos, {n_saltados} ya existentes, "
                    f"{n_fallidos} fallidos, de {len(checkpoints)} checkpoints totales"
                )
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

        if on_progreso is not None:
            on_progreso('replay', n_corridos + n_saltados + n_fallidos, len(checkpoints))

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


def ejecutar_replay_clasificador(
    db, desde_ts, hasta_ts, n_items=10, solo_f2p=True, model_name=None,
    umbral_pct=None, precio_minimo=None, excluir_item_ids=None,
    item_ids=None, tabla='precios_1h', ventana_dias=None, on_progreso=None, debe_detener=None,
):
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

    item_ids (opcional): ver entrenador.entrenar_modelo_global — lista
    explícita de ítems (modo manual), en vez de derivar el universo desde
    n_items/solo_f2p/precio_minimo/excluir_item_ids en cada checkpoint.
    tabla/ventana_dias: ver entrenador.entrenar_modelo_global — misma
    granularidad y ventana de historial configurables. Los tres son lo que
    le permite a esta función (antes atada a la liquidez sobre
    precios_1h con todo el historial) usarse para un grupo de ítems
    puntual con su propia configuración, ej. un modelo definido por el
    usuario desde la app de escritorio (ítems elegidos a mano y una
    ventana acotada, ver escritorio/paginas/pagina_modelos.py) o un
    ambiente de testeo ad hoc.

    on_progreso/debe_detener: mismo patrón que replay_historico.
    ejecutar_replay/ejecutar_replay_modelo — callback f(fase, actual,
    total) y f() -> bool, respectivamente.

    Devuelve un DataFrame con todas las filas de test acumuladas (columnas
    item_id, timestamp_target, price_actual, target, clase_predicha,
    clase_real, entre otras) — insumo directo de
    backtest.simular_clasificador_walkforward().
    """
    from entrenador import entrenar_clasificador_direccional, MODEL_NAME_CLASIF_F2P, MODEL_NAME_CLASIFICADOR
    import pandas as pd

    model_name = model_name or (MODEL_NAME_CLASIF_F2P if solo_f2p else MODEL_NAME_CLASIFICADOR)
    kwargs = dict(tabla=tabla, ventana_dias=ventana_dias)
    if item_ids is not None:
        kwargs['item_ids'] = item_ids
    else:
        kwargs['n_items'] = n_items
        kwargs['solo_f2p'] = solo_f2p
        if precio_minimo is not None:
            kwargs['precio_minimo'] = precio_minimo
        if excluir_item_ids is not None:
            kwargs['excluir_item_ids'] = excluir_item_ids
    if umbral_pct is not None:
        kwargs['umbral_pct'] = umbral_pct

    checkpoints = [ts for ts, tipo in calcular_checkpoints(desde_ts, hasta_ts) if tipo == 'horario']
    logging.info(f"Replay clasificador [{model_name}]: {len(checkpoints)} checkpoints horarios entre {desde_ts} y {hasta_ts}")

    resultados = []
    n_ok = n_vacios = n_fallidos = 0
    for ts in checkpoints:
        if debe_detener is not None and debe_detener():
            logging.info(f"Replay clasificador [{model_name}]: interrumpido por pedido externo ({n_ok + n_vacios + n_fallidos}/{len(checkpoints)})")
            break

        try:
            _, test = entrenar_clasificador_direccional(
                db, model_name=model_name, ahora_ts=ts, modo_evaluacion='walkforward', **kwargs,
            )
            if test is not None and not test.empty:
                resultados.append(test)
                n_ok += 1
            else:
                n_vacios += 1
        except Exception as e:
            logging.error(f"Replay clasificador: error en checkpoint {ts}: {e}")
            n_fallidos += 1

        if on_progreso is not None:
            on_progreso('walkforward', n_ok + n_vacios + n_fallidos, len(checkpoints))

        if (n_ok + n_vacios + n_fallidos) % 50 == 0:
            logging.info(
                f"Replay clasificador en curso: {n_ok} ok, {n_vacios} vacíos, {n_fallidos} fallidos, "
                f"de {len(checkpoints)} checkpoints totales"
            )

    logging.info(f"Replay clasificador completo: {n_ok} ok, {n_vacios} vacíos, {n_fallidos} fallidos")
    if not resultados:
        return pd.DataFrame()
    return pd.concat(resultados, ignore_index=True)


def _rango_disponible(db, tabla, cfg):
    """
    MIN/MAX timestamp disponible en `tabla` para los ítems de `cfg` — si
    modo_seleccion='manual', acotado a esos item_ids; si 'liquidez', el
    rango global de la tabla (no hay forma barata de saber qué ítems
    habrían entrado en el ranking de liquidez en cada momento histórico
    sin recorrerlo, así que se usa el rango completo como aproximación —
    entrenar_desde_config igual va a filtrar el universo real ítem por
    ítem en cada checkpoint vía obtener_top_items_liquidez_hasta).
    """
    import sqlite3

    conn = sqlite3.connect(db.db_path)
    c = conn.cursor()
    if cfg['modo_seleccion'] == 'manual' and cfg['item_ids']:
        placeholders = ','.join('?' * len(cfg['item_ids']))
        c.execute(f'SELECT MIN(timestamp), MAX(timestamp) FROM {tabla} WHERE item_id IN ({placeholders})', cfg['item_ids'])
    else:
        c.execute(f'SELECT MIN(timestamp), MAX(timestamp) FROM {tabla}')
    desde_ts, hasta_ts = c.fetchone()
    conn.close()
    return desde_ts, hasta_ts


PASO_SEGUNDOS_POR_TABLA = {'precios_5m': 300, 'precios_1h': 3600, 'precios_6h': 21600}


def ejecutar_replay_modelo(db, model_id, desde_ts=None, hasta_ts=None, on_progreso=None, debe_detener=None):
    """
    Walk-forward de CUALQUIER modelo de modelos_config — a diferencia de
    ejecutar_replay() (solo sirve para los 2 regresores productivos, con
    checkpoints alineados al scheduler fijo de job_horario/job_diario),
    esta función deriva todo (tabla, ventana, universo de ítems, tipo
    regresor/clasificador) de la fila de modelos_config, y los checkpoints
    se espacian al paso NATURAL de la tabla del modelo (cada 5m/1h/6h según
    corresponda), no al de ningún scheduler.

    desde_ts/hasta_ts (opcionales, default None = todo el historial
    disponible para los ítems del modelo): acotan el rango de checkpoints
    — sin esto, el walk-forward siempre recorre TODO lo que haya (el
    comportamiento pensado para "walk-forward inicial al crear un
    modelo", ver escritorio/hilo_walkforward.py). Pasar un rango explícito
    es lo que le permite a un ambiente de testeo acotar cada prueba a la
    ventana del backtest (ej. 90 días) en vez de recorrer meses de historial
    de más que después ni siquiera se usan
    para el backtest — sin esto, cada prueba de la búsqueda tardaría
    varias veces más de lo necesario.

    Pensada para correr una vez, justo después de crear un modelo desde la
    app de escritorio (ver escritorio/hilo_walkforward.py) — sobre lo que
    ya haya de historial disponible para sus ítems, sin esperar a que
    pasen semanas de recolección en vivo para tener una primera lectura de
    walk-forward real (no solo un split holdout).

    Reanudable igual que ejecutar_replay() (db.existe_checkpoint), y
    respeta on_progreso/debe_detener con la misma semántica que el resto
    del módulo — un walk-forward de varios meses a 1h son miles de
    checkpoints, cada uno un fit de XGBoost; correrlo en un hilo aparte e
    interrumpible no es opcional.

    NO actualiza modelos_config.ultimo_entrenamiento_ts en cada checkpoint
    (sería confuso mezclarlo con una corrida en vivo real, ver
    recolector._entrenar_desde_config) — el llamador puede hacerlo una vez
    al terminar si quiere reflejar que el modelo ya tiene una validación.

    Devuelve (n_corridos, n_saltados, n_fallidos), mismo formato que
    ejecutar_replay(); (0, 0, 0) si el modelo no existe o no hay
    historial disponible todavía para sus ítems/tabla.
    """
    from entrenador import entrenar_desde_config

    cfg = db.obtener_modelo_config(model_id)
    if cfg is None:
        logging.error(f"ejecutar_replay_modelo: no existe el modelo '{model_id}'")
        return 0, 0, 0

    tabla = cfg.get('tabla') or 'precios_1h'
    paso = PASO_SEGUNDOS_POR_TABLA.get(tabla, 3600)

    desde_disponible, hasta_disponible = _rango_disponible(db, tabla, cfg)
    if desde_disponible is None:
        logging.warning(f"[{model_id}] Sin historial disponible en {tabla} todavía, no se puede hacer walk-forward.")
        return 0, 0, 0

    # El último bucket puede no estar cerrado del lado de la API todavía
    # (mismo criterio que recolector.rellenar_huecos_al_inicio) — se
    # recorta un paso hacia atrás para no incluirlo como checkpoint.
    hasta_disponible -= paso

    # Si el llamador pidió un rango explícito, se acota al disponible (no
    # tiene sentido pedir checkpoints donde no hay datos para entrenar).
    desde_ts = max(desde_ts, desde_disponible) if desde_ts is not None else desde_disponible
    hasta_ts = min(hasta_ts, hasta_disponible) if hasta_ts is not None else hasta_disponible

    if hasta_ts < desde_ts:
        logging.warning(f"[{model_id}] Historial insuficiente en {tabla} todavía para ningún checkpoint.")
        return 0, 0, 0

    checkpoints = list(range(desde_ts, hasta_ts + 1, paso))
    logging.info(
        f"[{model_id}] Walk-forward inicial: {len(checkpoints)} checkpoints entre {desde_ts} y {hasta_ts} ({tabla})"
    )

    n_corridos = n_saltados = n_fallidos = 0
    for ts in checkpoints:
        if debe_detener is not None and debe_detener():
            logging.info(
                f"[{model_id}] Walk-forward inicial interrumpido por pedido externo "
                f"({n_corridos + n_saltados + n_fallidos}/{len(checkpoints)})"
            )
            break

        if db.existe_checkpoint(model_id, ts, 'walkforward'):
            n_saltados += 1
        else:
            try:
                resultado = entrenar_desde_config(
                    db, cfg, ahora_ts=ts, guardar_en_disco=False, modo_evaluacion='walkforward',
                )
                exito = (resultado[1] is not None) if cfg['tipo'] == 'clasificador' else bool(resultado)
                if exito:
                    n_corridos += 1
                else:
                    n_fallidos += 1
            except Exception as e:
                logging.error(f"[{model_id}] Walk-forward inicial: error en checkpoint ts={ts}: {e}")
                n_fallidos += 1

        if on_progreso is not None:
            on_progreso('walkforward', n_corridos + n_saltados + n_fallidos, len(checkpoints))

        if (n_corridos + n_saltados + n_fallidos) % 50 == 0:
            logging.info(
                f"[{model_id}] Walk-forward inicial en curso: {n_corridos} corridos, {n_saltados} ya existentes, "
                f"{n_fallidos} fallidos, de {len(checkpoints)} checkpoints totales"
            )

    logging.info(
        f"[{model_id}] Walk-forward inicial completo: {n_corridos} corridos, {n_saltados} ya existentes, "
        f"{n_fallidos} fallidos, de {len(checkpoints)} checkpoints totales"
    )
    return n_corridos, n_saltados, n_fallidos
