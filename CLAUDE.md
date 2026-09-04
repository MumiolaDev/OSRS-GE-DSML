# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Qué es este proyecto

OSRS GE Predictor de Precios: pipeline que recolecta precios de compra/venta del Grand Exchange
de Old School RuneScape desde la API pública `prices.runescape.wiki`, los almacena en SQLite,
calcula un screener de oportunidades de flip y entrena modelos XGBoost (regresor de precio +
clasificador direccional) que el usuario define él mismo, con los ítems y la ventana de
historial que elija — no hay ningún modelo sembrado por default (pedido explícito del usuario:
"no quiero que haya ningún modelo por default, todos tienen que poder eliminarse"), `modelos_config`
arranca vacía y el pipeline funciona igual de bien sin ningún modelo creado todavía (el screener
no depende de ninguno). Corre pensado para 24/7 (ver `docs/despliegue_24_7.md`), manda alertas de
oportunidades por Telegram (`alertas.py`), expone un dashboard de monitoreo en Streamlit
(`dashboard.py`, solo lectura, para el desarrollador) y una app de escritorio en PySide6/Qt
(`escritorio/`, ver su propio bullet más abajo) para el usuario final.

## Comandos

```bash
pip install -r requirements.txt   # instalar dependencias
python recolector.py              # recolector 24/7: puebla catálogo, recolecta 5m/1h/6h alineado
                                   # al reloj, rellena huecos + replay histórico al arrancar, y
                                   # dispara job_horario/job_diario (screener+reentreno+alertas) y
                                   # job_semanal (retención, con catch-up si se perdió la ventana)
python metricas.py                # recalcula resumen_actual (screener) a partir de precios_1h
python entrenador.py              # entrena el modelo global_horario sobre los ítems más líquidos
python baseline.py                # reglas triviales de referencia (flat/momentum), para comparar
python replay_historico.py        # (como módulo) reentrena walk-forward sobre un rango del pasado
python backtest.py                # simula la estrategia de flip sobre un rango ya replayeado
python alertas.py                 # evalúa resumen_actual y manda alertas por Telegram (una vez)
python mantenimiento.py           # retención/purga de todas las tablas de series de tiempo + vacuum
streamlit run dashboard.py        # dashboard de monitoreo, solo lectura
python -m pytest tests/           # tests unitarios de las funciones puras (sin DB, sin red)
```
No hay linter configurado todavía. Los tests (`tests/`) cubren funciones puras (impuesto/
margen/dimensionamiento/ventanas por tiempo de `metricas.py`, `_agrupar_en_rangos` y
`_ultimo_bucket_cerrado` de `recolector.py`, `calcular_checkpoints` de `replay_historico.py`,
`_clasificar_retorno` y `accuracy_direccional` de `entrenador.py`, la grilla regular y las
features relativas de `preprocesamiento.py`) más los que usan una DB SQLite temporal
(`modelos_config`, helpers de `pagina_modelos.py`) — nada que toque la API real.

## Arquitectura del pipeline

```
osrs_ge_api.py → recolector.py → base_de_datos.py (SQLite)
                     │  (schedule alineado al reloj: 5m/1h/6h + job_horario + job_diario + job_semanal)
                     ├─→ metricas.py ────────────→ resumen_actual (screener)
                     ├─→ preprocesamiento.py → entrenador.py → model_metrics / predicciones
                     │         │                    (regresor + clasificador, uno por cada modelo
                     │         │                     que el usuario haya creado en modelos_config —
                     │         │                     ninguno por default, ver escritorio/)
                     │         └─→ evaluacion.py (accuracy direccional + MAE por horizonte)
                     ├─→ replay_historico.py (walk-forward al rellenar huecos del pasado o al crear un modelo)
                     ├─→ alertas.py (Telegram, al final de job_horario)
                     └─→ mantenimiento.py (retención/downsampling/vacuum de todas las tablas)

prediccion.py (pronóstico recursivo) y backtest.py (simulación histórica) reusan el modelo
guardado y las predicciones walk-forward — no están en el loop del recolector.

baseline.py (reglas triviales) es un experimento de comparación fuera del loop del recolector.
entrenador.entrenar_clasificador_direccional() se reentrena desde job_horario/job_diario para
CUALQUIER modelo de tipo='clasificador' que el usuario haya creado con esa cadencia — sin
distinción de código entre regresor y clasificador ni entre "modelos del sistema" y "modelos del
usuario" (no existen los primeros) — ver el bullet de entrenador.py.

dashboard.py (Streamlit) lee resumen_actual / model_metrics / predicciones — solo lectura.
escritorio/ (PySide6/Qt) es la app de escritorio para el usuario final — lee y ESCRIBE
modelos_config (alta/pausa/borrado de modelos, botón "Entrenar ahora"), además de leer el resto.
```

- **`osrs_ge_api.py`**: cliente de la API. Los endpoints `/5m`, `/1h`, `/6h` devuelven un
  snapshot de **todos** los ítems del juego en cada llamada (no aceptan filtrar por item_id);
  el filtrado a ítems específicos, si se necesita, se hace después sobre el DataFrame. Todas
  las requests salen por una `requests.Session` compartida con `timeout=30` y reintentos con
  backoff (`REINTENTOS`) — **nunca** sacar el timeout: sin él, un socket colgado congela el
  scheduler entero de un proceso 24/7 sin log ni excepción.
  Tres gotchas verificados empíricamente (no documentados por la API): (1) el `timestamp` debe
  ser un múltiplo exacto del step del intervalo (300/3600/21600) o devuelve `400 Bad Request`
  — cualquier backfill manual tiene que alinear `desde`/`hasta` antes de generar el rango, no
  solo pasar `int(time.time())`; (2) la API no tiene datos reales antes de **2021-03-08**
  (~08:00 UTC) — timestamps anteriores devuelven `200 OK` con `data` vacío, no un error, así
  que "sin filas" ahí es esperable y no un bug; (3) **pedir el endpoint SIN `timestamp` no
  devuelve el último bucket cerrado sino el anterior a ese** (a las 13:39 UTC, `/1h` sin
  timestamp devuelve las 11:00, mientras que `/1h?timestamp=12:00` devuelve 3.055 ítems sin
  problema; el bucket en curso sí vuelve vacío). Por eso la recolección programada pasa
  siempre un timestamp explícito — ver `recolector._ultimo_bucket_cerrado`.
  `_procesar_data_historicals` descarta el ítem si le falta cualquiera de las dos puntas de
  precio en ese bucket, lo que deja huecos reales en la serie de un ítem poco líquido (medido:
  28% de las horas en Elysian spirit shield, 96% en un 3rd age) — de ahí el reindexado a
  grilla regular de `preprocesamiento.py`.
- **`recolector.py`**: loop de `schedule` pensado para correr indefinidamente, con todos los
  horarios alineados al reloj de pared en UTC (`.at(..., "UTC")`), no relativos a cuándo arrancó
  el proceso — ver `_programar_cada_n_minutos_alineado` (acepta `offset_minutos`, hoy 1: se pide
  un minuto DESPUÉS del cierre del bucket, no en el instante exacto). `ITEM_IDS` vacío = sin
  filtro (uso normal). Lo que registra el scheduler es **`collect_programado(db, interval)`**,
  que pide explícitamente los últimos buckets cerrados que falten
  (`_ultimo_bucket_cerrado`/`_timestamps_pendientes`, mira 2 para auto-repararse si uno falla o
  todavía no estaba agregado); `collect_5min`/`collect_1h`/`collect_6h` siguen existiendo para
  el backfill, donde el timestamp siempre es explícito. Sin esto la recolección en vivo iba
  ~2 horas atrasada y el "pronóstico de la próxima hora" era en realidad el de una hora que ya
  había terminado — y el walk-forward, que asume datos hasta el checkpoint, medía un escenario
  que en vivo nunca se daba;
  `backfill(interval, start_ts, end_ts, db, delay=1.0)` descarga un rango histórico (**siempre
  `delay >= 1.0`**, estimar tamaño y confirmar con el usuario antes de uno grande). Al backfillear
  meses/años de `precios_1h`, el cuello de botella no es la API (~1.6s/request) sino el propio
  `INSERT OR IGNORE`: con la tabla en varios millones de filas, cada insert tarda cada vez más
  porque SQLite tiene que mantener el índice único de la PK compuesta `(item_id, timestamp)` —
  ese índice es imprescindible para el `OR IGNORE` (así detecta duplicados) y **no se puede sacar
  sin perder esa garantía**; sacar el índice secundario de `timestamp` (no la PK) casi no cambia
  el tiempo, ya se probó. La única palanca que funciona de verdad es acotar el rango del backfill
  (`insertar_precios()` ya usa `executemany`, no loop de `execute()` — eso sí ayudó, ~2x).
  `job_horario` (cada hora, :05) refresca `resumen_actual`, reentrena todos los modelos de
  `modelos_config` con `cadencia='horaria'`/`estado='activo'` (ninguno por default — si el
  usuario no creó ningún modelo desde la app de escritorio todavía, no reentrena nada) y evalúa
  alertas; `job_diario` (03:00 UTC) hace lo mismo para `cadencia='diaria'` y calcula métricas de
  horizonte para los de tipo='regresor'; `job_semanal` (domingo 04:00 UTC, con catch-up si se
  perdió la ventana — ver
  `verificar_catchup_semanal`) corre la retención. `rellenar_huecos_al_inicio` además dispara
  `replay_historico.ejecutar_replay_modelo()` sobre los huecos detectados, **para cada modelo
  activo de `modelos_config` cuya tabla sea la de ese intervalo** — no para `global_horario`/
  `global_diario`, que es lo que hacía antes vía `ejecutar_replay()` y que dejó de tener sentido
  cuando `modelos_config` pasó a arrancar vacía (un hueco de una semana disparaba 168 fits de
  XGBoost sobre 200 ítems para modelos que nadie tiene, mientras los del usuario quedaban sin
  cubrir). Acota el límite inferior del hueco a `mantenimiento.RETENCION_DIAS[tabla]` (no a
  `MIN(timestamp)`): sin eso, un dato suelto viejo en la tabla hace que esto intente
  rellenar/replayear meses de historia que `mantenimiento.py` va a purgar en la próxima corrida
  semanal de todos modos. Loggea a `logs/` además de consola.
- **`base_de_datos.py`**: esquema SQLite. `OSRSBaseDatos.conectar()` es el único lugar que abre
  conexiones (con `busy_timeout` de 15s): el recolector, la app de escritorio y el dashboard
  escriben/leen sobre la misma DB y con el default de 5s una escritura que cae encima de otra
  falla en vez de esperar. Las tablas de series de tiempo (`precios_5m/1h/6h`,
  `precios_1h_diario`, `predicciones`) usan `PRIMARY KEY` compuesta + `INSERT OR IGNORE`/
  `OR REPLACE` — idempotentes a propósito, seguir el mismo patrón si se agrega una tabla nueva.
  `resumen_actual` y `model_metrics` guardan el estado del screener y del modelo (se
  regeneran/acumulan, no son series de tiempo). `obtener_precios_id(..., desde_timestamp=...,
  hasta_timestamp=...)` filtra por fecha en la query SQL, no trayendo todo el historial a pandas
  — `hasta_timestamp` es lo que permite simular "qué se sabía hasta este momento" para el replay
  histórico. La tabla `items` se puebla vía `guardar_items()` con `get_item_mapping()`, no
  manualmente. `_migrar_esquema()` aplica migraciones idempotentes (`ALTER TABLE`/`CREATE TABLE
  IF NOT EXISTS`) sobre una DB ya existente — seguir ese patrón, no editar directamente un
  `CREATE TABLE` que ya pudo haber corrido en una DB real. `model_metrics` tiene un índice
  único sobre `(COALESCE(item_id,-1), train_timestamp, model_name, horizonte_horas,
  modo_evaluacion)` y `predicciones` incluye `modo_evaluacion` en su PK — ambas `INSERT OR
  REPLACE`; un `model_name` nuevo para cada variante/experimento evita pisar filas de otro.
  `obtener_precios_multi(item_ids, tabla, ...)` trae varios ítems en una sola query — preferirla
  sobre un loop de `obtener_precios_id` por ítem (ese loop, con ~200 conexiones SQLite
  separadas, era el cuello de botella real de `build_training_set`). `modelos_config`
  (`crear_modelo_config`/`obtener_modelo_config`/`actualizar_modelo_config`/
  `eliminar_modelo_config`/`listar_modelos_config`) arranca **vacía** en una DB nueva — no hay
  ningún modelo sembrado por default (pedido explícito del usuario, ver `escritorio/`), así que
  `job_horario`/`job_diario` (`recolector.py`) no tienen nada que reentrenar hasta que el usuario
  cree uno. `contar_modelos_config_activos()` topea `MAX_MODELOS_ACTIVOS` (8) contando solo
  `cadencia != 'manual'` — un par regresor+clasificador horario (ver `escritorio/`) consume 2 de
  esos 8 slots.
- **`preprocesamiento.py`**: features **todas relativas** (retornos acumulados `ret_lag_k`,
  distancia a la media móvil `dist_ma_price_w`, volatilidad, spread, desbalance de volumen,
  encoding cíclico de hora y día de semana) y target = log-retorno del siguiente período, no
  precio crudo. Nada en niveles absolutos, a propósito: el target es una diferencia y un árbol
  no puede restar dos features, así que con lags en log-precio absoluto (la versión 1 de este
  archivo) el modelo solo podía memorizar rangos de precio por ítem — y encima `log_price`
  estaba excluido de las features, o sea que tenía que predecir un movimiento medido desde un
  punto de referencia que no veía. Medido sobre 24 ítems y 365 horas reales, 3 cortes
  temporales: 54.9% (IC95% 52.7-57.1) de accuracy direccional con niveles vs **68.8%**
  (66.7-70.8) con relativas. El MAE no mejora — ninguna de las dos le gana al baseline "el
  precio no cambia" (`baseline.py`), la ganancia está en la dirección.
  `_reindexar_a_grilla` deja cada serie sobre la grilla temporal regular antes de calcular
  nada: sin eso `shift(1)` significa "la fila anterior que exista" (que puede ser de hace diez
  horas, ver el bullet de `osrs_ge_api.py`) y el target deja de ser el movimiento de UN período.
  `FEATURES_VERSION` se guarda en el bundle y `prediccion.py` la verifica — un .pkl de otra
  versión no falla solo, predice ruido en silencio (`reindex(columns=...)` rellena con NaN y
  XGBoost los acepta). `build_training_set(..., hasta_timestamp=...)` arma el dataset
  multi-ítem, opcionalmente acotado a un momento del pasado; el `paso_segundos` se deriva de la
  tabla, nunca se pasa a mano.
- **`entrenador.py`**: entrena un `XGBRegressor` con split temporal (no aleatorio, para no
  filtrar futuro hacia el pasado) — `modo_evaluacion='holdout'` en vivo, `'walkforward'` en el
  replay (entrena con todo menos el último período y evalúa solo ahí, mucho más barato que
  repetir un split 80/20 en cada checkpoint histórico). `ahora_ts` simula un momento del pasado;
  `guardar_en_disco=False` evita pisar el `.pkl` durante el replay. Guarda el modelo en `models/`
  y persiste métricas/predicciones en la DB, taggeadas por `model_name`/`model_version` para que
  cualquier cantidad de modelos convivan sin pisarse — no hay ningún modelo sembrado por default
  (ver el bullet de `base_de_datos.py` y de `escritorio/`), `MODEL_NAME_HORARIO`/
  `MODEL_NAME_DIARIO`/`MODEL_NAME_CLASIF_F2P_100GP`/etc. son solo nombres de referencia que la
  app/el dashboard usan como default si un modelo con ese nombre exacto llega a existir.
  **Métricas (`accuracy_direccional`, una sola definición compartida)**: se mide sobre los
  períodos en que (a) el precio se movió más que `UMBRAL_CLASIF_PCT` y (b) el modelo se jugó
  por una dirección. Las dos condiciones hacen falta: sin (a), los empates hunden al regresor
  (el retorno real es exactamente 0 entre el 7% y el 35% de las horas según el ítem porque los
  precios de la API son enteros, y el regresor nunca predice 0 — la misma corrida daba 31.7%
  contando los empates y 55.9% sin contarlos); sin (b), el clasificador queda castigado por
  abstenerse, que no es equivocarse (uno que se juega en el 3% de los casos marcaba 29%, que
  se lee como "se equivoca casi siempre" en vez de "casi nunca opina"). Lo que evita que
  abstenerse sea un truco para inflar el número es `n_evaluado` — la app lo usa para el
  intervalo de confianza, y un modelo que se juega diez veces queda "indistinguible del azar"
  por ancho de intervalo, no por su número puntual. Antes cada modelo usaba su propia
  definición sobre poblaciones distintas y las dos se guardaban en la misma columna: parte de
  la ventaja del clasificador sobre el regresor que este archivo documentaba venía de ahí.
  `model_metrics.mae`/`rmse` son ahora SIEMPRE gp (antes la fila agregada iba en log-retorno y
  las de detalle por ítem en gp, en la misma columna); el espacio log vive en
  `mae_retorno`/`rmse_retorno`, y `n_evaluado` dice sobre cuántos movimientos se midió la
  accuracy (la app lo usa para el intervalo de confianza).
  `entrenar_clasificador_direccional()` (mismo archivo) es un `XGBClassifier` de 3 clases
  (sube/estable/baja, `UMBRAL_CLASIF_PCT`, subido de 0.5% a 1.0% porque el movimiento horario
  mediano de los ítems líquidos está entre 0.2% y 0.9%) en vez de derivar la dirección del
  signo de la regresión. `UMBRAL_COSTO_PCT` (2%, el impuesto GE) es el piso por debajo del cual
  una señal 'sube' no cubre el costo de operar salvo capturando el spread — `entrenar_...`
  loguea un warning si el umbral queda por debajo, en vez de dejarlo como default silencioso.
  Una configuración concreta (`solo_f2p=True, n_items=10,
  precio_minimo=100, excluir_item_ids=[2353, 449, 453]`, Steel bar/Adamantite ore/Coal) quedó
  validada con un test de significancia por permutación sobre 90 días/5.639 trades (gana
  significativamente más que el azar, p=0.002, aunque con la ganancia muy concentrada en un
  solo ítem — ver el comentario junto a `MODEL_NAME_CLASIF_F2P_100GP` para el detalle completo)
  — es una recomendación validada, no algo que corra automáticamente: el usuario tiene que
  crearlo él mismo. `precio_minimo`/`excluir_item_ids` son parámetros de
  `obtener_top_items_liquidez[_hasta]` (`base_de_datos.py`), enhebrados también por
  `replay_historico.ejecutar_replay_clasificador` y `backtest.simular_clasificador*`.
- **`baseline.py`**: reglas triviales sin entrenar nada (`baseline_flat`="no cambia",
  `baseline_momentum`="sigue la tendencia anterior"), guardadas en `model_metrics` con el mismo
  esquema para comparar de igual a igual — el regresor de `entrenador.py` apenas le gana a
  `baseline_flat` en error (MAE), la ventaja real está en el clasificador de arriba, no en XGBoost
  en sí.
- **`evaluacion.py`**: mide accuracy direccional y MAE a horizontes de 2..6 pasos (no solo 1),
  reusando el pronóstico recursivo de `prediccion.py` acotado a datos del pasado — solo se llama
  desde `job_diario` (caro, no se justifica en cada corrida horaria).
- **`replay_historico.py`**: al rellenar un hueco del pasado, reentrena en los momentos exactos
  en que `job_horario`/`job_diario` habrían corrido en vivo (`calcular_checkpoints`), usando solo
  datos disponibles hasta cada checkpoint — reanudable (`OSRSBaseDatos.existe_checkpoint`), no
  toca `resumen_actual` ni el `.pkl` productivo. Deliberadamente caro (prioriza fidelidad); deja
  un historial real de `modo_evaluacion='walkforward'` que alimenta `backtest.py`.
- **`metricas.py`**: screener de margen/ROI/volatilidad por ítem, pensado para mostrarse a un
  humano (no para features de modelo — eso es `preprocesamiento.py`). Puebla `resumen_actual`
  (sin filtrar liquidez — eso lo hace `filtrar_screener_liquido()` en el punto de consumo,
  porque ítems casi sin liquidez generan `roi_pct` absurdos; ese filtro acepta además
  `antiguedad_maxima_horas`, que los consumidores en vivo pasan en 6: `resumen_actual` guarda
  el último dato DISPONIBLE de cada ítem y para uno poco líquido puede ser de hace días,
  mostrado como si fuera el precio de ahora). `volumen_24h`/`pct_cambio_24h`/`pct_cambio_7d` se
  calculan sobre horas de RELOJ (`_ventana`), no sobre las últimas N filas — con huecos, 24
  filas pueden abarcar días y el filtro de liquidez sobrestimaba justo a los ítems más
  ilíquidos. `calcular_resumen_todos` trae el historial en lotes de `ITEMS_POR_LOTE` ítems con
  `obtener_precios_multi`, no una conexión SQLite por ítem (eran ~1.600 por corrida de
  `job_horario` con la DB casi vacía, ~4.600 con la DB llena, y si esto falla `job_horario`
  aborta el reentrenamiento entero). `dimensionar_oportunidad()` acota
  las unidades simuladas al volumen real esperado, no al `buy_limit` completo.
  `margen_neto_proyectado()` combina la predicción del modelo con el impuesto GE para una señal
  de flip con horizonte.
- **`prediccion.py`**: pronóstico recursivo hacia adelante (`pronosticar_item`, con
  `hasta_timestamp` opcional para reusarlo también en evaluación histórica, no solo en el
  dashboard). `cargar_modelo(model_name=...)` resuelve la ruta vía `entrenador.MODEL_PATHS`.
- **`backtest.py`**: simula la estrategia de flip sobre un rango histórico con precios reales
  (compra y venta ya conocidas) — usar_modelo=True/False para comparar "solo screener" vs
  "screener+modelo". Dos supuestos que ahora son explícitos y antes inflaban el resultado:
  (1) la decisión se toma al CIERRE del período `t` (cuando la API publica sus datos) y la
  compra ocurre en `t+1`, no a `avg_low_price(t)`, que es el promedio de un período ya
  terminado — con un movimiento horario mediano de 0.2-0.9%, una barra de ventaja alcanza para
  inventar toda la rentabilidad; (2) cada trade se valúa en sus **dos cotas**, optimista
  (comprar en la punta baja y vender en la alta: captura el spread, requiere que las dos
  órdenes se completen) y pesimista (cruzar el spread en las dos puntas). Medido sobre 365
  horas reales: la cota optimista cierra en ganancia entre el 15% y el 84% de las veces según
  el ítem, la pesimista entre el 0% y el 1% — **la diferencia entera es el spread, no el
  modelo**, y el resultado real de operar está entre las dos. `_resumen` devuelve las dos, y
  una comparación con/sin modelo solo vale si se sostiene en ambas. También respeta el reset de
  4h del límite de compra (`HORAS_BUY_LIMIT`); el capital sigue siendo ilimitado, así que
  `profit_total` es una cota superior. Rápido si el rango ya fue cubierto por
  `replay_historico.py` (lee la señal directo de `predicciones`); si no, cae a un fallback más
  lento vía `prediccion.pronosticar_item`. La señal de entrada es siempre la predicción de 1
  paso para el período de compra, también con horizontes mayores.
  `simular_clasificador()`/`simular_clasificador_walkforward()` son el equivalente para el
  clasificador direccional de `entrenador.py` (compra solo si predice "sube", horizonte fijo
  en 1 paso, sin recursión) — la versión walkforward llama
  `replay_historico.ejecutar_replay_clasificador()`, que no está enganchada a
  `rellenar_huecos_al_inicio` (hay que dispararla a mano sobre el rango que se quiera validar).
- **`alertas.py`**: manda por Telegram (Bot API) el top de oportunidades del screener filtrado
  por liquidez, con cooldown por ítem (`alertas_enviadas`) para no espamear. Requiere
  `TELEGRAM_BOT_TOKEN`/`TELEGRAM_CHAT_ID` como variables de entorno (ver
  `docs/despliegue_24_7.md`); sin ellas, loguea un warning y no rompe el job que la llama.
- **`mantenimiento.py`**: retención de todas las tablas de series de tiempo — ventanas en
  `RETENCION_DIAS` (`precios_1h`: **90 días**, recortado de 365 porque a esa escala cada insert
  del backfill se volvía carísimo — ver el bullet de `recolector.py`; `precios_6h`: **7 días**;
  `precios_5m`: 30 días), usado tanto acá como en `recolector.rellenar_huecos_al_inicio` para no
  tener la ventana duplicada en dos archivos. `archivar_datos_antiguos` agrega `precios_1h` a
  diario antes de purgar; `purgar_datos_antiguos` purga directo sin downsampling
  (`precios_5m`/`precios_6h`/`predicciones`, esta última con 180 días fijo);
  `podar_metricas_por_item` (poda solo el detalle por
  ítem de `model_metrics`, conserva el agregado indefinidamente) y `mantenimiento_vacuum`
  (`PRAGMA incremental_vacuum` — la DB ya está migrada a `auto_vacuum=INCREMENTAL`, confirmado
  con `PRAGMA auto_vacuum` -> `2`; sin esa migración el PRAGMA no libera nada, no falla).
  `ejecutar_mantenimiento_semanal()` agrupa todo — ver proyección de tamaño en
  `docs/escalabilidad_futura.md`.
- **`dashboard.py`**: panel de monitoreo en Streamlit (screener con filtro de liquidez, calidad
  del modelo en el tiempo por modelo/modo/horizonte, predicción vs realidad, pronóstico a
  futuro, señal direccional F2P). Selector de modelo (`global_horario`/`global_diario` por
  default) en los tabs relevantes — son solo nombres de referencia (ver el bullet de
  `entrenador.py`): si el usuario no creó un modelo con exactamente ese nombre desde la app de
  escritorio, esos tabs muestran "no hay modelo entrenado" en vez de romper (`FileNotFoundError`
  capturado en cada tab). El tab "Señal direccional (F2P)" es aparte porque un clasificador
  predice una clase categórica (baja/estable/sube), no un precio continuo — calculado en vivo con
  `prediccion.pronosticar_clase_item()` y cruzado con `margen_neto`/`roi_pct` de `resumen_actual`
  para que sea accionable, default `f2p10_100gp_clasif` (mismo caso: solo si existe). Nunca
  escribe en la DB.
- **`escritorio/`**: app de escritorio en PySide6/Qt para el usuario final — 4 tabs
  deliberadamente pocas (Inicio, Oportunidades, Mis modelos, Configuración), a diferencia del
  dashboard de Streamlit que tiene mucha más carga de información pensada para el desarrollador.
  `escritorio/main.py` es el punto de entrada (`python -m escritorio.main`).
  `escritorio/paginas/pagina_inicio.py` controla el hilo del recolector (arrancar/parar) y
  muestra progreso real del relleno de huecos/replay. `pagina_oportunidades.py` fusiona
  screener + señal direccional de cualquier clasificador activo en una sola tabla (no una
  variante F2P fija). `pagina_modelos.py` es la única pantalla que ESCRIBE en `modelos_config`:
  "+ Nuevo modelo" crea siempre un par regresor+clasificador (cadencia horaria, tabla
  precios_1h) sobre los ítems que el usuario elija, con la ventana de historial como único otro
  parámetro configurable — no hay ningún modelo protegido contra borrado, todos son del usuario
  (ver el bullet de `base_de_datos.py`: `modelos_config` arranca vacía). Al seleccionar un
  regresor en la tabla, muestra un gráfico (`pyqtgraph`) de predicted_price vs. actual_price
  (tabla `predicciones`) para un ítem elegido del propio modelo — el clasificador no tiene
  equivalente (no persiste un precio continuo) y muestra un mensaje en vez de un gráfico vacío.
  El walk-forward corre una vez al crear el modelo (todo el historial disponible en ese
  momento); "Rehacer walk-forward" (cualquier tipo, regresor o clasificador) pide una ventana en
  días y vuelve a correr `ejecutar_replay_modelo` acotado a eso — pensado para cubrir historial
  recolectado después de la creación sin recorrer meses ya cubiertos; resumible como siempre
  (`existe_checkpoint` saltea lo que ya corrió, no lo recalcula).
  `pagina_configuracion.py` edita el token de Telegram y la ruta de la DB, persistidos en
  `config.json` (`configuracion.py`).

## Convenciones del código

- Nombres de funciones/variables, docstrings y comentarios en **español**, consistente con el
  código existente.
- Cualquier dependencia nueva debe agregarse a `requirements.txt` (instalarla con pip, no
  asumir que ya está disponible en el entorno).
- No commitear artefactos generados: `*.db`, `*.sqlite`, `__pycache__/`, `models/`, `*.pkl`,
  `logs/` (ya cubiertos por `.gitignore` — mantenerlo así al agregar nuevos tipos de artefactos).

## Git: dos remotos, uno de solo lectura

Este repo (`OSRS-GE-mobile`) es una copia de trabajo separada del proyecto original del
usuario en GitHub, creada para experimentar con Claude Code Remote Control sin tocar el
original.

- `origin` → este repo, acá se pushea normalmente.
- `upstream` → el repo original del usuario, con el push deshabilitado a propósito
  (`git remote -v` debe mostrar una URL inválida en el push de `upstream`). **Nunca** intentar
  restaurar o pushear a `upstream`. Antes de cualquier `git push`, confirmar que el remote es
  `origin`.
