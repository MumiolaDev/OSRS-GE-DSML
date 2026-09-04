# CLAUDE.md

Guía para Claude Code (claude.ai/code) al trabajar en este repositorio.

## Qué es

Pipeline que recolecta precios de compra/venta del Grand Exchange de Old School RuneScape desde
`prices.runescape.wiki`, los guarda en SQLite, calcula un screener de oportunidades de flip y
entrena modelos XGBoost que predicen el **margen neto ejecutable** del próximo período (tipo
`'spread'`). Corre 24/7 (`docs/despliegue_24_7.md`), alerta por Telegram, y tiene una app de
escritorio en PySide6 para el usuario final más un dashboard Streamlit de solo lectura para el
desarrollador.

**No hay ningún modelo por default** — pedido explícito del usuario: *"no quiero que haya ningún
modelo por default, todos tienen que poder eliminarse"*. `modelos_config` arranca vacía, el
usuario crea los suyos desde la app, y todo el pipeline funciona sin ninguno (el screener no
depende de modelos).

## Comandos

```bash
pip install -r requirements.txt
python recolector.py           # recolección 24/7 + jobs programados (normalmente lo corre systemd)
python -m escritorio.main      # app de escritorio
python estado.py               # ¿está sano el pipeline? (--json para waybar, --breve)
streamlit run dashboard.py     # dashboard de monitoreo, solo lectura
python -m pytest tests/        # 170 tests: funciones puras + DB temporal, nunca la API real
./deploy/instalar_servicio.sh  # Linux: instala el servicio de systemd de usuario
```

Fuera del loop del recolector, para correr a mano: `metricas.py` (screener), `entrenador.py`,
`baseline.py` (reglas triviales de comparación), `replay_historico.py` (walk-forward),
`backtest.py`, `alertas.py`, `mantenimiento.py`. No hay linter configurado.

## Arquitectura

```
osrs_ge_api.py → recolector.py → base_de_datos.py (SQLite)
                      ├─→ metricas.py ──────────→ resumen_actual (screener)
                      ├─→ preprocesamiento.py → entrenador.py → model_metrics / predicciones
                      │                              └─→ evaluacion.py (horizontes, solo job_diario)
                      ├─→ replay_historico.py (walk-forward al rellenar huecos o al crear un modelo)
                      ├─→ alertas.py (Telegram, al final de job_horario)
                      └─→ mantenimiento.py (retención + vacuum, semanal)

prediccion.py y backtest.py reusan el modelo guardado; no están en el loop del recolector.
escritorio/ (PySide6) lee y ESCRIBE modelos_config; dashboard.py (Streamlit) solo lee.
bloqueo.py da un candado flock: un solo recolector por DB. estado.py reporta la salud.
```

Jobs del recolector: **`job_horario`** (:05) refresca el screener, reentrena los modelos de
cadencia horaria y evalúa alertas; **`job_diario`** (03:00 UTC) hace lo mismo para la cadencia
diaria y calcula métricas de horizonte; **`job_semanal`** (dom 04:00 UTC, con catch-up si se
perdió la ventana) corre la retención.

## Módulos: lo que no es obvio

- **`osrs_ge_api.py`** — Todas las requests salen por una `Session` compartida con `timeout=30` y
  reintentos con backoff. **Nunca sacar el timeout**: un socket colgado congela el scheduler
  entero de un proceso 24/7, sin log ni excepción. Los endpoints `/5m`, `/1h`, `/6h` devuelven
  **todos** los ítems del juego en cada llamada (no aceptan filtrar por id). Tres gotchas
  verificados empíricamente y no documentados por la API: (1) el `timestamp` debe ser múltiplo
  exacto del step (300/3600/21600) o devuelve `400`; (2) no hay datos antes del **2021-03-08** y
  se contesta `200` con `data` vacío, no un error; (3) **pedir sin `timestamp` devuelve el bucket
  anterior al último cerrado**, por eso la recolección programada siempre pasa uno explícito.
  `_procesar_data_historicals` descarta el ítem si le falta una de las dos puntas de precio, lo
  que deja huecos reales (28% de las horas en un Elysian, 96% en un 3rd age) — de ahí el
  reindexado a grilla de `preprocesamiento.py`.

- **`recolector.py`** — `schedule` alineado al reloj de pared en UTC (no a cuándo arrancó el
  proceso), un minuto después del cierre de cada bucket. `collect_programado` pide explícitamente
  los últimos buckets cerrados que falten, mirando 2 hacia atrás para auto-repararse.
  **`precios_5m` se guarda solo para los ítems de los modelos activos**
  (`INTERVALOS_ACOTADOS_A_MODELOS` / `_filtro_items`): guardarlo entero tendía a 634 MB para algo
  que ni el screener ni los modelos leen; acotado son ~18 MB. `precios_1h`/`precios_6h` siguen
  guardando todo, porque el screener y el ranking de liquidez cubren el catálogo completo. En el
  filtro, `None` = todos y `[]` = ninguno (saltea la request). `RELLENO_MAXIMO_DIAS` acota cuánto
  hacia atrás rellena huecos al arrancar. `backfill()`: **siempre `delay >= 1.0`**, y estimar el
  tamaño y confirmarlo con el usuario antes de uno grande — el cuello de botella no es la API
  sino el `INSERT OR IGNORE` con la tabla en millones de filas; el índice único de la PK es
  imprescindible para el `OR IGNORE` y no se puede sacar (sacar el secundario de `timestamp` ya
  se probó y no cambia nada), así que la única palanca real es acotar el rango.

- **`base_de_datos.py`** — `conectar()` es el **único** lugar que abre conexiones (con
  `busy_timeout` de 15s): recolector, app y dashboard escriben sobre la misma DB. Las tablas de
  series de tiempo usan PK compuesta + `INSERT OR IGNORE`/`OR REPLACE`, idempotentes a propósito
  — seguir ese patrón al agregar una. `_migrar_esquema()` aplica migraciones idempotentes; **no
  editar un `CREATE TABLE` que ya pudo haber corrido en una DB real**. `obtener_precios_multi()`
  trae varios ítems en una query: preferirla siempre sobre un loop de `obtener_precios_id` (ese
  loop, con ~200 conexiones separadas, era el cuello de botella real). `hasta_timestamp=` es lo
  que permite simular "qué se sabía hasta este momento". `MAX_MODELOS_ACTIVOS` = 8.

- **`preprocesamiento.py`** — Features **todas relativas**, a propósito: el target es una
  diferencia y un árbol no puede restar dos features, así que con lags en nivel el modelo solo
  memoriza rangos de precio por ítem. Medido sobre 24 ítems y 365 horas: 54.9% de accuracy
  direccional con niveles vs **68.8%** con relativas. `_reindexar_a_grilla` corre primero: sin
  eso `shift(1)` significa "la fila anterior que exista", que puede ser de hace diez horas, y el
  target deja de ser el movimiento de un período. Dos targets: `target` (log-retorno) y
  `target_margen` (margen neto ejecutable del período siguiente, el del modelo de spread).
  `FEATURES_VERSION` viaja en el bundle y `prediccion.py` la verifica — un `.pkl` de otra versión
  no falla solo, predice ruido en silencio.

- **`entrenador.py`** — `entrenar_desde_config()` es el único punto de entrada y ramifica por
  `tipo`. **`entrenar_modelo_spread()` es el que usa la app y el único rentable**: medido
  walk-forward sobre 720 horas de decisión y 80 ítems, elegir qué comprar rindió **+4.71%** por
  operación con el modelo de spread, +2.27% con el screener puro, +0.25% al azar y **−0.38%** con
  el regresor de precio. El motivo es estructural: cuando llega el primer período en el que se
  puede comprar, la suba que el regresor predijo ya ocurrió y se paga. El modelo no predice mejor
  el futuro — baja el listón de ejecución (necesita que se completen las dos órdenes el 60.6% de
  las veces, contra 93.6% comprando a ciegas). Su métrica es `rentabilidad_top_k()`, medida sobre
  el ranking y no sobre el universo, porque la decisión real es comprar unos pocos ítems.
  **`accuracy_direccional` tiene una sola definición compartida** y se mide solo donde (a) el
  precio se movió más que `UMBRAL_CLASIF_PCT` y (b) el modelo se jugó por una dirección: sin (a)
  los empates hunden al regresor (el retorno real es exactamente 0 entre el 7% y el 35% de las
  horas, porque los precios de la API son enteros), y sin (b) el clasificador queda castigado por
  abstenerse, que no es equivocarse. Lo que impide que abstenerse sea un truco es `n_evaluado`,
  con el que la app calcula el intervalo de confianza. `model_metrics.mae`/`rmse` son **siempre
  gp**; el espacio log vive en `mae_retorno`/`rmse_retorno`. El regresor de precio y el
  clasificador direccional siguen existiendo solo como comparación: la app ya no los crea. Todo
  va taggeado por `model_name`/`model_version`, así que un nombre nuevo por experimento evita
  pisar filas de otro.

- **`metricas.py`** — Screener pensado para mostrarse a un humano, no para features de modelo.
  `resumen_actual` se puebla **sin** filtrar liquidez; el filtro va en el punto de consumo
  (`filtrar_screener_liquido`, con `antiguedad_maxima_horas=6` en vivo, porque la tabla guarda el
  último dato *disponible* de cada ítem y en uno ilíquido puede ser de hace días, mostrado como
  si fuera de ahora). `volumen_24h`/`pct_cambio_*` se calculan sobre horas de **reloj**, no sobre
  las últimas N filas: con huecos, 24 filas pueden abarcar días. `calcular_resumen_todos` trae el
  historial en lotes con `obtener_precios_multi`; si falla, `job_horario` aborta el reentreno.

- **`prediccion.py`** — Pronóstico recursivo hacia adelante, con `hasta_timestamp` opcional para
  reusarlo en evaluación histórica. `pronosticar_margen_items()` es lo que ordena la pantalla de
  Oportunidades.

- **`replay_historico.py`** — Reentrena en los momentos exactos en que los jobs habrían corrido
  en vivo, usando solo datos disponibles hasta cada checkpoint. Reanudable
  (`existe_checkpoint`), no toca `resumen_actual` ni el `.pkl` productivo, y es deliberadamente
  caro: prioriza fidelidad. Deja el historial de `modo_evaluacion='walkforward'` que alimenta
  `backtest.py`.

- **`backtest.py`** — Dos supuestos que antes inflaban el resultado y ahora son explícitos: (1)
  la decisión se toma al **cierre** del período `t` y la compra ocurre en `t+1` (con un movimiento
  horario mediano de 0.2-0.9%, una barra de ventaja alcanza para inventar toda la rentabilidad);
  (2) cada trade se valúa en sus **dos cotas**, capturando el spread y cruzándolo. Medido sobre
  365 horas: la optimista cierra en ganancia entre el 15% y el 84% de las veces según el ítem, la
  pesimista entre el 0% y el 1% — **la diferencia entera es el spread, no el modelo**, y una
  comparación con/sin modelo solo vale si se sostiene en las dos. Respeta el reset de 4h del
  límite de compra; el capital es ilimitado, así que `profit_total` es una cota superior.

- **Ejecución (medido sobre `precios_5m`, muestra chica)** — De los ítems del top-5 por margen:
  78% de las compras se llenan, 52% de los flips se completan dentro de la hora y 65% en dos; de
  lo que queda colgado, el 78% recupera el precio de venta dentro de 6h. La esperanza queda
  positiva en todas las ramas (+2.74% a +4.77% por intento). Dos conclusiones operativas: no
  forzar la venta dentro de la hora, y **no ceder margen para perseguir el fill** (probado a
  0.25%/0.5%/1% peor: la esperanza empeora). El proxy es optimista (que un bucket haya tocado el
  precio no prueba que *mi* orden se llenó) y todavía no corre solo.

- **`mantenimiento.py`** — `RETENCION_DIAS`: `precios_1h` 90 días, `precios_5m` 14, `precios_6h`
  7. `archivar_datos_antiguos` agrega 1h a diario antes de purgar; `podar_metricas_por_item` poda
  solo el detalle y conserva el agregado. La DB ya está migrada a `auto_vacuum=INCREMENTAL` (sin
  esa migración el PRAGMA no libera nada y tampoco falla).

- **`alertas.py`** — Telegram con cooldown por ítem. Credenciales por variable de entorno o
  `config.json` (`configuracion.py`, el entorno gana); sin ellas loguea un warning y no rompe el
  job que la llama.

- **`bloqueo.py` / `estado.py` / `deploy/`** — `bloqueo.py` es un candado `flock` sobre
  `<db_path>.lock`: un solo recolector por DB. Es `flock` y no un archivo de PID a propósito,
  porque el kernel lo libera solo cuando el proceso muere, como sea que muera. `estado.py`
  responde "¿el pipeline está sano?" de solo lectura, y su veredicto sale de la **antigüedad del
  último dato horario**, no de si el proceso vive: un recolector colgado sigue verde en
  `systemctl status` mientras hace horas que no inserta una fila. `deploy/` tiene las plantillas
  de la unidad de systemd (de **usuario**: escribe en el home, no necesita privilegios), la
  unidad `OnFailure` que notifica, el lanzador `.desktop` y el instalador idempotente.

- **`escritorio/`** — 4 tabs, deliberadamente pocas (Inicio, Oportunidades, Mis modelos,
  Configuración), a diferencia del dashboard que es para el desarrollador. Corre nativa en
  Wayland/Hyprland. **Oportunidades** ordena por el margen predicho para la próxima hora, que es
  la respuesta a "qué comprar" — y a diferencia del `margen_neto` del screener (el que ya se vio y
  puede haber desaparecido) es el del período en que efectivamente se puede operar.
  `pagina_modelos.py` es la única pantalla que **escribe** `modelos_config`: crea modelos de tipo
  `'spread'`, ninguno protegido contra borrado. `pagina_inicio.py` controla el hilo del
  recolector, y si detecta uno corriendo en otro proceso (el candado) pasa a modo monitor.

- **`dashboard.py`** — Solo lectura, nunca escribe. Los nombres de modelo que trae por default
  son solo referencias: si el usuario no creó uno con ese nombre exacto, el tab muestra "no hay
  modelo entrenado" en vez de romper.

## Convenciones

- Nombres, docstrings y comentarios **en español**, consistente con el código existente.
- Cualquier dependencia nueva va a `requirements.txt` (instalarla, no asumirla).
- No commitear artefactos generados: `*.db`, `*.sqlite`, `__pycache__/`, `models/`, `*.pkl`,
  `logs/`, `config.json`, `*.db.lock` — ya cubiertos por `.gitignore`, mantenerlo así.

## Git: dos remotos, uno de solo lectura

Este repo (`OSRS-GE-mobile`) empezó como copia de trabajo del proyecto original del usuario en
GitHub, para experimentar con Claude Code Remote Control sin tocarlo.

- `origin` → este repo, acá se pushea normalmente.
- `upstream` → el repo original, con el push deshabilitado a propósito. **Nunca** intentar
  restaurarlo ni pushearle. Antes de cualquier `git push`, confirmar que el remote es `origin`.
