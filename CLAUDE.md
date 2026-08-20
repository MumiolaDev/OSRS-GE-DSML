# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Qué es este proyecto

OSRS GE Predictor de Precios: pipeline que recolecta precios de compra/venta del Grand Exchange
de Old School RuneScape desde la API pública `prices.runescape.wiki`, los almacena en SQLite,
calcula un screener de oportunidades de flip y entrena un modelo (XGBoost) que predice el precio
del siguiente período. Corre pensado para 24/7 (ver `docs/despliegue_24_7.md`) y expone un
dashboard de monitoreo en Streamlit; la futura "app web" para el usuario final todavía no existe.

## Comandos

```bash
pip install -r requirements.txt   # instalar dependencias
python recolector.py              # recolector 24/7: puebla catálogo, recolecta 5m/1h en loop,
                                   # y dispara job_diario (screener+reentreno) y job_semanal (retención)
python metricas.py                # recalcula resumen_actual (screener) a partir de precios_1h
python entrenador.py              # entrena el modelo global sobre los ítems más líquidos
python mantenimiento.py           # archiva a diario y purga precios_1h fuera de la retención
streamlit run dashboard.py        # dashboard de monitoreo, solo lectura
```
No hay suite de tests ni linter configurados todavía.

## Arquitectura del pipeline

```
osrs_ge_api.py → recolector.py → base_de_datos.py (SQLite)
                     │  (schedule interno: 5m/1h + job_diario + job_semanal)
                     ├─→ metricas.py ────────────→ resumen_actual (screener)
                     ├─→ preprocesamiento.py → entrenador.py → model_metrics / predicciones
                     └─→ mantenimiento.py (retención/downsampling)

dashboard.py (Streamlit) lee resumen_actual / model_metrics / predicciones — solo lectura.
```

- **`osrs_ge_api.py`**: cliente de la API. Los endpoints `/5m`, `/1h`, `/6h` devuelven un
  snapshot de **todos** los ítems del juego en cada llamada (no aceptan filtrar por item_id);
  el filtrado a ítems específicos, si se necesita, se hace después sobre el DataFrame.
- **`recolector.py`**: loop de `schedule` pensado para correr indefinidamente. `ITEM_IDS` vacío
  = sin filtro (uso normal); `collect_5min`/`collect_1h`/`collect_6h` recolectan un snapshot
  puntual; `backfill(interval, start_ts, end_ts, db, delay=1.0)` descarga un rango histórico
  (**siempre `delay >= 1.0`**, estimar tamaño y confirmar con el usuario antes de uno grande).
  `job_diario` (03:00) refresca `resumen_actual` y reentrena el modelo; `job_semanal` (domingo)
  corre la retención. Loggea a `logs/` además de consola.
- **`base_de_datos.py`**: esquema SQLite. Las tablas de series de tiempo (`precios_5m/1h/6h`,
  `precios_1h_diario`, `predicciones`) usan `PRIMARY KEY` compuesta + `INSERT OR IGNORE`/
  `OR REPLACE` — idempotentes a propósito, seguir el mismo patrón si se agrega una tabla nueva.
  `resumen_actual` y `model_metrics` guardan el estado del screener y del modelo (se
  regeneran/acumulan, no son series de tiempo). `obtener_precios_id(..., desde_timestamp=...)`
  filtra por fecha en la query SQL, no trayendo todo el historial a pandas. La tabla `items`
  se puebla vía `guardar_items()` con `get_item_mapping()`, no manualmente.
- **`preprocesamiento.py`**: features en espacio logarítmico (lags y medias móviles de
  log-precio/log-volumen, encoding cíclico de hora y día de semana) y target = log-retorno del
  siguiente período, no precio crudo — necesario para que un modelo global sea comparable entre
  ítems de escalas de precio muy distintas. `build_training_set()` arma el dataset multi-ítem.
- **`entrenador.py`**: entrena un único `XGBRegressor` global sobre los ~200 ítems más líquidos
  (`obtener_top_items_liquidez`), con split temporal (no aleatorio, para no filtrar futuro hacia
  el pasado). Guarda el modelo en `models/` y persiste métricas/predicciones en la DB.
- **`metricas.py`**: screener de margen/ROI/volatilidad por ítem, pensado para mostrarse a un
  humano (no para features de modelo — eso es `preprocesamiento.py`). Puebla `resumen_actual`.
- **`mantenimiento.py`**: retención de datos — agrega a resolución diaria (`precios_1h_diario`)
  y purga de `precios_1h` lo más viejo que la ventana de retención (default 365 días).
- **`dashboard.py`**: panel de monitoreo en Streamlit (screener, calidad del modelo en el
  tiempo, predicción vs realidad por ítem). Nunca escribe en la DB.

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
