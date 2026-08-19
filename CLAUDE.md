# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Qué es este proyecto

OSRS GE Predictor de Precios: pipeline de datos que recolecta precios de compra/venta del Grand
Exchange de Old School RuneScape desde la API pública `prices.runescape.wiki`, los almacena en
SQLite, genera features de series de tiempo y (en desarrollo) entrena un modelo para predecir
precios. El fin último es exponer esto en una app web.

## Comandos

```bash
pip install -r requirements.txt      # instalar dependencias
python recolector.py                  # recolector en vivo: puebla catálogo de items y
                                       # corre indefinidamente (schedule: cada 5 min / 1h)
```
No hay suite de tests ni linter configurados todavía. `entrenador.py` aún no tiene el
entrenamiento activo (está comentado) — no asumas que produce un modelo utilizable.

## Arquitectura del pipeline

```
osrs_ge_api.py (OSRSGeAPI)  →  recolector.py  →  base_de_datos.py (OSRSBaseDatos, SQLite)
                                                        ↓
                                    preprocesamiento.py → entrenador.py (WIP) → [futuro: app web]
```

- **`osrs_ge_api.py`**: cliente de la API. Los endpoints `/5m`, `/1h`, `/6h` devuelven un
  snapshot de **todos** los ítems del juego en cada llamada (no aceptan filtrar por item_id);
  el filtrado a ítems específicos, si se necesita, se hace después sobre el DataFrame.
- **`recolector.py`**:
  - `ITEM_IDS` vacío = sin filtro (recolecta todos los ítems). Solo llenarlo para pruebas
    puntuales.
  - `collect_5min` / `collect_1h` / `collect_6h` recolectan un snapshot puntual.
  - `backfill(interval, start_ts, end_ts, db, delay=1.0)` descarga un rango histórico
    llamando a la API una vez por paso del intervalo — **siempre usar `delay >= 1.0`** entre
    requests (la API es un servicio comunitario, no hay que saturarla). Antes de un backfill
    grande, estimar filas/tamaño resultante y confirmar con el usuario.
  - Actualmente solo se recolecta 5m bajo demanda/pruebas; el uso normal es **1h y 6h**
    (5m genera ~12x más volumen y no se justifica para el uso actual del proyecto).
- **`base_de_datos.py`**: cada tabla de precios (`precios_5m`, `precios_1h`, `precios_6h`)
  usa `PRIMARY KEY (item_id, timestamp)` + `INSERT OR IGNORE` — los inserts son idempotentes
  a propósito, así que recolectar/backfillear el mismo rango dos veces no duplica datos.
  Si se agrega una tabla de series de tiempo nueva, seguir el mismo patrón (PK compuesta +
  índice en `timestamp` si se van a hacer consultas por rango de fecha entre ítems). La tabla
  `items` (id → nombre/members/buy_limit) se puebla vía `guardar_items()` con
  `get_item_mapping()`, no manualmente.
- **`preprocesamiento.py`**: genera lags, medias móviles y encoding cíclico de hora a partir
  de una serie de un solo ítem; excluye la columna target del set de features para no generar
  leakage.

## Convenciones del código

- Nombres de funciones/variables, docstrings y comentarios en **español**, consistente con el
  código existente.
- Cualquier dependencia nueva debe agregarse a `requirements.txt` (instalarla con pip, no
  asumir que ya está disponible en el entorno).
- No commitear artefactos generados: `*.db`, `*.sqlite`, `__pycache__/`, `models/`, `*.pkl`
  (ya cubiertos por `.gitignore` — mantenerlo así al agregar nuevos tipos de artefactos).

## Git: dos remotos, uno de solo lectura

Este repo (`OSRS-GE-mobile`) es una copia de trabajo separada del proyecto original del
usuario en GitHub, creada para experimentar con Claude Code Remote Control sin tocar el
original.

- `origin` → este repo, acá se pushea normalmente.
- `upstream` → el repo original del usuario, con el push deshabilitado a propósito
  (`git remote -v` debe mostrar una URL inválida en el push de `upstream`). **Nunca** intentar
  restaurar o pushear a `upstream`. Antes de cualquier `git push`, confirmar que el remote es
  `origin`.
