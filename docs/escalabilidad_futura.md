# Proyección de tamaño de la base de datos y criterios de migración

## Medición base (2026-08-25, antes de aplicar retención extendida)

`data/osrs_ge.db` pesaba 227 MiB (238,026,752 bytes) con:

| tabla | filas | historia real | filas/día medidas |
|---|---|---|---|
| `precios_5m` | ~1.03M | 6.15 días | ~167.500 (sin retención hasta este cambio) |
| `precios_1h` | ~2.75M | 55.9 días | ~49.300 (retención ya existente: 365 días) |
| `precios_6h` | ~9.121 | 0.5 días | ~12.160 (sin retención hasta este cambio) |

Ratio medido: ~3.79M filas de precios ⇒ **~60 bytes/fila** (incluye el overhead de índices —
las tablas de precios tienen PK compuesta `(item_id, timestamp)`, no `INTEGER PRIMARY KEY`, así
que SQLite mantiene el B-tree de la tabla más el índice único implícito de la PK más el índice
explícito de `timestamp`: tres estructuras por tabla).

## Sin la retención de `mantenimiento.py` (estado antes de este cambio, extrapolado)

Crecimiento combinado (`5m`+`1h`+`6h`): ~228.960 filas/día ⇒ ~13.7 MB/día.

- **6 meses**: +2.38 GB ⇒ DB ≈ 2.6 GB.
- **12 meses**: +4.77 GB ⇒ DB ≈ 5.0 GB, **sin techo** — sigue creciendo indefinidamente, más lo
  que sumen `predicciones`/`model_metrics` sin poda (con dos modelos + replay histórico, que es
  bastante más que antes).

## Con la retención aplicada (`purgar_datos_antiguos`, `podar_metricas_por_item`, ver `mantenimiento.py`)

- `precios_5m` (tope 30 días): converge a ~5.0M filas ≈ **~300 MB**, y ahí se estabiliza.
- `precios_1h` (retención ya existente, 365 días): a los 6 meses (55.9 + 182 = 237.9 días) todavía
  no alcanza el tope ⇒ ≈11.7M filas ≈ **~700 MB**; a los 12 meses ya lo alcanza y converge a
  ~18M filas ≈ **~1.08 GB** (consistente con la estimación de "~40M filas/año en el peor caso"
  que ya tenía el propio código, sobre un peor caso de ~4.600 ítems/hora — el promedio real
  medido es bastante menor, ~2.053 ítems/hora).
- `precios_6h` (tope 365 días): converge a ~4.4M filas ≈ **~266 MB**.
- `predicciones` (tope 180 días) y `model_metrics` (detalle por ítem podado a 90 días, agregado
  indefinido): del orden de **decenas de MB** — no dominan el total, incluso con dos modelos
  (`global_horario`/`global_diario`) y las filas adicionales del replay histórico.

**Total estimado: ≈1.2 GB a los 6 meses, ≈1.8-2 GB a los 12 meses, y desde ahí el tamaño se
estabiliza** — todas las tablas grandes alcanzan su tope de retención alrededor del mes 13, en
vez de seguir creciendo para siempre.

## ¿Sigue siendo apropiado SQLite?

**Sí — no hace falta migrar ahora.** A esta escala (~2 GB en estado estable, un solo proceso
escritor — el recolector —, con lectores concurrentes vía WAL — el dashboard y scripts como
`backtest.py`) es exactamente el patrón para el que SQLite + WAL está pensado. El cuello de
botella real en este tipo de proyectos casi nunca es el tamaño bruto del archivo, sino:

1. **Consultas sin acotar** — ya resuelto: todas las lecturas por rango de tiempo pasan
   `desde_timestamp`/`hasta_timestamp` en la query SQL (`obtener_precios_id`), no traen todo a
   pandas para recortar después.
2. **Falta de mantenimiento** — ya resuelto: retención (`mantenimiento.py`) + `incremental_vacuum`
   periódico (ver `docs/migracion_auto_vacuum.md`).
3. **Un solo escritor por diseño** — el recolector es el único proceso que escribe; dashboard,
   backtest y scripts de análisis son solo lectores. Mientras esto se mantenga así, WAL evita
   que los lectores bloqueen al escritor (o viceversa).

## Criterios concretos para reconsiderar Postgres/TimescaleDB (u otra alternativa)

Ninguno de estos aplica hoy — quedan acá como señales concretas a vigilar, no como una alarma:

- La base de datos supera de forma **sostenida** ~5 GB (no un pico puntual antes de que corra la
  retención semanal) — señal de que la retención configurada ya no alcanza para el volumen real,
  no necesariamente de que SQLite en sí sea el problema (primero revisar si hay que acortar
  ventanas de retención antes de asumir que hace falta cambiar de motor).
- Aparece un **segundo proceso escritor concurrente** (ej. una app web que además escribe, no
  solo lee) — el diseño actual de "un solo escritor" es lo que hace que WAL alcance; con
  múltiples escritores concurrentes, un motor con locking más granular (Postgres) empieza a
  tener ventajas reales.
- El dashboard/las alertas necesitan **agregaciones que un solo archivo SQLite ya no sirve con
  la latencia deseada** (ej. dashboards en tiempo real sobre agregaciones de millones de filas,
  actualizándose cada pocos segundos) — no es el caso hoy (refresh manual del dashboard, alertas
  con cadencia horaria).
- Se necesita **replicación o alta disponibilidad** (el archivo `.db` vive en una sola máquina,
  sin failover) — relevante si esto deja de ser un proyecto personal de un usuario y pasa a
  necesitar disponibilidad garantizada para terceros.

Si en el futuro se cumple alguno de estos, la migración más natural sería a PostgreSQL (con o sin
la extensión TimescaleDB para las tablas de series de tiempo) antes que a otra alternativa —
mismo modelo relacional y SQL, así que la mayor parte de las queries de este proyecto se
portarían con cambios menores.
