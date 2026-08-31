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
- `precios_1h` (**90 días** — recortado de 365 después de esta medición, ver
  `mantenimiento.RETENCION_DIAS` y el bullet de `recolector.py` en `CLAUDE.md`: a 365 días cada
  insert del backfill se volvía carísimo): con el throughput real medido hoy (~29.500 filas/día,
  ver más abajo) converge a ≈2.7M filas ≈ **~160 MB** en vez de la ~1.08 GB estimada acá
  originalmente con 365 días — la retención más corta es justo lo que evita ese crecimiento.
- `precios_6h` (tope 365 días): converge a ~4.4M filas ≈ **~266 MB**.
- `predicciones` (tope 180 días) y `model_metrics` (detalle por ítem podado a 90 días, agregado
  indefinido): del orden de **decenas de MB** — no dominan el total.

**Total estimado en régimen estable (con la retención de 90 días en `precios_1h`): ≈750 MB-1 GB**,
no los ≈1.8-2 GB que daba la primera versión de esta cuenta.

### Medición real (2026-08-31)

`data/osrs_ge.db` pesa **965 MB** hoy, con `precios_1h` en 4.53M filas — por encima de la
proyección de régimen estable de arriba, porque a esta fecha `precios_1h` tiene 153.6 días de
historia real (no 90): la retención de 90 días está configurada pero `mantenimiento.
purgar_datos_antiguos`/`job_semanal` no vienen corriendo con regularidad (no había ningún proceso
`recolector.py` corriendo al momento de esta medición) — no es un problema de la proyección en sí,
es que la purga todavía no alcanzó su régimen estable. Vale la pena confirmar que `job_semanal`
esté corriendo de verdad (ver `docs/despliegue_24_7.md`) antes de tomar el número de 965 MB como
el tamaño "de crucero" del proyecto.

## ¿Sigue siendo apropiado SQLite?

**Sí — no hace falta migrar ahora.** A esta escala (~2 GB en estado estable, un solo proceso
escritor — el recolector —, con lectores concurrentes vía WAL — el dashboard y scripts como
`backtest.py`) es exactamente el patrón para el que SQLite + WAL está pensado. El cuello de
botella real en este tipo de proyectos casi nunca es el tamaño bruto del archivo, sino:

1. **Consultas sin acotar** — ya resuelto: todas las lecturas por rango de tiempo pasan
   `desde_timestamp`/`hasta_timestamp` en la query SQL (`obtener_precios_id`), no traen todo a
   pandas para recortar después.
2. **Falta de mantenimiento** — ya resuelto: retención (`mantenimiento.py`) + `incremental_vacuum`
   periódico (la DB ya está migrada a `auto_vacuum=INCREMENTAL`, confirmado con
   `PRAGMA auto_vacuum` -> `2`).
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
