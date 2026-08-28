# Migrar la base de datos a `auto_vacuum=INCREMENTAL`

`mantenimiento.py` purga filas viejas de varias tablas (`archivar_datos_antiguos`,
`purgar_datos_antiguos`, `podar_metricas_por_item`), pero SQLite **no reduce el tamaño del
archivo `.db` con un `DELETE`** — las páginas quedan marcadas como libres dentro del archivo,
pero el archivo en sí no achica hasta que corre un `VACUUM`.

`mantenimiento_vacuum()` (llamada desde `job_semanal`) usa `PRAGMA incremental_vacuum`, que
libera esas páginas de forma incremental y sin bloquear la base de datos por mucho tiempo — pero
**solo funciona si la base de datos ya está en modo `auto_vacuum=INCREMENTAL`**. Este modo no se
puede activar con un `PRAGMA` suelto sobre una base de datos que ya tiene tablas y datos: hay que
migrarla una vez, a mano, con la base de datos detenida.

Mientras esta migración no se haga, `mantenimiento_vacuum()` no falla ni rompe nada — el
`PRAGMA incremental_vacuum` simplemente no tiene páginas que liberar y no hace nada.

## Cuándo hacerlo

No es urgente ni bloqueante para el resto del pipeline. Vale la pena hacerlo una vez que:

- Ya se aplicó la retención extendida (`precios_5m`/`precios_6h`/`predicciones`/`model_metrics`,
  ver `mantenimiento.py`) y las tablas grandes empezaron a purgar filas de verdad (no en una base
  de datos recién creada, donde no hay nada que purgar todavía).
- Se puede coordinar una ventana corta con el recolector detenido (la migración necesita acceso
  exclusivo al archivo).

## Pasos

1. **Detener `recolector.py`** (el servicio NSSM o la tarea programada, ver
   `docs/despliegue_24_7.md`) — nada más debe estar leyendo ni escribiendo el archivo `.db`
   mientras corre esto.
2. **Backup del archivo actual**: copiar `data/osrs_ge.db` (y los archivos `-wal`/`-shm` si
   existen junto a él) a otro lado antes de tocar nada.
3. **Correr la migración** (`sqlite3 data/osrs_ge.db`, o un script Python con `sqlite3.connect`):

   ```sql
   PRAGMA auto_vacuum = INCREMENTAL;
   VACUUM;
   ```

   El `VACUUM` reescribe el archivo completo — necesita temporalmente hasta ~2x el espacio en
   disco del archivo original, y puede tardar (proporcional al tamaño de la base de datos; con
   los ~200-300 MB actuales, del orden de minutos, no horas). Es el único paso de este proceso
   que bloquea la base de datos por completo.
4. **Verificar** que quedó en el modo correcto:

   ```sql
   PRAGMA auto_vacuum;
   ```

   Debe devolver `2` (INCREMENTAL). Este valor queda persistido en el header del archivo `.db` —
   no hace falta repetir la migración después, ni volver a correr `VACUUM` completo nunca más
   (de acá en adelante, `mantenimiento_vacuum()` hace el trabajo incremental automáticamente).
5. **Reiniciar `recolector.py`** y confirmar en `logs/recolector.log` que retomó la recolección
   normalmente.

## Por qué no se automatiza

Activar `auto_vacuum=INCREMENTAL` sobre una base de datos que ya tiene tablas requiere el
`VACUUM` completo del paso 3, que bloquea la base de datos entera mientras corre — no es
aceptable meterlo en un job periódico automático (`job_semanal` corre con el recolector viviendo
en el mismo proceso, y un `VACUUM` completo ahí tumbaría la recolección en vivo por el tiempo que
tarde). Por eso es un paso manual, de una sola vez, coordinado con una ventana de mantenimiento.
