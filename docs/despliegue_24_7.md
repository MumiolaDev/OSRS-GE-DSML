# Desplegar `recolector.py` como proceso 24/7 en Windows

`recolector.py` ya corre indefinidamente (loop de `schedule`), pero como proceso normal muere
al cerrar la terminal y no se reinicia solo si la PC se reinicia o si el proceso crashea. Para
que sobreviva ambas cosas hay dos caminos. Ninguno de los dos pasos siguientes se ejecuta
automáticamente — son acciones sobre el sistema operativo que hay que correr a mano.

## Opción recomendada: servicio de Windows con NSSM

Más robusto: sobrevive reinicios de la PC **sin necesidad de tener una sesión de usuario
logueada**, y reinicia el proceso solo si crashea.

1. Descargar [NSSM](https://nssm.cc/download) (herramienta de terceros, ampliamente usada para
   correr scripts como servicio de Windows) y descomprimirlo en algún lugar del PATH, o usar la
   ruta completa al ejecutable.
2. Instalar el servicio (PowerShell como administrador):

   ```powershell
   nssm install OSRSRecolector "C:\ruta\a\python.exe" "Z:\Programación\Python\ClaudeTest\recolector.py"
   nssm set OSRSRecolector AppDirectory "Z:\Programación\Python\ClaudeTest"
   nssm set OSRSRecolector AppStdout "Z:\Programación\Python\ClaudeTest\logs\nssm_stdout.log"
   nssm set OSRSRecolector AppStderr "Z:\Programación\Python\ClaudeTest\logs\nssm_stderr.log"
   nssm set OSRSRecolector AppExit Default Restart
   ```
3. Arrancarlo: `nssm start OSRSRecolector`.
4. Verificar: `Get-Service OSRSRecolector` debe mostrar `Running`; revisar
   `logs\recolector.log` (el logging propio del script) para confirmar que está recolectando.
5. Para desinstalar: `nssm remove OSRSRecolector confirm`.

## Alternativa sin instalar nada: Tarea de Task Scheduler

Menos robusta (no arranca si la PC quedó reiniciada sin que el usuario haya iniciado sesión),
pero no requiere instalar software adicional.

1. Abrir el Programador de tareas (`taskschd.msc`) → "Crear tarea básica".
2. Desencadenador: "Al iniciar sesión".
3. Acción: iniciar un programa → `python.exe`, argumentos: ruta completa a `recolector.py`,
   "iniciar en": la carpeta del proyecto.
4. En la pestaña "Configuración" de la tarea: marcar "Si la tarea falla, reiniciar cada" (ej.
   1 minuto, hasta 3 intentos) para que se recupere de un crash.
5. Marcar "Ejecutar tanto si el usuario inició sesión como si no" en la pestaña "General" (pide
   la contraseña de la cuenta) para que no dependa de tener la sesión abierta.

## Verificar que quedó corriendo de verdad

Independiente de la opción elegida:

- `logs\recolector.log` debe tener líneas nuevas cada ~5-60 minutos (recolección) y una entrada
  de "Job diario" una vez al día.
- Consultar la DB (`sqlite3 data\osrs_ge.db "SELECT MAX(timestamp) FROM precios_1h;"`) y
  confirmar que el timestamp más reciente está a menos de un par de horas de ahora.
- Reiniciar la PC una vez a propósito y confirmar que el proceso/servicio volvió a levantar
  solo, antes de confiar en que va a sobrevivir un corte de luz real.
