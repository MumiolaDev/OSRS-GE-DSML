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

- `logs\recolector.log` debe tener líneas nuevas cada ~5-60 minutos (recolección), una entrada
  de "Job horario" cada hora (:05) y una de "Job diario" una vez al día (03:00 UTC).
- Consultar la DB (`sqlite3 data\osrs_ge.db "SELECT MAX(timestamp) FROM precios_1h;"`) y
  confirmar que el timestamp más reciente está a menos de un par de horas de ahora.
- Reiniciar la PC una vez a propósito y confirmar que el proceso/servicio volvió a levantar
  solo, antes de confiar en que va a sobrevivir un corte de luz real.
- Los horarios programados (`recolector.py`, ver `_programar_cada_n_minutos_alineado` y el
  bloque `schedule.every(...)` en `__main__`) están pensados en **UTC** explícitamente (segundo
  argumento `"UTC"` en cada `.at(...)`) — no depende de la zona horaria del sistema operativo,
  así que no hace falta configurar el reloj del servidor en UTC para que la alineación sea
  correcta.

## Variables de entorno para las alertas de Telegram

`alertas.py` (enganchado al final del job horario) manda un mensaje agrupado con las mejores
oportunidades del screener cuando `TELEGRAM_BOT_TOKEN`/`TELEGRAM_CHAT_ID` están seteadas como
variables de entorno del proceso — si no lo están, solo loguea un warning y sigue sin romper
nada más.

1. Crear el bot: hablarle a [@BotFather](https://t.me/BotFather) en Telegram, mandar `/newbot`,
   seguir los pasos y guardar el `TOKEN` que entrega (algo como
   `123456789:AAExampleTokenNoEsReal`).
2. Mandarle cualquier mensaje al bot recién creado (buscarlo por el username que le pusiste) —
   Telegram necesita esa primera interacción para poder resolver tu `chat_id` después.
3. Obtener el `chat_id`: abrir en el navegador
   `https://api.telegram.org/bot<TOKEN>/getUpdates` (reemplazando `<TOKEN>`) y buscar el campo
   `"chat":{"id": ...}` en la respuesta JSON — ese número es el `chat_id`.
4. Setear las dos variables de entorno **del proceso que corre `recolector.py`**, nunca
   hardcodeadas en el código ni commiteadas:
   - Servicio NSSM: `nssm set OSRSRecolector AppEnvironmentExtra TELEGRAM_BOT_TOKEN=... TELEGRAM_CHAT_ID=...`
     (o editar la clave de registro del servicio directamente si `AppEnvironmentExtra` no está
     disponible en la versión instalada de NSSM) y reiniciar el servicio.
   - Tarea de Task Scheduler: no hay forma nativa de setear variables de entorno solo para la
     tarea — definirlas como variables de entorno de **usuario** (`setx TELEGRAM_BOT_TOKEN "..."`,
     `setx TELEGRAM_CHAT_ID "..."`) antes de crear/reiniciar la tarea.
5. Probar en aislado antes de confiar en que las alertas en vivo van a llegar:
   `python alertas.py` (corre `evaluar_alertas` una vez, sobre el estado actual de
   `resumen_actual`) y confirmar que el mensaje llega al chat de Telegram.
