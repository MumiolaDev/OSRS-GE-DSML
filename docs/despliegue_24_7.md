# Dejar el recolector corriendo 24/7

`recolector.py` ya corre indefinidamente (loop de `schedule`), pero como proceso normal muere
al cerrar la terminal y no vuelve solo si la PC se reinicia o si el proceso crashea. Este
documento cubre cómo dejarlo como servicio en **Linux** (la forma soportada hoy) y, más abajo,
en Windows.

Un solo recolector a la vez: el proceso toma un candado `flock` sobre `data/osrs_ge.db.lock`
(ver `bloqueo.py`) y un segundo recolector sobre la misma DB no arranca. Eso es lo que hace
seguro tener el servicio andando y abrir igual la app de escritorio — la app detecta el
candado, desactiva su botón "Iniciar recolector" y pasa a mostrar el estado del servicio en
vivo en vez de arrancar un segundo proceso que duplicaría las descargas y se pisaría al
reentrenar.

## Linux: servicio de usuario de systemd (recomendado)

```bash
./deploy/instalar_servicio.sh
```

Eso genera dos unidades en `~/.config/systemd/user/` a partir de las plantillas de `deploy/`
(resolviendo la ruta del proyecto y el intérprete del entorno virtual), instala un lanzador
`.desktop` para la app, y deja el servicio habilitado y corriendo. Es idempotente: correrlo de
nuevo regenera todo con las rutas actuales.

Es un servicio **de usuario** (`systemctl --user`) y no del sistema a propósito: el recolector
escribe en el home (`data/`, `models/`, `logs/`, `config.json`) y no necesita ningún privilegio.

Falta un paso que el script no puede hacer solo porque pide autenticación:

```bash
loginctl enable-linger $USER
```

Sin *lingering*, las unidades de usuario arrancan recién cuando iniciás sesión y se cortan
cuando la cerrás. Con lingering, el recolector arranca con la PC y sobrevive el logout — que
es lo que "24/7" quiere decir en una máquina de escritorio.

### El día a día

```bash
systemctl --user status osrs-recolector     # ¿está corriendo?
systemctl --user restart osrs-recolector    # después de tocar el código
systemctl --user stop osrs-recolector       # pararlo (ej. para un backfill grande a mano)
journalctl --user -u osrs-recolector -f     # el log en vivo
journalctl --user -u osrs-recolector --since today | grep -i error
python estado.py                            # ¿el pipeline está SANO? (ver abajo)
./deploy/instalar_servicio.sh --desinstalar  # sacar todo
```

El servicio se reinicia solo si crashea (`Restart=always`, 60s de espera). Si crashea 5 veces
en 5 minutos queda en estado `failed` y dispara una notificación de escritorio
(`osrs-recolector-fallo.service` vía `OnFailure=`) — el corte es a propósito: un error
permanente de arranque no debe convertirse en un loop pegándole a la API para siempre.

No hay dependencia de `network-online.target` (no existe en el manager de usuario): si la red
todavía no está lista al arrancar, la request falla, el cliente reintenta con backoff
(`osrs_ge_api.REINTENTOS`) y en el peor caso el servicio se reinicia y se pone al día solo con
el relleno de huecos.

### Ver el estado sin abrir nada

```bash
python estado.py            # resumen completo
python estado.py --breve    # una línea
python estado.py --json     # para la barra de estado
```

`estado.py` es de solo lectura y contesta la pregunta que importa, que **no** es "¿el proceso
está vivo?" sino "¿los datos están al día?". No son lo mismo: un recolector colgado sigue
apareciendo verde en `systemctl status` mientras hace horas que no inserta una fila. El
veredicto sale de la antigüedad del último dato horario; el estado del proceso es un dato más.

Sale con código 0 si está todo bien, 1 si hay algo que mirar y 2 si no pudo ni leer la DB, así
que sirve directo en un script.

### Waybar (Hyprland)

Módulo para tener el estado siempre a la vista, sin abrir nada. En `~/.config/waybar/config`:

```jsonc
"custom/osrs": {
    "exec": "/RUTA/AL/PROYECTO/osrs_env/bin/python /RUTA/AL/PROYECTO/estado.py --json",
    "return-type": "json",
    "interval": 300,
    "on-click": "gtk-launch osrs-ge",
    "on-click-right": "systemctl --user restart osrs-recolector"
}
```

(agregar `"custom/osrs"` a `modules-right`). Y en `~/.config/waybar/style.css`:

```css
#custom-osrs.ok      { color: #a6e3a1; }
#custom-osrs.atencion{ color: #f9e2af; }
#custom-osrs.error   { color: #f38ba8; }
```

Las clases salen del campo `class` del JSON, que es el veredicto de `estado.py`. El tooltip
trae el resumen completo. `interval: 300` (5 min) alcanza de sobra: el dato de referencia se
actualiza una vez por hora.

### Atajo de teclado para la app

El instalador deja `~/.local/share/applications/osrs-ge.desktop`, así que la app aparece en el
lanzador (rofi/wofi/fuzzel) como "OSRS GE Predictor". Para un atajo directo, en
`~/.config/hypr/hyprland.conf`:

```
bind = $mainMod, O, exec, gtk-launch osrs-ge
```

La app corre nativa en Wayland (PySide6 6.11 + `qt6-wayland`). Si alguna vez hiciera falta
forzar XWayland: `QT_QPA_PLATFORM=xcb python -m escritorio.main`.

## Windows

### Opción recomendada: servicio de Windows con NSSM

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

### Alternativa sin instalar nada: Tarea de Task Scheduler

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

- `python estado.py` debe decir "Datos al día" y mostrar el recolector como corriendo.
- El log debe tener líneas nuevas cada ~5-60 minutos (recolección), una entrada de "Job
  horario" cada hora (:05) y una de "Job diario" una vez al día (03:00 UTC). En Linux:
  `journalctl --user -u osrs-recolector -f`; el script además escribe siempre a
  `logs/recolector.log`.
- Consultar la DB (`sqlite3 data/osrs_ge.db "SELECT MAX(timestamp) FROM precios_1h;"`) y
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
   - Servicio de systemd (Linux): `mkdir -p ~/.config/osrs-ge` y escribir
     `~/.config/osrs-ge/env` con `TELEGRAM_BOT_TOKEN=...` y `TELEGRAM_CHAT_ID=...` (una por
     línea, sin comillas ni `export`) — la unidad ya lo lee vía `EnvironmentFile=`. Después
     `systemctl --user restart osrs-recolector`. Alternativa sin tocar archivos: cargarlas
     desde la pestaña Configuración de la app, que las guarda en `config.json` (ver
     `configuracion.py`, las variables de entorno ganan sobre el archivo).
   - Tarea de Task Scheduler: no hay forma nativa de setear variables de entorno solo para la
     tarea — definirlas como variables de entorno de **usuario** (`setx TELEGRAM_BOT_TOKEN "..."`,
     `setx TELEGRAM_CHAT_ID "..."`) antes de crear/reiniciar la tarea.
5. Probar en aislado antes de confiar en que las alertas en vivo van a llegar:
   `python alertas.py` (corre `evaluar_alertas` una vez, sobre el estado actual de
   `resumen_actual`) y confirmar que el mensaje llega al chat de Telegram.
