# OSRS GE Predictor

Pipeline de datos y modelos de machine learning para detectar oportunidades de *flipping* en el
Grand Exchange de [Old School RuneScape](https://oldschool.runescape.com/).

![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Tests](https://img.shields.io/badge/tests-164-brightgreen)
![Platform](https://img.shields.io/badge/platform-Linux%20%7C%20Windows-lightgrey)

En el Grand Exchange cada ítem tiene dos precios: uno al que la gente vende y otro, más alto, al
que la gente compra. Comprar en el primero y vender en el segundo deja la diferencia, menos el
2% de impuesto que cobra el juego. Con más de 4.000 ítems y precios que cambian cada pocos
minutos, la pregunta difícil no es *cómo* operar sino *qué* ítem y *cuándo*.

Este proyecto recolecta el historial completo de precios y volúmenes, calcula un screener de
mercado en tiempo real, y entrena modelos XGBoost que predicen el **margen neto ejecutable** de
cada ítem para el período siguiente.

## Características

- **Recolección continua** desde la API pública de la OSRS Wiki en tres intervalos (5 min, 1 h y
  6 h), con reintentos, backoff y relleno automático de los huecos que hayan quedado mientras el
  proceso estuvo caído.
- **Almacenamiento en SQLite** con esquema versionado y migraciones idempotentes, escrituras
  idempotentes por clave primaria compuesta, y retención + compactación automáticas.
- **Screener de mercado**: margen neto (impuesto descontado), ROI, volumen de 24 h, volatilidad
  y tendencia para el catálogo completo, con filtros de liquidez y antigüedad del dato.
- **Modelos de predicción** (XGBoost) sobre features exclusivamente relativas, entrenados y
  reentrenados de forma programada. El usuario define sus propios modelos desde la app: el
  proyecto no incluye ninguno preentrenado.
- **Validación walk-forward**: reentrenamiento histórico en los mismos instantes en que habría
  ocurrido en producción, usando solo los datos disponibles hasta cada punto, con baselines
  triviales y aleatorios como control.
- **Backtesting** con supuestos de ejecución explícitos (decisión en el cierre del período,
  compra en el siguiente, valuación en las dos puntas del spread).
- **App de escritorio** en PySide6 y **dashboard** de monitoreo en Streamlit.
- **Alertas por Telegram** con cooldown por ítem.
- **Despliegue 24/7** como servicio de usuario de systemd, con candado de instancia única y un
  chequeo de salud independiente del estado del proceso.

## Cómo funciona

```
          API OSRS Wiki
                │
         osrs_ge_api.py          cliente HTTP con timeout, reintentos y backoff
                │
         recolector.py           scheduler alineado al reloj UTC (:05 / 03:00 / dom 04:00)
                │
       base_de_datos.py          SQLite: series de precios, métricas, predicciones
                │
   ┌────────────┼──────────────┬────────────────┬──────────────┐
   │            │              │                │              │
metricas.py  preprocesamiento  replay_historico  alertas.py  mantenimiento.py
(screener)        │            (walk-forward)   (Telegram)    (retención)
                  │
            entrenador.py ──→ prediccion.py ──→ escritorio/ · dashboard.py
```

Tres tareas programadas cubren el ciclo de vida: una **horaria** que refresca el screener,
reentrena los modelos de cadencia horaria y evalúa alertas; una **diaria** que hace lo mismo
para la cadencia diaria y calcula métricas por horizonte; y una **semanal** que aplica la
política de retención, con catch-up si la ventana se perdió.

## Instalación

**Requisitos:** Python 3.11 o superior (desarrollado sobre 3.14) y, para la app de escritorio,
un entorno gráfico con soporte de Qt (probado en Wayland y X11).

```bash
git clone https://github.com/MumiolaDev/OSRS-GE-mobile.git
cd OSRS-GE-mobile
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

La base de datos se crea sola en `data/osrs_ge.db` la primera vez que se ejecuta cualquier
componente.

## Uso

```bash
python recolector.py            # recolección continua y tareas programadas
python -m escritorio.main       # app de escritorio: oportunidades y gestión de modelos
python estado.py                # salud del pipeline (--json, --breve)
streamlit run dashboard.py      # dashboard de monitoreo (solo lectura)
python -m pytest tests/         # suite de tests
```

El flujo habitual es dejar `recolector.py` corriendo y trabajar desde la app. Con el recolector
detenido, la app puede levantarlo en un hilo propio; si detecta que ya hay uno activo en otro
proceso, pasa a modo monitor en lugar de duplicar las descargas.

Los módulos del pipeline también se pueden ejecutar de forma independiente para experimentar:
`metricas.py` (screener), `entrenador.py`, `baseline.py`, `replay_historico.py` (walk-forward),
`backtest.py`, `alertas.py`, `mantenimiento.py` y `backfill_historico.py`.

## Configuración

La configuración se resuelve con la prioridad **variable de entorno → `config.json` → default**,
de modo que un despliegue de servidor pueda usar el entorno y un usuario de escritorio pueda
editar los mismos valores desde la interfaz gráfica.

| Clave (`config.json`) | Variable de entorno | Descripción |
|---|---|---|
| `telegram_bot_token` | `TELEGRAM_BOT_TOKEN` | Token del bot que envía las alertas |
| `telegram_chat_id` | `TELEGRAM_CHAT_ID` | Chat de destino de las alertas |
| `db_path` | — | Ruta alternativa de la base de datos |

Sin credenciales de Telegram el pipeline funciona igual: las alertas se omiten con un warning en
el log. `config.json` está en `.gitignore` y nunca debe commitearse.

## Ejecución continua

En Linux, `deploy/` contiene las plantillas para dejarlo corriendo como servicio de usuario de
systemd (arranca con la máquina, se reinicia ante fallos y notifica si muere):

```bash
./deploy/instalar_servicio.sh
loginctl enable-linger $USER    # requiere autenticación, por eso va en un paso aparte
```

El instalador es idempotente y también registra un lanzador `.desktop` para la app. El detalle
completo, incluida la alternativa para Windows, está en [`docs/despliegue_24_7.md`](docs/despliegue_24_7.md).

## Metodología y resultados

El objetivo de predicción es el **margen neto ejecutable del período siguiente**, no el precio.
La diferencia es estructural: cuando llega el primer período en el que efectivamente se puede
comprar, el movimiento que un regresor de precio anticipó ya ocurrió y hay que pagarlo.

Comparación walk-forward sobre 720 horas de decisión y 80 ítems, midiendo el retorno por
operación de cada criterio de selección:

| Criterio de selección | Retorno por operación |
|---|---|
| Modelo de margen ejecutable | **+4,71 %** |
| Screener sin modelo | +2,27 % |
| Selección aleatoria | +0,25 % |
| Regresor de precio | −0,38 % |

La ventaja del modelo no está en pronosticar mejor el futuro, sino en bajar el listón de
ejecución: exige que las dos órdenes se completen el 60,6 % de las veces para no perder capital,
frente al 93,6 % que exige comprar sin criterio.

Otras decisiones de diseño sostenidas por medición:

- **Features relativas, no niveles.** Un árbol de decisión no puede restar dos features, así que
  con lags en nivel el modelo termina memorizando rangos de precio por ítem. Sobre 24 ítems y
  365 horas: 54,9 % de accuracy direccional con features en nivel contra 68,8 % con relativas.
- **Reindexado a grilla temporal antes de cualquier lag.** Los huecos del histórico son reales y
  frecuentes (28 % de las horas en un ítem líquido, 96 % en uno raro); sin reindexar, `shift(1)`
  significa "la fila anterior que exista" y el target deja de representar un período.
- **Accuracy direccional acotada.** Se mide solo donde el precio se movió por encima de un
  umbral y el modelo se comprometió con una dirección, y siempre acompañada del tamaño de la
  muestra evaluada para poder calcular el intervalo de confianza.

## Limitaciones

- **La ejecución no está validada de punta a punta.** Que el margen exista no garantiza que las
  dos órdenes se llenen. Con datos de 5 minutos se midió que, de los ítems seleccionados, el
  78 % de las compras se completa y el 52 % de los flips se cierra dentro de la hora (65 % en
  dos). Es una muestra chica y el proxy es optimista: que un bucket haya tocado un precio no
  prueba que una orden concreta se haya ejecutado.
- **El backtest asume capital ilimitado**, por lo que el beneficio total es una cota superior.
- **Sin ejecución automática.** El proyecto informa decisiones; no opera. La automatización de
  operaciones dentro del juego viola los términos de servicio de Jagex.
- El despliegue como servicio está probado en Linux; en Windows se documenta pero no se
  mantiene con la misma profundidad.

## Tests

164 tests con `pytest`, sobre funciones puras y bases de datos temporales. La suite nunca llama
a la API real ni toca la base de producción.

```bash
python -m pytest tests/
```

## Estructura del proyecto

| Ruta | Rol |
|---|---|
| `osrs_ge_api.py` | Cliente de la API de precios de la OSRS Wiki |
| `recolector.py` | Recolección continua, relleno de huecos y tareas programadas |
| `base_de_datos.py` | Esquema, migraciones y acceso a SQLite |
| `metricas.py` | Screener de mercado (margen, ROI, volatilidad, liquidez) |
| `preprocesamiento.py` | Feature engineering y construcción de targets |
| `entrenador.py` | Entrenamiento y evaluación de modelos |
| `prediccion.py` | Pronóstico hacia adelante con el modelo guardado |
| `replay_historico.py` | Reentrenamiento walk-forward sobre el histórico |
| `backtest.py` | Simulación de la estrategia con precios reales |
| `baseline.py` | Reglas triviales de referencia |
| `evaluacion.py` | Calidad del modelo a distintos horizontes |
| `alertas.py` | Notificaciones por Telegram |
| `mantenimiento.py` | Retención, archivado y compactación |
| `bloqueo.py` | Candado de instancia única sobre la base de datos |
| `estado.py` | Chequeo de salud del pipeline |
| `configuracion.py` | Resolución de configuración local |
| `escritorio/` | App de escritorio (PySide6) |
| `dashboard.py` | Dashboard de monitoreo (Streamlit) |
| `deploy/` | Unidades de systemd, lanzador e instalador |
| `docs/` | Despliegue 24/7 y proyecciones de escalabilidad |
| `tests/` | Suite de tests |

## Desarrollo asistido por IA

Este proyecto se desarrolló con [Claude Code](https://claude.com/claude-code) como par de
programación. `CLAUDE.md` es el archivo de contexto que el agente lee al inicio de cada sesión:
documenta la arquitectura, las invariantes del sistema y el razonamiento detrás de las
decisiones no obvias, y se mantiene actualizado junto con el código. Es también una lectura útil
para cualquier persona que quiera entender el proyecto en profundidad.

## Licencia

[MIT](LICENSE).

## Aviso

Proyecto independiente, sin afiliación ni respaldo de Jagex Ltd. Old School RuneScape es una
marca registrada de Jagex Ltd. Los datos provienen de la
[API pública de precios de la OSRS Wiki](https://prices.runescape.wiki/api/v1/osrs), usada
respetando sus límites de uso.
