# OSRS GE — Predictor de Precios

Programa que ayuda a decidir **qué comprar y cuándo** en el Grand Exchange de Old School
RuneScape.

## Qué hace, en simple

En el Grand Exchange cada ítem tiene dos precios: uno bajo, al que la gente vende, y uno alto,
al que la gente compra. Comprar abajo y vender arriba —un *flip*— deja la diferencia, menos el
2% de impuesto que cobra el juego al vender. El problema es cuál de los 4.000 ítems conviene y
en qué momento.

El programa hace cuatro cosas:

1. **Recolecta** los precios y volúmenes de todos los ítems del juego desde la API pública de la
   wiki de OSRS, cada 5 minutos, cada hora y cada 6 horas, y los guarda en una base SQLite local.
   Corre solo, todo el día, y se pone al día si estuvo apagado.
2. **Calcula el margen** que deja cada ítem ahora mismo, con el impuesto ya descontado, más ROI,
   volumen, volatilidad y tendencia. Eso es el *screener*: la foto del mercado en este momento.
3. **Predice** con un modelo de machine learning (XGBoost) el margen que va a dejar cada ítem en
   la **próxima hora**. Los modelos los crea el usuario, eligiendo qué ítems y cuánto historial;
   no viene ninguno de fábrica.
4. **Lo muestra** en una app de escritorio con las oportunidades ordenadas por margen predicho, y
   manda las mejores por Telegram.

## Por qué predice el margen y no el precio

Parece más natural predecir si el precio va a subir. Se probó, y no funciona: cuando llega el
primer momento en el que uno puede comprar, la suba ya ocurrió y hay que pagarla. Medido sobre
720 horas de decisión y 80 ítems, elegir qué comprar con un modelo de precio rendía **−0,38%**
por operación, peor que elegir al azar (+0,25%).

Predecir el margen ejecutable —lo que deja el flip completo, comprando abajo y vendiendo arriba,
ya descontado el impuesto— rindió **+4,71%**, contra +2,27% del screener solo. La ventaja no es
adivinar mejor el futuro: es que baja el listón de ejecución. Con el modelo, las dos órdenes
tienen que completarse el 60,6% de las veces para no perder plata; comprando a ciegas, el 93,6%.

## Qué tan bien funciona (y qué no se sabe todavía)

Todo lo anterior está validado con *walk-forward*: el modelo se reentrena en los mismos momentos
en que lo habría hecho en vivo y solo ve los datos que existían hasta ahí, comparado siempre
contra reglas triviales, contra el screener sin modelo y contra comprar al azar.

Lo que queda abierto es la **ejecución**: que el margen exista no garantiza que las dos órdenes
se llenen. Con datos de 5 minutos se midió que, de los ítems que el modelo elige, el 78% de las
compras se completan y el 52% de los flips se cierran dentro de la hora (65% en dos); de lo que
queda colgado, el 78% recupera el precio de venta dentro de las 6 horas siguientes. La
esperanza da positiva en todos los casos, pero es una muestra chica y la medición todavía no
corre sola.

## Estructura

| Archivo | Rol |
|---|---|
| `osrs_ge_api.py` | Cliente de la API de precios (OSRS Wiki) |
| `recolector.py` | Recolección 24/7 + relleno de huecos + tareas programadas |
| `backfill_historico.py` | Relleno manual de un rango histórico, sin reiniciar el recolector |
| `base_de_datos.py` | Esquema y acceso a la base SQLite |
| `metricas.py` | Screener de mercado (margen, ROI, volatilidad) |
| `preprocesamiento.py` | Feature engineering para los modelos |
| `entrenador.py` | Entrena los modelos y mide su calidad |
| `prediccion.py` | Pronóstico hacia adelante con el modelo entrenado |
| `replay_historico.py` | Reentrenamiento walk-forward sobre el pasado |
| `backtest.py` | Simulación de la estrategia con precios reales |
| `baseline.py` | Reglas triviales de referencia, para comparar |
| `evaluacion.py` | Calidad del modelo a horizontes de varios pasos |
| `alertas.py` | Avisos de oportunidades por Telegram |
| `mantenimiento.py` | Retención y compactación de la base |
| `bloqueo.py` | Candado: un solo recolector por base de datos |
| `estado.py` | Salud del pipeline en una pantalla |
| `configuracion.py` | Configuración local (token de Telegram, ruta de la base) |
| `escritorio/` | App de escritorio (PySide6/Qt) para el usuario final |
| `dashboard.py` | Panel de monitoreo en Streamlit, para el desarrollador |
| `deploy/` | Servicio de systemd, lanzador e instalador (Linux) |
| `tests/` | 170 tests unitarios (no tocan la API ni la base real) |

## Cómo empezar

```bash
pip install -r requirements.txt
python recolector.py          # empieza a recolectar
python -m escritorio.main     # la app: oportunidades y modelos
python estado.py              # ¿está todo al día?
```

Para dejarlo corriendo permanentemente en Linux (arranca con la PC y se reinicia solo):

```bash
./deploy/instalar_servicio.sh
loginctl enable-linger $USER   # este paso pide autenticación, por eso va aparte
```

El detalle de arquitectura y las decisiones de diseño están en `CLAUDE.md`; el despliegue 24/7 y
las alertas de Telegram, en `docs/despliegue_24_7.md`.

## Fuente de datos

Los precios salen de la [API pública de la OSRS Wiki](https://prices.runescape.wiki/api/v1/osrs),
que separa compra y venta instantánea por intervalo — justo lo que necesita este proyecto. Se
contrastó puntualmente contra el endpoint oficial de Jagex como control (desviación menor al 1%),
pero esa fuente no se usa: es más gruesa (solo promedio diario) y no está pensada para acceso
programático sostenido.
