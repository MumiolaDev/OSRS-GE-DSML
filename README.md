# OSRS GE — Predictor de Precios

Pipeline en Python que recolecta precios de compra/venta del **Grand Exchange** de Old School
RuneScape desde la API pública de la OSRS Wiki, los almacena en SQLite, calcula un screener de
oportunidades de flip y entrena modelos de ML (XGBoost) que predicen el margen neto que va a
dejar cada ítem en la próxima hora — con el objetivo de ayudar a decidir qué comprar, cuándo
comprar y cuándo vender. Corre pensado para 24/7 (servicio de systemd en Linux), manda alertas
de oportunidades por Telegram, y tiene una app de escritorio en PySide6/Qt para el usuario
final más un dashboard de monitoreo en Streamlit para el desarrollador.

## Qué hace hoy

- **Recolecta** precios de todo el catálogo de ítems del juego (compra/venta instantánea y
  volumen) en 5m/1h/6h desde la API pública de la OSRS Wiki, alineado al reloj de pared. Al
  arrancar, rellena solo los huecos que hayan quedado de cortes anteriores — no hace falta
  correr nada a mano después de un corte.
- **Almacena** el histórico en SQLite (modo WAL, lecturas concurrentes con la recolección),
  con retención/purga/downsampling automáticos por tabla y liberación de espacio en disco
  (`incremental_vacuum`) para que no crezca sin límite.
- **Calcula un screener** de oportunidades de flip: margen y ROI (con el impuesto real del
  Grand Exchange aplicado), potencial de ganancia por ciclo de compra, % de cambio, volatilidad,
  percentil dentro del historial reciente y tendencia de volumen.
- **Entrena los modelos que el usuario define** (no hay ninguno por default), sobre los ítems y
  la ventana de historial que elija. El modelo predice el **margen neto ejecutable de la próxima
  hora**: cuánto deja comprar en la punta baja y vender en la alta, ya descontado el impuesto del
  Grand Exchange. Eso responde directamente *qué comprar* — la app ordena las oportunidades por
  esa columna. Se valida con un **replay walk-forward** sobre el historial y un **backtest** que
  simula la estrategia con precios reales, comparado contra reglas triviales de referencia
  (baseline), contra el screener sin modelo y contra comprar a ciegas. El backtest informa
  siempre **dos cotas de ejecución** — capturando el spread y cruzándolo — porque la diferencia
  entre ambas es mucho más grande que cualquier ventaja del modelo, y la verdad está en el medio.
- **Manda alertas** de las mejores oportunidades del screener por Telegram, con cooldown por
  ítem para no espamear.
- **Expone un dashboard** de monitoreo en Streamlit (solo lectura): screener filtrable,
  calidad del modelo en el tiempo, predicción vs. realidad, pronóstico a futuro y el ranking de
  qué comprar.
- **Corre desatendido**: se instala como servicio de usuario de systemd con un comando
  (`./deploy/instalar_servicio.sh`), se reinicia solo si crashea y avisa por notificación de
  escritorio si se da por vencido. Un candado de instancia única impide que el servicio y la
  app recolecten a la vez sobre la misma base.
- **Tests unitarios** (pytest) de las funciones puras del pipeline (impuesto/margen, ventanas
  por tiempo, agrupamiento de huecos, checkpoints del replay, umbral del clasificador,
  accuracy direccional, grilla temporal, features relativas, candado y veredicto de salud).

## Qué falta / en desarrollo

- **Aplicación web** para el usuario final — hoy hay una app de escritorio (PySide6/Qt) y el
  dashboard interno de monitoreo, de solo lectura.
- **Modelar la probabilidad de que las órdenes se completen.** Toda la ventaja medida del
  modelo asume que la compra en la punta baja y la venta en la alta se llenan; si hay que cruzar
  el spread, ninguna estrategia gana. Ya hay una primera medición con datos de 5 minutos (de los
  ítems que el modelo elige: 78% de las compras se llenan y 52% de los flips se completan dentro
  de la hora, 65% en dos; de lo que queda colgado, el 78% recupera el precio de venta dentro de
  6 horas), pero es una muestra chica y todavía no corre sola: falta convertirla en una
  calibración periódica que la app muestre.
- El margen predicho anota las alertas de Telegram pero todavía no las filtra: el filtro
  principal sigue siendo el screener.

## Estructura del proyecto

| Archivo | Rol |
|---|---|
| `osrs_ge_api.py` | Cliente de la API de precios (OSRS Wiki) |
| `recolector.py` | Recolección 24/7 + relleno de huecos + jobs programados |
| `base_de_datos.py` | Esquema y acceso a la base SQLite |
| `metricas.py` | Screener de mercado (margen, ROI, volatilidad, etc.) |
| `preprocesamiento.py` | Feature engineering para los modelos de ML |
| `entrenador.py` | Entrena los modelos (margen ejecutable, y regresor/clasificador de comparación) |
| `baseline.py` | Reglas triviales de referencia, para comparar contra el modelo real |
| `evaluacion.py` | Evaluación del modelo a horizontes de varios pasos |
| `replay_historico.py` | Reentrenamiento walk-forward sobre un rango del pasado |
| `backtest.py` | Simulación de la estrategia de flip sobre precios históricos reales |
| `prediccion.py` | Pronóstico hacia adelante con el modelo ya entrenado |
| `alertas.py` | Notificaciones de oportunidades por Telegram |
| `mantenimiento.py` | Retención, downsampling y vacuum de la base de datos |
| `bloqueo.py` | Candado de instancia única del recolector (un solo proceso por base) |
| `estado.py` | Estado de salud del pipeline en una pantalla (terminal, barra de estado, scripts) |
| `escritorio/` | App de escritorio (PySide6/Qt) para el usuario final |
| `deploy/` | Servicio de systemd, lanzador de escritorio e instalador (Linux) |
| `dashboard.py` | Panel de monitoreo en Streamlit |
| `backfill_historico.py` | Relleno manual de huecos sin reiniciar el recolector |
| `tests/` | Tests unitarios de las funciones puras |
| `testing.ipynb` | Notebook de exploración |

## Cómo empezar

```bash
pip install -r requirements.txt
python recolector.py              # recolector 24/7 (recolección + reentrenamiento + alertas)
python -m escritorio.main         # app de escritorio
python estado.py                  # ¿está sano el pipeline?
streamlit run dashboard.py        # dashboard de monitoreo, solo lectura
python -m pytest tests/           # tests unitarios
```

Para dejarlo corriendo permanentemente en Linux (arranca con la PC, se reinicia solo):

```bash
./deploy/instalar_servicio.sh
loginctl enable-linger $USER      # este paso pide autenticación, por eso va aparte
```

Ver `CLAUDE.md` para el detalle de arquitectura, cada módulo y sus decisiones de diseño, y
`docs/despliegue_24_7.md` para dejarlo corriendo como servicio (incluye cómo configurar las
alertas de Telegram).

## Fuente de datos

Los precios se obtienen de la [API pública de la OSRS Wiki](https://prices.runescape.wiki/api/v1/osrs),
que separa compra y venta instantánea por intervalo — justo lo que necesita este proyecto. Se
contrastó puntualmente contra el endpoint oficial de Jagex como control de veracidad (desviación
menor a 1%), pero esa fuente no se usa para recolectar: es más coarse (solo promedio diario) y no
está pensada para acceso programático sostenido.
