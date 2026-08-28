# OSRS GE — Predictor de Precios

Pipeline en Python que recolecta precios de compra/venta del **Grand Exchange** de Old School
RuneScape desde la API pública de la OSRS Wiki, los almacena en SQLite, calcula un screener de
oportunidades de flip y entrena modelos de ML (XGBoost) que predicen hacia dónde va el precio
de cada ítem — con el objetivo de ayudar a decidir cuándo comprar y cuándo vender. Corre
pensado para 24/7, manda alertas de oportunidades por Telegram y expone un dashboard de
monitoreo en Streamlit; la futura app web para el usuario final todavía no existe.

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
- **Entrena dos modelos** sobre los ítems más líquidos: un regresor (XGBoost) que predice el
  precio del siguiente período — en dos cadencias, una horaria (señal rápida) y otra diaria
  (referencia de calidad estable) — y un clasificador direccional de 3 clases (baja/estable/
  sube) que le gana en accuracy direccional al regresor. Ambos se validan con un
  **replay walk-forward** sobre el historial y un **backtest** que simula la estrategia de flip
  con precios reales, comparado contra reglas triviales de referencia (baseline) y contra
  comprar a ciegas.
- **Manda alertas** de las mejores oportunidades del screener por Telegram, con cooldown por
  ítem para no espamear.
- **Expone un dashboard** de monitoreo en Streamlit (solo lectura): screener filtrable,
  calidad del modelo en el tiempo, predicción vs. realidad, pronóstico a futuro y la señal del
  clasificador direccional.
- **Tests unitarios** (pytest) de las funciones puras del pipeline (impuesto/margen,
  agrupamiento de huecos, checkpoints del replay, umbral del clasificador).

## Qué falta / en desarrollo

- **Aplicación web** para el usuario final — hoy solo existe el dashboard interno de
  monitoreo, de solo lectura.
- El clasificador direccional todavía no alimenta `alertas.py` (solo el dashboard); las
  alertas hoy se basan en el screener + la señal informativa del regresor.

## Estructura del proyecto

| Archivo | Rol |
|---|---|
| `osrs_ge_api.py` | Cliente de la API de precios (OSRS Wiki) |
| `recolector.py` | Recolección 24/7 + relleno de huecos + jobs programados |
| `base_de_datos.py` | Esquema y acceso a la base SQLite |
| `metricas.py` | Screener de mercado (margen, ROI, volatilidad, etc.) |
| `preprocesamiento.py` | Feature engineering para los modelos de ML |
| `entrenador.py` | Entrena el regresor y el clasificador direccional |
| `baseline.py` | Reglas triviales de referencia, para comparar contra el modelo real |
| `evaluacion.py` | Evaluación del modelo a horizontes de varios pasos |
| `replay_historico.py` | Reentrenamiento walk-forward sobre un rango del pasado |
| `backtest.py` | Simulación de la estrategia de flip sobre precios históricos reales |
| `prediccion.py` | Pronóstico hacia adelante con el modelo ya entrenado |
| `alertas.py` | Notificaciones de oportunidades por Telegram |
| `mantenimiento.py` | Retención, downsampling y vacuum de la base de datos |
| `dashboard.py` | Panel de monitoreo en Streamlit |
| `backfill_historico.py` | Relleno manual de huecos sin reiniciar el recolector |
| `tests/` | Tests unitarios de las funciones puras |
| `testing.ipynb` | Notebook de exploración |

## Cómo empezar

```bash
pip install -r requirements.txt
python recolector.py              # recolector 24/7 (recolección + reentrenamiento + alertas)
streamlit run dashboard.py        # dashboard de monitoreo, solo lectura
python -m pytest tests/           # tests unitarios
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
