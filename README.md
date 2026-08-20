# OSRS GE — Análisis de Datos y ML

Proyecto de análisis de datos en Python sobre el **Grand Exchange** de Old School RuneScape:
extracción automatizada de precios de compra/venta, almacenamiento histórico, y cálculo de
métricas de mercado, con el objetivo final de servir esta información en una aplicación web.

## Qué hace hoy

- **Recolecta** precios de todo el catálogo de ítems del juego (compra instantánea, venta
  instantánea y volumen) en intervalos de 1h y 6h, desde la API pública de la OSRS Wiki.
- **Almacena** el histórico en una base de datos SQLite, con un catálogo de ítems (nombre,
  si es members, límite de compra) para poder interpretar los datos.
- **Calcula métricas de mercado** listas para mostrar: margen y ROI de flip (con el impuesto
  real del Grand Exchange aplicado), potencial de ganancia por ciclo de compra, % de cambio,
  volatilidad, percentil dentro del historial reciente y tendencia de volumen — pensadas para
  un futuro "screener" ordenable en la web app.

## Qué falta / en desarrollo

- **Entrenamiento de modelo predictivo**: el pipeline de features (`preprocesamiento.py`)
  está listo, pero el entrenamiento (`entrenador.py`) todavía no está activo.
- **Aplicación web**: por ahora los datos se exploran vía notebook o consultas puntuales; la
  web app que consuma esto (screener + gráficos por ítem) es el siguiente objetivo grande.

## Estructura del proyecto

| Archivo | Rol |
|---|---|
| `osrs_ge_api.py` | Cliente de la API de precios (OSRS Wiki) |
| `recolector.py` | Recolección programada + backfill histórico |
| `base_de_datos.py` | Esquema y acceso a la base SQLite |
| `metricas.py` | Métricas de mercado para visualización (margen, ROI, volatilidad, etc.) |
| `preprocesamiento.py` | Feature engineering para el futuro modelo de ML |
| `entrenador.py` | Entrenamiento del modelo predictivo (en desarrollo) |
| `testing.ipynb` | Notebook de exploración |

## Cómo empezar

```bash
pip install -r requirements.txt
python recolector.py    # recolector en vivo (todo el catálogo, 1h y 6h)
python metricas.py      # recalcula el resumen de métricas sobre lo recolectado
```

## Fuente de datos

Los precios se obtienen de la [API pública de la OSRS Wiki](https://prices.runescape.wiki/api/v1/osrs),
que separa compra y venta instantánea por intervalo horario — justo lo que necesita este
proyecto. Se contrastó puntualmente contra el endpoint oficial de Jagex como control de
veracidad (desviación menor a 1%), pero esa fuente no se usa para recolectar: es más coarse
(solo promedio diario) y no está pensada para acceso programático sostenido.
