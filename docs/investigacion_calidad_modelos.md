# Investigación de calidad de modelos: ensemble, calibración y dimensionamiento

Registro de la investigación hecha en la noche del 2026-08-30 sobre cómo combinar el
clasificador y el regresor para responder las tres preguntas reales del usuario: **qué** ítem
comprar, **en qué cantidad**, y **cuándo** venderlo. Ambiente de testeo aparte
(`busqueda_ensemble.py`, `busqueda_calibrada.py`) — nunca toca `modelos_config` del usuario ni
ningún `.pkl` productivo, salvo la creación explícita de `f2p10_100gp_clasif_u08` (ver más abajo).

## Fase 1 — ensemble con umbral fijo, casi sin señal (`busqueda_ensemble.py`)

Primer intento: clasificador (dirección) + regresor (magnitud) combinados por intersección —
compra solo si el clasificador predice "sube" **y** el regresor predice una magnitud por encima
de un umbral. Se probó sobre 3 ítems individuales (Adamantite bar, Runite ore, Dragon bones) con
umbrales fijos de 0.8/1.0/1.2%, los mismos para los tres ítems.

**Resultado:** de 27 combinaciones (3 ítems × 3 umbrales × 3 métodos), solo 2 operaron. El motivo,
verificado contra los datos reales (no asumido): en 336 checkpoints por ítem, el regresor nunca
predijo un movimiento mayor a 0.64% (Adamantite bar), 0.71% (Runite ore) o 0.79% (Dragon bones) —
por debajo de los tres umbrales probados. Un umbral fijo elegido a ojo en una corrida anterior
(sobre runas baratas y volátiles) no tiene ningún motivo para servir en ítems de escala y
volatilidad completamente distinta.

**Lección → hipótesis para la fase 2:** calibrar el umbral con la propia distribución histórica
de cada ítem, no con una constante compartida.

## Fase 2 — calibración por ítem + 8 ítems reales (`busqueda_calibrada.py`)

### Selección de ítems

En vez de elegir ítems "de alto valor" a ojo (que en la fase 1 resultaron ser, sin saberlo de
antemano, casi todos de muy baja volatilidad), esta vez se sacó `volatilidad_30d_pct` real de
`resumen_actual` para armar una muestra que cubre un espectro real:

| ítem | item_id | volatilidad_30d_pct | categoría |
|---|---|---|---|
| Runite ore | 451 | 0.40% | control (arrastrado de la fase 1) |
| Adamantite bar | 2361 | 0.93% | control (arrastrado de la fase 1) |
| Dragon bones | 536 | 1.97% | control (arrastrado de la fase 1) |
| Air orb | 573 | 7.75% | volatilidad media, runecrafting |
| Stamina potion(4) | 12625 | 7.11% | volatilidad media, consumibles |
| Kwuarm | 263 | 9.29% | volatilidad media, herblore |
| Bolt of canvas | 31475 | 40.11% | volatilidad alta, construcción |
| Basalt | 22603 | 106.18% | volatilidad extrema, crafting |

### Umbral calibrado por ítem

En vez de una constante, el umbral de cada ítem es el **percentil 80 de |log-retorno horario| de
ese mismo ítem**, calculado solo con datos ANTERIORES al inicio del backtest de 2 semanas (mismo
criterio walk-forward que el resto del proyecto — calibrar con datos que ya incluyan el rango de
evaluación sería fuga de información). Resultado: umbrales de 0.47% a 6.94% según el ítem, en vez
de 0.8% fijo para los ocho.

**Efecto medido:** con umbral fijo, 2 de 27 combinaciones operaron. Con umbral calibrado, **6 de
8 ítems generaron al menos una operación real** en alguno de los tres métodos (clasificador/
regresor/ensemble) — la calibración por ítem sí resuelve el problema de "casi sin señal" de la
fase 1.

### Resultado completo (umbral calibrado, capital inicial 10M gp, 2 semanas)

| ítem | umbral | clasificador | regresor | ensemble |
|---|---|---|---|---|
| Runite ore | 0.47% | −197.353 gp (3 trades, 33% win) | **−923.448 gp** (8 trades, 12% win) | **+2.976 gp** (1 trade, 100% win) |
| Adamantite bar | 0.71% | −54.370 gp (1 trade, 0% win) | 0 trades | 0 trades |
| Dragon bones | 0.83% | −91.230 gp (1 trade, 0% win) | 0 trades | 0 trades |
| Air orb | 1.59% | 0 trades | −90.896 gp (1 trade, 0% win) | 0 trades |
| Stamina potion(4) | 1.74% | 0 trades | **+198.000 gp** (1 trade, 100% win) | 0 trades |
| Kwuarm | 0.95% | −310.436 gp (6 trades, 17% win) | −98.145 gp (2 trades, 0% win) | 0 trades |
| Bolt of canvas | 6.94% | +424.219 gp (8 trades, 100% win) | +825.731 gp (16 trades, 94% win) | +414.352 gp (7 trades, 100% win) |
| Basalt ⚠️ | 5.14% | +11.283.101 gp | +10.022.369 gp | +6.167.837 gp | (ver advertencia abajo) |

**Lectura honesta:** el clasificador solo perdió plata en 4 de 8 ítems, el regresor solo en 3 de
8 — ninguno de los dos es confiablemente rentable por sí solo con esta muestra chica (8 ítems, 2
semanas). El **ensemble** (intersección) es notablemente más conservador: en Runite ore convirtió
un regresor que perdía −9.2% en una sola operación limpia con 100% de acierto — el filtro de
"ambos tienen que coincidir" funciona como se esperaba, a costa de operar mucho menos.

## Dimensionamiento por confianza (meta-labeling)

`entrenador.entrenar_clasificador_direccional` ahora expone `prob_sube` (probabilidad softmax de
la clase "sube", antes se descartaba y solo se guardaba el argmax). Se probó escalar el tamaño de
la posición según esa confianza (`FRACCION_PARTICIPACION * (0.5 + prob_sube)`) en vez de arriesgar
siempre la misma fracción fija — la idea de **meta-labeling** de Marcos López de Prado (*Advances
in Financial Machine Learning*, 2018): un modelo primario decide la dirección, un segundo score de
confianza decide cuánto apostar.

| ítem | fijo | por confianza | diferencia |
|---|---|---|---|
| Runite ore | −197.353 gp | −197.353 gp | sin cambio |
| Kwuarm | −310.436 gp | −306.846 gp | +3.590 gp (marginal) |
| Bolt of canvas | +424.219 gp | **+461.699 gp** | **+37.480 gp (+8.8%)** |

**Lectura:** el dimensionamiento por confianza no arregla una señal mala (Runite ore, Kwuarm
casi no cambian), pero **amplifica una señal que ya es buena** (Bolt of canvas, +8.8% de mejora
sobre la ganancia ya positiva) — consistente con la teoría: no reemplaza al modelo primario, lo
complementa.

## Salida por triple barrera

En vez de vender siempre exactamente 1 hora después (horizonte fijo de todas las corridas
anteriores), se probó la salida con 3 barreras (mismo López de Prado): take-profit (el propio
umbral calibrado del ítem), stop-loss (-50% del take-profit, riesgo/beneficio 2:1) y timeout (6
horas) — la que se cruce primero, contra precios reales ya ocurridos. No hace falta entrenar nada
nuevo para esto: son las mismas señales de arriba, solo cambia CUÁNDO se resuelve la venta.

| ítem | salida fija (1h) | triple barrera | diferencia |
|---|---|---|---|
| Kwuarm (clasificador) | −310.436 gp (17% win) | **−125.710 gp (33% win)** | **pérdida casi a la mitad** |
| Bolt of canvas (clasificador) | +424.219 gp | +429.653 gp | +5.434 gp (marginal) |
| Runite ore, Adamantite bar, Dragon bones, Air orb, Stamina potion(4) | — | — | sin cambio (mismo resultado que la salida fija) |

**Lectura:** el hallazgo más interesante de la noche. En Kwuarm, donde el clasificador acertaba
la dirección pero el horizonte fijo de 1h vendía en el peor momento, dejar que la salida se
resuelva contra el movimiento real (cortar la pérdida antes o esperar la ganancia un poco más)
casi duplicó el resultado sin cambiar en nada el modelo. Para el resto de los ítems no cambió
nada porque sus movimientos se resuelven casi siempre dentro de la primera hora (la salida fija
ya coincidía con la primera barrera cruzada). Esto sugiere que **la regla de salida importa tanto
como el modelo de entrada**, al menos para ítems donde el movimiento se desarrolla en más de un
período.

## ⚠️ Advertencia importante: Basalt y la volatilidad "de eventos raros"

Basalt dio +112% de ganancia en 2 semanas — una cifra que, en vez de reportarse como un hallazgo
positivo, se investigó antes de confiar en ella (mismo criterio de todo este proceso: verificar
antes de reportar). El resultado: **no es una oportunidad real, es un artefacto de iliquidez
puntual.**

- Salto máximo de una hora a la siguiente: **338.96%**.
- En ese mismo salto, `avg_low_price` (4372 gp) fue MAYOR que `avg_high_price` (1933 gp) —
  lógicamente imposible en un mercado sano — sobre apenas **7 unidades** de volumen.
- Un piso de volumen NO alcanza para filtrar esto: el volumen horario promedio de Basalt (8.277)
  es más alto que el de Bolt of canvas (1.267), que sí dio resultados creíbles.

Lo que sí separa limpio a los ítems "de verdad volátiles" de los "inflados por outliers" es la
razón entre el salto máximo histórico y el salto TÍPICO (mediana) del propio ítem:

| ítem | salto mediana | salto máximo | razón máx/mediana |
|---|---|---|---|
| Runite ore | 0.20% | 1.10% | 5.5x |
| Stamina potion(4) | 0.80% | 3.75% | 4.7x |
| Adamantite bar | 0.33% | 2.57% | 7.9x |
| Air orb | 0.58% | 3.56% | 6.1x |
| Dragon bones | 0.41% | 3.74% | 9.2x |
| Kwuarm | 0.45% | 5.94% | 13.2x |
| Bolt of canvas | 1.02% | 79.82% | **78.2x** ⚠️ cautela moderada |
| Basalt | 1.86% | 338.96% | **182.0x** ⚠️ ANÓMALO |

Los 6 ítems "limpios" caen entre 4.7x y 13.2x; Bolt of canvas y Basalt se separan claramente del
resto. Esto se formalizó en `busqueda_calibrada.diagnosticar_saltos_anomalos()` (validado contra
estos 8 ítems reales, reproduce exactamente esta tabla) — un diagnóstico, no un filtro automático
todavía: no está enganchado a ninguna selección de ítems en producción.

**Lección para la próxima selección de ítems (fase 1 y 2 de este documento, y cualquier corrida
futura):** `volatilidad_30d_pct` y el volumen agregado de 24h NO alcanzan para saber si un ítem es
genuinamente tradeable o si su volatilidad viene de un puñado de ticks con iliquidez puntual —
conviene revisar la razón máximo/mediana (o correr `diagnosticar_saltos_anomalos`) antes de
confiar en el backtest de un ítem nuevo. Bolt of canvas, con 78x, está en una zona gris: sus
resultados (+4-8%) son mucho más creíbles que los de Basalt pero deberían tomarse con cautela
moderada, no con la misma confianza que los 6 ítems "limpios".

## Cambios de código de esta fase

- **`entrenador.py`**: `entrenar_modelo_global` arreglado para soportar `precio_minimo`/
  `excluir_item_ids` (antes solo el clasificador los tenía — bug real, encontrado en la primera
  campaña de `busqueda_hiperparametros.py`, ver commit `5c71266`). `entrenar_clasificador_direccional`
  ahora expone `prob_sube` en el DataFrame de test devuelto.
- **`busqueda_hiperparametros.py`**: `backtest_capital_constante` gana un parámetro opcional
  `fraccion_fn` (dimensionamiento por confianza en vez de fracción fija) — de paso corrige un bug
  menor sin efecto observable hasta ahora: usaba la constante global `FRACCION_PARTICIPACION` en
  vez del parámetro `fraccion_participacion` de la función (ningún llamador existente lo
  sobreescribía, así que nunca importó en la práctica).
- **`busqueda_calibrada.py`** (nuevo): calibración de umbral por ítem, dimensionamiento por
  confianza, salida por triple barrera, y `diagnosticar_saltos_anomalos` — todo reutilizando el
  motor de `busqueda_hiperparametros.backtest_capital_constante` sin duplicarlo.
- Además, en la DB de producción se creó `f2p10_100gp_clasif_u08` (umbral 0.8% en vez de 0.5%,
  mismo universo F2P≥100gp que el productivo) — entrenándose cada hora junto al resto, sin
  reemplazarlo. Su walk-forward inicial ya corrió (2195/3687 checkpoints).

## Recomendaciones concretas para seguir

1. **La calibración por percentil funciona — vale la pena llevarla a producción.** Un candidato
   concreto: cuando el usuario cree un modelo clasificador de un ítem puntual desde la app de
   escritorio, calcular y sugerir el umbral calibrado (percentil 80 de su propio historial) en vez
   de dejar el default fijo de 0.5%.
2. **Triple barrera es la mejora con más impacto medido de las tres** (Kwuarm: pérdida casi a la
   mitad, sin tocar el modelo) — el siguiente paso lógico es engancharla al backtest "real"
   (`backtest.py`) para that el usuario la vea junto al resto de las métricas, no solo en este
   ambiente de testeo.
3. **Filtrar por `diagnosticar_saltos_anomalos` (o una métrica equivalente) antes de confiar en
   cualquier resultado de backtest de un ítem nuevo** — un solo tick de iliquidez puede inflar
   tanto la volatilidad reportada como la ganancia backtesteada sin que sea una oportunidad real.
4. **Muestra todavía chica** (8 ítems, 2 semanas, 48 filas de backtest) para sacar una conclusión
   general sobre "el clasificador solo" vs "el regresor solo" vs "el ensemble" — antes de decidir
   cuál usar en producción, valdría repetir esto sobre una ventana más larga (ej. 90 días) y/o más
   ítems, ahora que la mecánica de calibración/dimensionamiento/salida ya está validada y es
   barata de correr (no requiere reentrenar para probar dimensionamiento o salida, solo para el
   umbral calibrado en sí).
5. **Combinar los tres hallazgos a la vez** (umbral calibrado + dimensionamiento por confianza +
   triple barrera, todos juntos sobre el mismo ítem) no se llegó a probar esta noche — cada uno se
   validó por separado contra la salida/dimensionamiento de siempre. Es el siguiente experimento
   natural.
