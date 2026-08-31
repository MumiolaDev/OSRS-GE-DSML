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

## Las tres mejoras combinadas (`busqueda_combinada.py`)

Dimensionamiento por confianza y salida por triple barrera se probaron cada uno POR SEPARADO
contra el comportamiento de siempre. Quedaba la pregunta obvia: ¿se combinan o se pisan entre sí?
Se probaron las dos juntas (más el umbral calibrado, ya presente en ambas) sobre los dos ítems
donde alguna mejora individual había dado resultado — reentrenando el clasificador walk-forward
una vez más por ítem, con el mismo umbral calibrado ya encontrado, para tener `prob_sube` en
memoria (no se persiste en la DB entre corridas).

| ítem | siempre (fijo/fijo) | solo confianza | solo triple barrera | **las dos combinadas** |
|---|---|---|---|---|
| Kwuarm | −310.436 gp (−3,10%) | −306.846 gp (−3,07%) | −125.710 gp (−1,26%) | **−101.564 gp (−1,02%)** |
| Bolt of canvas | +424.219 gp (+4,24%) | +461.699 gp (+4,62%) | +429.653 gp (+4,30%) | **+466.981 gp (+4,67%)** |

**Resultado limpio: se combinan, no se pisan.** En los dos ítems, la variante con las dos mejoras
juntas superó a cada una por separado y al comportamiento de siempre — en Kwuarm, la pérdida bajó
otro 19% más allá de lo que ya lograba la triple barrera sola; en Bolt of canvas, la ganancia
subió otro 1,2% más allá de lo que lograba cada mejora individual. No hay evidencia de que una
mejora cancele o interfiera con la otra en esta muestra (2 ítems) — dimensionar por confianza y
elegir cuándo salir son decisiones independientes entre sí (una es "cuánto", la otra es "cuándo"),
así que tiene sentido que sumen en vez de competir.

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

## Fase 3 — ¿el método es mejor que el azar, con seguridad aceptable? (`busqueda_significancia.py`)

Todo lo anterior tenía dos debilidades reales: los 8 ítems se eligieron a mano (sin
garantía de que representen el universo real), y nunca se hizo un test de significancia
formal — solo se comparó `win_rate`/ganancia sin ninguna referencia de azar. Esta fase
ataca las dos cosas a la vez, pedido explícito del usuario tras ver los resultados de la
fase 2 y sentir que "no se está logrando el objetivo del proyecto":

1. **Ventana 90 días, solo `precios_1h`** (antes: 14 días) — 2.159 checkpoints horarios
   walk-forward, 6.4x más que la fase 2.
2. **Universo NO elegido a mano**: exactamente la configuración YA en producción
   (`f2p10_100gp_clasif` — 10 ítems F2P más líquidos, precio≥100gp, Steel bar excluido),
   re-derivada en cada checkpoint. Esto prueba el método tal como el usuario ya lo está
   usando, no una selección optimista.
3. **Test de significancia por permutación (Monte Carlo, 500 sorteos)**: en cada
   checkpoint, un "jugador al azar" compra la MISMA cantidad de ítems que el modelo
   predijo "sube" en ese momento, sorteados entre los MISMOS candidatos líquidos de ese
   checkpoint — así una diferencia de resultado solo puede deberse a QUÉ eligió el
   modelo, no a cuántas veces operó ni sobre qué universo. Se prefirió sobre un test
   binomial contra 50% porque acá una "victoria" no vale lo mismo en gp para todos los
   trades (montos de posición muy distintos por ítem) — y porque, como se ve abajo, el
   propio azar no gana el 50% de las veces en este juego.

### Resultado

90 días, 5.639 trades del modelo real:

| métrica | modelo real | azar (media de 500 sorteos) |
|---|---|---|
| ganancia total | **+11.892.000 gp** | −4.839.155 gp |
| win rate | 38.6% (IC95%: 37.4%–39.9%) | 35.5% |
| p-valor (permutación, ganancia) | **0.002** | — |
| p-valor (permutación, win rate) | **0.002** | — |
| p-valor (binomial vs. 50%) | 1.0 (no aplica — ver abajo) | — |

**El método SÍ es mejor que el azar, con una seguridad muy alta** (p≈0.002, muy por
debajo del 0.05 convencional; en 500 sorteos, casi ninguno igualó o superó el resultado
real). El dato interesante es que el propio azar, en este universo y con esta mecánica
de apuesta, **pierde plata en promedio** (−4.84M gp) — comprar F2P líquidos al azar y
vender 1h después ya es un juego de valor esperado negativo (el impuesto GE y el spread
se lo comen), así que el binomial test contra 50% da p=1.0 y es literalmente la pregunta
equivocada acá: el modelo no necesita ganar más de la mitad de las veces, necesita ganar
más seguido y/o más grande que elegir al azar en el mismo juego negativo — y eso sí lo
logra, de forma medida, no solo intuida.

### Pero la ganancia está muy concentrada

| ítem | ganancia | trades | win rate |
|---|---|---|---|
| Cosmic rune | **+10.872.000 gp** | 564 | 64.9% |
| Death rune | +3.350.000 gp | 311 | 44.7% |
| Gold ore | +2.880.000 gp | 679 | 37.7% |
| Chaos rune | +2.412.000 gp | 284 | 47.9% |
| Law rune | +1.548.000 gp | 147 | 50.3% |
| Gold bar | +630.000 gp | 104 | 51.0% |
| Nature rune | +270.000 gp | 539 | 37.3% |
| Yew logs | −576.000 gp | 894 | 34.5% |
| Mithril ore | −1.170.000 gp | 769 | 36.2% |
| Adamantite ore | −4.086.000 gp | 521 | 34.2% |
| Coal | **−4.238.000 gp** | 827 | **22.9%** |

**Cosmic rune solo explica el 91% de la ganancia total** (10.87M de 11.89M gp). Sin ese
ítem, el resto del universo (los otros 10 ítems combinados) da apenas **+1.020.000 gp en
90 días sobre 5.075 trades** (win rate 35.7%, casi idéntico al 35.5% del azar) — una
señal mucho más débil, dentro de lo que podría ser ruido para varios de esos ítems
individualmente.

Verificación importante: el edge de Cosmic rune **no es un evento puntual** — se
mantiene estable en las dos mitades de la ventana (primeros 45 días: 317 trades,
+5.904.000 gp, 62.8% win rate; últimos 45 días: 247 trades, +4.968.000 gp, 67.6% win
rate). Es una señal real y persistente en ese ítem específico, no un artefacto de un
período afortunado.

**Coal es un problema aparte**: 22.9% de win rate en 827 trades es *peor* que el 35.5%
promedio del azar — el modelo no solo no encuentra señal ahí, activamente se equivoca
más seguido que adivinar al azar en ese mismo universo. Adamantite ore (34.2%, −4.09M
gp) es un caso más débil de lo mismo. Mismo patrón que ya llevó a excluir Steel bar
(2353) del universo productivo.

### Lectura honesta para la pregunta del usuario

*"¿Sirve este método para saber qué y cuándo comprar/vender, con una seguridad
aceptable, mejor que el azar?"* — **Sí, en conjunto y con una seguridad estadística
alta (p≈0.002)**, pero esa respuesta agregada esconde que la mayor parte del valor viene
de una señal fuerte y estable en un solo ítem (Cosmic rune), mientras que dos ítems
(Coal, Adamantite ore) restan valor de forma consistente y el resto del universo aporta
una señal débil, no claramente distinguible del azar por sí sola. El método no es
"ruido total" (la fase 2 con 8 ítems y 2 semanas no alcanzaba a distinguir esto), pero
tampoco es una señal pareja y universal sobre cualquier ítem líquido F2P — es más
preciso describirlo como "una señal real mezclada con varios ítems sin edge claro".

### Recomendaciones concretas

1. **Excluir Coal (453) y Adamantite ore (449) del universo productivo**, mismo criterio
   que ya excluyó Steel bar — están restando plata de forma consistente, no por mala
   suerte puntual.
2. **Investigar qué hace distinto a Cosmic rune** antes de asumir que el resto del
   universo puede llegar a ese nivel con más datos — puede ser un patrón de demanda
   específico (consumo constante en quests/altares) que no se repite en runas de combate
   como Nature/Chaos.
3. **Repetir este mismo test de significancia (permutación) sobre un universo más amplio
   que el F2P de 10 ítems** — si hay más "Cosmic runes" (ítems con edge fuerte y estable)
   fuera de ese universo, vale la pena encontrarlos con el mismo método ahora que ya está
   armado y validado, en vez de asumir que el edge encontrado es el único que existe.
4. Esta fase reemplaza la lectura de la fase 2 como referencia principal para decidir
   sobre el universo productivo — no por invalidar la fase 2 (los hallazgos de
   calibración/triple-barrera siguen siendo válidos como mejoras de ejecución), sino
   porque una muestra de 90 días/5.639 trades con test de significancia es una base mucho
   más sólida que 8 ítems/2 semanas sin él para decidir QUÉ ítems mantener en producción.

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
5. ~~Combinar los tres hallazgos a la vez~~ — **hecho** (`busqueda_combinada.py`, ver sección
   arriba): en Kwuarm y Bolt of canvas, dimensionamiento por confianza + triple barrera juntos
   superaron a cada mejora por separado y al comportamiento de siempre — se combinan sin pisarse.
   Sigue pendiente extenderlo a los 8 ítems completos (acá solo se probó en los 2 donde ya había
   señal de que alguna mejora individual servía).
