# Decisiones sobre calidad del clasificador direccional F2P

Resumen de una investigación de una noche (2026-08-30) sobre si el clasificador
direccional productivo (`f2p10_100gp_clasif`, ver `entrenador.py`) predice qué y cuándo
comprar/vender con una confianza mejor que el azar, y qué hacer al respecto. Los scripts
de exploración (`busqueda_ensemble.py`, `busqueda_calibrada.py`, `busqueda_combinada.py`,
`busqueda_hiperparametros.py`, `busqueda_significancia.py`) ya se borraron una vez
extraídas las conclusiones — este documento es lo que queda: qué se aplicó a producción
y qué quedó como idea sin validar.

## Aplicado a producción

**`excluir_item_ids` de `f2p10_100gp_clasif` pasó de `[2353]` a `[2353, 449, 453]`**
(Steel bar, Adamantite ore, Coal) — ver `base_de_datos._sembrar_modelos_productivos` y el
comentario junto a `MODEL_NAME_CLASIF_F2P_100GP` en `entrenador.py`.

Motivo: un test de significancia por permutación (Monte Carlo, 500 sorteos) sobre 90 días
walk-forward / 5.639 trades — el universo EXACTO de producción, no ítems elegidos a mano,
resolución horaria — mostró que el conjunto gana significativamente más que un jugador al
azar con la misma mecánica de apuesta (**+11.89M gp vs. −4.84M gp promedio del azar,
p=0.002** tanto en ganancia como en win rate). Pero desglosado por ítem:

- **Coal (453)**: 22.9% win rate en 827 trades — peor que el 35.5% del propio azar. El
  modelo se equivoca ahí más seguido que adivinar al azar en el mismo universo.
- **Adamantite ore (449)**: 34.2% win rate, −4.09M gp — mismo patrón, más débil.
- **Cosmic rune (564)**: explica el 91% de la ganancia total (+10.87M gp, ~65% win rate),
  estable en las dos mitades de la ventana de 90 días — no es un evento puntual, es una
  señal real y persistente en ese ítem específico.
- El resto del universo (sin Cosmic rune) aporta apenas +1.02M gp/90 días — una señal
  débil, casi indistinguible del azar (35.7% win rate vs. 35.5% del azar).

Conclusión aplicada: sacar los dos ítems que restan plata de forma consistente (mismo
criterio que ya había excluido Steel bar). La pregunta abierta — por qué Cosmic rune
concentra casi toda la señal, y si hay más ítems así fuera del universo F2P de 10 — queda
sin investigar.

## Explorado pero NO aplicado a producción

Tres ideas de ejecución (no de selección de ítems) se probaron sobre una muestra mucho
más chica (8 ítems elegidos a mano, 2 semanas) y **sin ningún test de significancia** —
prometedoras, pero no validadas con el mismo rigor que la decisión de arriba:

- **Umbral calibrado por ítem** (percentil 80 de `|log-retorno|` histórico del propio
  ítem, en vez de un umbral fijo compartido) — resolvió el problema de "casi sin señal"
  de un umbral fijo, pero calibrar el umbral no es lo mismo que validar que la señal
  resultante sea rentable.
- **Dimensionamiento de posición por confianza** (meta-labeling: escalar el tamaño según
  `prob_sube`, no una fracción fija) — amplificó una señal ya buena (Bolt of canvas,
  +8.8%) pero no arregló una mala (Kwuarm, Runite ore).
- **Salida por triple barrera** (take-profit/stop-loss/timeout contra precios reales, en
  vez de vender siempre exactamente 1h después) — la mejora individual con más impacto
  medido (Kwuarm: pérdida casi a la mitad sin tocar el modelo), y combinada con
  dimensionamiento por confianza superó a cada una por separado en los 2 ítems donde se
  probaron juntas.

Si se retoman, el paso obligatorio antes de aplicarlas a producción es repetirlas con el
mismo estándar que la decisión de arriba: ventana grande (90 días+), universo no elegido a
mano, y test de significancia contra un azar con la misma mecánica de apuesta — no alcanza
con `win_rate`/ganancia descriptivos sobre una muestra chica.
