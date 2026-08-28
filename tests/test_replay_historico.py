"""
Tests de calcular_checkpoints (replay_historico.py) — determina en qué
momentos exactos habrían corrido job_horario/job_diario durante un rango
del pasado. Un bug acá dispara un replay del tamaño equivocado: de más
(carísimo, cientos de reentrenamientos de sobra) o de menos (deja huecos de
walk-forward sin cubrir).

Se usa el timestamp 0 (1970-01-01 00:00:00 UTC) como "desde" en la mayoría
de los tests: da un caso chico y verificable a mano sin tener que calcular
epochs de fechas reales.
"""
from replay_historico import calcular_checkpoints


def test_rango_de_5_horas_desde_epoch():
    # desde=0 (00:00 UTC), hasta=18000 (05:00 UTC), offsets default (:05, 03:00)
    checkpoints = calcular_checkpoints(0, 5 * 3600)

    horarios = [ts for ts, tipo in checkpoints if tipo == 'horario']
    diarios = [ts for ts, tipo in checkpoints if tipo == 'diario']

    # 00:05, 01:05, 02:05, 03:05, 04:05 -- 05:05 queda justo fuera del rango
    assert horarios == [300, 3900, 7500, 11100, 14700]
    # 03:00 es el único cruce de las 03:00 UTC dentro de este rango de 5h
    assert diarios == [10800]


def test_checkpoints_horarios_siempre_en_el_mismo_minuto():
    checkpoints = calcular_checkpoints(0, 10 * 86400, minuto_offset_horario=5)
    horarios = [ts for ts, tipo in checkpoints if tipo == 'horario']
    assert horarios  # no vacío
    assert all((ts - 5 * 60) % 3600 == 0 for ts in horarios)


def test_checkpoints_diarios_siempre_a_la_misma_hora_utc():
    checkpoints = calcular_checkpoints(0, 10 * 86400, hora_diaria=3)
    diarios = [ts for ts, tipo in checkpoints if tipo == 'diario']
    assert diarios  # no vacío
    assert all(ts % 86400 == 3 * 3600 for ts in diarios)


def test_orden_cronologico():
    checkpoints = calcular_checkpoints(0, 3 * 86400)
    timestamps = [ts for ts, _ in checkpoints]
    assert timestamps == sorted(timestamps)


def test_rango_vacio_si_desde_es_posterior_a_hasta():
    assert calcular_checkpoints(10_000, 0) == []


def test_limite_inferior_inclusive():
    # desde_ts cae EXACTO en un checkpoint horario (00:05:00) -> se incluye
    checkpoints = calcular_checkpoints(300, 300)
    assert (300, 'horario') in checkpoints


def test_se_salta_al_siguiente_checkpoint_si_desde_ya_lo_paso():
    # desde_ts = 301, un segundo después de las 00:05:00 -> el primer
    # checkpoint horario válido es 01:05:00 (3900), no 00:05:00
    checkpoints = calcular_checkpoints(301, 3900)
    horarios = [ts for ts, tipo in checkpoints if tipo == 'horario']
    assert horarios == [3900]


def test_offsets_personalizados():
    checkpoints = calcular_checkpoints(0, 2 * 3600, minuto_offset_horario=30, hora_diaria=1)
    horarios = [ts for ts, tipo in checkpoints if tipo == 'horario']
    diarios = [ts for ts, tipo in checkpoints if tipo == 'diario']
    assert horarios == [1800, 5400]  # 00:30, 01:30
    assert diarios == [3600]  # 01:00
