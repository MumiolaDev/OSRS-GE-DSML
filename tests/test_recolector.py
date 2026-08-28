"""
Tests de _agrupar_en_rangos (recolector.py) — agrupa timestamps faltantes en
rangos contiguos para acotar el replay histórico solo a los tramos que
realmente tenían un hueco. Un bug acá podría disparar un replay del tamaño
equivocado (de más, carísimo; de menos, deja huecos sin walk-forward).
"""
from recolector import _agrupar_en_rangos


def test_timestamps_vacios_da_lista_vacia():
    assert _agrupar_en_rangos([], step=5) == []


def test_un_solo_timestamp_es_un_rango_de_un_punto():
    assert _agrupar_en_rangos([100], step=5) == [(100, 100)]


def test_rango_unico_contiguo():
    assert _agrupar_en_rangos([100, 105, 110, 115], step=5) == [(100, 115)]


def test_dos_rangos_separados():
    assert _agrupar_en_rangos([100, 105, 110, 200, 205], step=5) == [(100, 110), (200, 205)]


def test_no_asume_orden_de_entrada():
    # el mismo caso que el anterior, pero desordenado
    resultado = _agrupar_en_rangos([205, 100, 200, 110, 105], step=5)
    assert resultado == [(100, 110), (200, 205)]


def test_step_distinto_de_5():
    # step de 1 hora (3600s): tres timestamps consecutivos, uno suelto
    timestamps = [3600, 7200, 10800, 21600]
    assert _agrupar_en_rangos(timestamps, step=3600) == [(3600, 10800), (21600, 21600)]
