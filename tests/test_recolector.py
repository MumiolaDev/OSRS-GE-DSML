"""
Tests de _agrupar_en_rangos (recolector.py) — agrupa timestamps faltantes en
rangos contiguos para acotar el replay histórico solo a los tramos que
realmente tenían un hueco. Un bug acá podría disparar un replay del tamaño
equivocado (de más, carísimo; de menos, deja huecos sin walk-forward).
Además: el filtro de ítems por intervalo (_filtro_items/items_de_interes),
que es lo que acota precios_5m a los ítems que realmente se operan.
"""
import os
import tempfile

import pytest

import recolector
from base_de_datos import OSRSBaseDatos
from recolector import _agrupar_en_rangos, _filtro_items, items_de_interes


@pytest.fixture
def db_temporal():
    path = tempfile.mktemp(suffix='.db')
    base = OSRSBaseDatos(path)
    yield base
    for sufijo in ('', '-wal', '-shm'):
        p = path + sufijo
        if os.path.exists(p):
            os.remove(p)


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


class TestFiltroDeItems:
    """
    precios_5m dejó de guardar el catálogo entero: con 1.776 ítems por
    snapshot y 288 snapshots por día, la tabla tendía a 634 MB para algo que
    ningún modelo ni el screener leen. Ahora guarda solo los ítems de los
    modelos activos, que son los que hacen falta para calibrar la ejecución
    — ver recolector.INTERVALOS_ACOTADOS_A_MODELOS.
    """

    def test_sin_modelos_no_hay_items_de_interes(self, db_temporal):
        assert items_de_interes(db_temporal) == []

    def test_junta_los_items_de_todos_los_modelos_activos(self, db_temporal):
        db_temporal.crear_modelo_config(
            model_id='a', nombre='A', tipo='spread', cadencia='horaria',
            modo_seleccion='manual', item_ids=[10, 20])
        db_temporal.crear_modelo_config(
            model_id='b', nombre='B', tipo='spread', cadencia='horaria',
            modo_seleccion='manual', item_ids=[20, 30])
        assert items_de_interes(db_temporal) == [10, 20, 30]

    def test_un_modelo_pausado_no_cuenta(self, db_temporal):
        db_temporal.crear_modelo_config(
            model_id='a', nombre='A', tipo='spread', cadencia='horaria',
            modo_seleccion='manual', item_ids=[10, 20])
        db_temporal.actualizar_modelo_config('a', estado='pausado')
        assert items_de_interes(db_temporal) == []

    def test_1h_y_6h_nunca_se_filtran(self, db_temporal):
        """El screener cubre todo el catálogo y el ranking de liquidez
        necesita el universo completo: acotar precios_1h rompería las dos
        cosas."""
        assert _filtro_items(db_temporal, '1h') is None
        assert _filtro_items(db_temporal, '6h') is None

    def test_5m_se_filtra_a_los_items_de_los_modelos(self, db_temporal):
        db_temporal.crear_modelo_config(
            model_id='a', nombre='A', tipo='spread', cadencia='horaria',
            modo_seleccion='manual', item_ids=[10, 20])
        assert _filtro_items(db_temporal, '5m') == [10, 20]

    def test_5m_sin_modelos_da_lista_vacia_no_none(self, db_temporal):
        """La distinción importa: None significa "todos" y una lista vacía
        significa "ninguno" — con lista vacía collect_programado saltea la
        request entera en vez de pedirla para tirar el resultado."""
        assert _filtro_items(db_temporal, '5m') == []

    def test_el_override_manual_gana_sobre_todo(self, db_temporal, monkeypatch):
        monkeypatch.setattr(recolector, 'ITEM_IDS', [99])
        assert _filtro_items(db_temporal, '5m') == [99]
        assert _filtro_items(db_temporal, '1h') == [99]
