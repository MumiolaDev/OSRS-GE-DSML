"""
Tests de las operaciones de modelos_config (base_de_datos.py) — primer
archivo de tests que toca una DB SQLite real (temporal, aislada por test
vía el fixture `db`), porque el CRUD de modelos_config no tiene una
versión "pura" sin DB para probar por separado (a diferencia de
metricas.py/recolector.py/replay_historico.py/entrenador.py, ver el resto
de tests/).

Cubre lo que ya se validó a mano con smoke tests durante el desarrollo de
la app de escritorio (ver el plan de la fase "app de escritorio v1",
workstream 1/5): siembra de los modelos productivos, alta, filtros,
actualización, los topes técnicos (MAX_ITEMS_POR_MODELO/
MAX_MODELOS_ACTIVOS) y borrado.
"""
import os
import tempfile

import pytest

from base_de_datos import OSRSBaseDatos, MAX_ITEMS_POR_MODELO, MAX_MODELOS_ACTIVOS

MODELOS_PRODUCTIVOS = {'global_horario', 'global_diario', 'f2p10_100gp_clasif'}


@pytest.fixture
def db():
    path = tempfile.mktemp(suffix='.db')
    base = OSRSBaseDatos(path)
    yield base
    for sufijo in ('', '-wal', '-shm'):
        p = path + sufijo
        if os.path.exists(p):
            os.remove(p)


class TestSiembraModelosProductivos:
    def test_siembra_los_3_modelos_productivos(self, db):
        ids = {m['model_id'] for m in db.listar_modelos_config()}
        assert ids == MODELOS_PRODUCTIVOS

    def test_siembra_es_idempotente_y_no_pisa_cambios_del_usuario(self, db):
        db.actualizar_modelo_config('global_horario', estado='pausado')
        OSRSBaseDatos(db.db_path)  # segunda apertura -> _migrar_esquema/siembra de nuevo
        modelos = db.listar_modelos_config()
        assert len([m for m in modelos if m['model_id'] == 'global_horario']) == 1
        assert db.obtener_modelo_config('global_horario')['estado'] == 'pausado'


class TestCrearModeloConfig:
    def test_alta_manual_basica(self, db):
        db.crear_modelo_config(
            model_id='custom_x', nombre='Mi modelo', tipo='regresor', cadencia='manual',
            modo_seleccion='manual', item_ids=[1, 2, 3],
        )
        m = db.obtener_modelo_config('custom_x')
        assert m['nombre'] == 'Mi modelo'
        assert m['item_ids'] == [1, 2, 3]
        assert m['tipo'] == 'regresor'
        assert m['estado'] == 'activo'
        assert m['ultimo_entrenamiento_ts'] is None

    def test_tope_de_items_por_modelo(self, db):
        with pytest.raises(ValueError):
            db.crear_modelo_config(
                model_id='custom_grande', nombre='Grande', tipo='regresor', cadencia='manual',
                modo_seleccion='manual', item_ids=list(range(MAX_ITEMS_POR_MODELO + 1)),
            )
        assert db.obtener_modelo_config('custom_grande') is None

    def test_tope_de_modelos_activos(self, db):
        # Ya hay 3 sembrados con cadencia horaria/diaria; llenar hasta el tope.
        for i in range(MAX_MODELOS_ACTIVOS - len(MODELOS_PRODUCTIVOS)):
            db.crear_modelo_config(
                model_id=f'extra_{i}', nombre=f'extra {i}', tipo='regresor',
                cadencia='horaria', modo_seleccion='manual', item_ids=[1],
            )
        assert db.contar_modelos_config_activos() == MAX_MODELOS_ACTIVOS
        with pytest.raises(ValueError):
            db.crear_modelo_config(
                model_id='uno_de_mas', nombre='de más', tipo='regresor',
                cadencia='horaria', modo_seleccion='manual', item_ids=[1],
            )
        assert db.obtener_modelo_config('uno_de_mas') is None

    def test_cadencia_manual_no_cuenta_contra_el_tope(self, db):
        for i in range(MAX_MODELOS_ACTIVOS + 3):
            db.crear_modelo_config(
                model_id=f'manual_{i}', nombre=f'manual {i}', tipo='regresor',
                cadencia='manual', modo_seleccion='manual', item_ids=[1],
            )
        assert db.contar_modelos_config_activos() == len(MODELOS_PRODUCTIVOS)


class TestListarModelosConfig:
    def test_filtra_por_cadencia_y_estado(self, db):
        ids = {m['model_id'] for m in db.listar_modelos_config(cadencia='horaria', estado='activo')}
        assert ids == {'global_horario', 'f2p10_100gp_clasif'}

    def test_sin_filtro_devuelve_todos(self, db):
        assert len(db.listar_modelos_config()) == len(MODELOS_PRODUCTIVOS)


class TestActualizarModeloConfig:
    def test_actualiza_campos_simples(self, db):
        db.actualizar_modelo_config('global_horario', ultimo_entrenamiento_ts=12345)
        assert db.obtener_modelo_config('global_horario')['ultimo_entrenamiento_ts'] == 12345

    def test_pausar_y_reactivar_dentro_del_tope(self, db):
        db.actualizar_modelo_config('global_horario', estado='pausado')
        assert db.obtener_modelo_config('global_horario')['estado'] == 'pausado'
        db.actualizar_modelo_config('global_horario', estado='activo')
        assert db.obtener_modelo_config('global_horario')['estado'] == 'activo'

    def test_reactivar_respeta_el_tope_de_activos(self, db):
        db.actualizar_modelo_config('global_horario', estado='pausado')
        for i in range(MAX_MODELOS_ACTIVOS - (len(MODELOS_PRODUCTIVOS) - 1)):
            db.crear_modelo_config(
                model_id=f'lleno_{i}', nombre=f'lleno {i}', tipo='regresor',
                cadencia='horaria', modo_seleccion='manual', item_ids=[1],
            )
        assert db.contar_modelos_config_activos() == MAX_MODELOS_ACTIVOS
        with pytest.raises(ValueError):
            db.actualizar_modelo_config('global_horario', estado='activo')

    def test_id_inexistente_no_falla(self, db):
        assert db.actualizar_modelo_config('no_existe', estado='pausado') == 0


class TestEliminarModeloConfig:
    def test_elimina_un_modelo_custom(self, db):
        db.crear_modelo_config(
            model_id='custom_borrar', nombre='x', tipo='regresor',
            cadencia='manual', modo_seleccion='manual', item_ids=[1],
        )
        assert db.eliminar_modelo_config('custom_borrar') == 1
        assert db.obtener_modelo_config('custom_borrar') is None

    def test_id_inexistente_no_falla(self, db):
        assert db.eliminar_modelo_config('no_existe') == 0
