"""
Tests de las operaciones de modelos_config (base_de_datos.py) — primer
archivo de tests que toca una DB SQLite real (temporal, aislada por test
vía el fixture `db`), porque el CRUD de modelos_config no tiene una
versión "pura" sin DB para probar por separado (a diferencia de
metricas.py/recolector.py/replay_historico.py/entrenador.py, ver el resto
de tests/).

Cubre lo que ya se validó a mano con smoke tests durante el desarrollo de
la app de escritorio (ver el plan de la fase "app de escritorio v1",
workstream 1/5): alta, filtros, actualización, los topes técnicos
(MAX_ITEMS_POR_MODELO/MAX_MODELOS_ACTIVOS) y borrado. No hay ningún modelo
sembrado por default (pedido explícito del usuario) — una DB recién creada
arranca con modelos_config vacía, así que cada test crea explícitamente
los modelos que necesita en vez de depender de un estado inicial sembrado.
"""
import os
import sqlite3
import tempfile
import time

import pytest

from base_de_datos import OSRSBaseDatos, MAX_ITEMS_POR_MODELO, MAX_MODELOS_ACTIVOS


@pytest.fixture
def db():
    path = tempfile.mktemp(suffix='.db')
    base = OSRSBaseDatos(path)
    yield base
    for sufijo in ('', '-wal', '-shm'):
        p = path + sufijo
        if os.path.exists(p):
            os.remove(p)


class TestSinSiembraPorDefault:
    def test_db_nueva_arranca_sin_modelos(self, db):
        assert db.listar_modelos_config() == []

    def test_segunda_apertura_no_agrega_nada(self, db):
        db.crear_modelo_config(
            model_id='custom_x', nombre='Mi modelo', tipo='regresor', cadencia='manual',
            modo_seleccion='manual', item_ids=[1],
        )
        OSRSBaseDatos(db.db_path)  # segunda apertura -> _migrar_esquema de nuevo
        ids = {m['model_id'] for m in db.listar_modelos_config()}
        assert ids == {'custom_x'}


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
        for i in range(MAX_MODELOS_ACTIVOS):
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
        assert db.contar_modelos_config_activos() == 0


class TestListarModelosConfig:
    def test_filtra_por_cadencia_y_estado(self, db):
        db.crear_modelo_config(
            model_id='h1', nombre='h1', tipo='regresor', cadencia='horaria',
            modo_seleccion='manual', item_ids=[1],
        )
        db.crear_modelo_config(
            model_id='d1', nombre='d1', tipo='regresor', cadencia='diaria',
            modo_seleccion='manual', item_ids=[1],
        )
        ids = {m['model_id'] for m in db.listar_modelos_config(cadencia='horaria', estado='activo')}
        assert ids == {'h1'}

    def test_sin_filtro_devuelve_todos(self, db):
        for i in range(3):
            db.crear_modelo_config(
                model_id=f'm{i}', nombre=f'm{i}', tipo='regresor', cadencia='manual',
                modo_seleccion='manual', item_ids=[1],
            )
        assert len(db.listar_modelos_config()) == 3


class TestActualizarModeloConfig:
    def _crear(self, db, model_id='custom_x', cadencia='horaria'):
        db.crear_modelo_config(
            model_id=model_id, nombre=model_id, tipo='regresor', cadencia=cadencia,
            modo_seleccion='manual', item_ids=[1],
        )

    def test_actualiza_campos_simples(self, db):
        self._crear(db)
        db.actualizar_modelo_config('custom_x', ultimo_entrenamiento_ts=12345)
        assert db.obtener_modelo_config('custom_x')['ultimo_entrenamiento_ts'] == 12345

    def test_pausar_y_reactivar_dentro_del_tope(self, db):
        self._crear(db)
        db.actualizar_modelo_config('custom_x', estado='pausado')
        assert db.obtener_modelo_config('custom_x')['estado'] == 'pausado'
        db.actualizar_modelo_config('custom_x', estado='activo')
        assert db.obtener_modelo_config('custom_x')['estado'] == 'activo'

    def test_reactivar_respeta_el_tope_de_activos(self, db):
        self._crear(db)
        db.actualizar_modelo_config('custom_x', estado='pausado')
        for i in range(MAX_MODELOS_ACTIVOS):
            db.crear_modelo_config(
                model_id=f'lleno_{i}', nombre=f'lleno {i}', tipo='regresor',
                cadencia='horaria', modo_seleccion='manual', item_ids=[1],
            )
        assert db.contar_modelos_config_activos() == MAX_MODELOS_ACTIVOS
        with pytest.raises(ValueError):
            db.actualizar_modelo_config('custom_x', estado='activo')

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


class TestObtenerResumenDatos:
    """
    obtener_resumen_datos() -- usado por escritorio/paginas/pagina_inicio.py
    para el panel "Datos disponibles" (ver el plan de la fase "pestaña
    Inicio: estado detallado, progreso y datos disponibles").
    """

    def test_tablas_vacias(self, db):
        resumen = db.obtener_resumen_datos()
        assert set(resumen.keys()) == {'precios_5m', 'precios_1h', 'precios_6h', 'precios_1h_diario'}
        for info in resumen.values():
            assert info == {'filas': 0, 'desde_ts': None, 'hasta_ts': None, 'dias_historial': None}

    def test_precios_1h_con_datos(self, db):
        ahora = (int(time.time()) // 3600) * 3600
        conn = sqlite3.connect(db.db_path)
        c = conn.cursor()
        filas = [(1, ahora - h * 3600, 100, 90, 10, 10) for h in range(48, 0, -1)]  # 48 filas, últimas 47h
        c.executemany(
            'INSERT INTO precios_1h (item_id, timestamp, avg_high_price, avg_low_price, high_volume, low_volume) VALUES (?,?,?,?,?,?)',
            filas,
        )
        conn.commit()
        conn.close()

        info = db.obtener_resumen_datos()['precios_1h']
        assert info['filas'] == 48
        assert info['desde_ts'] == ahora - 48 * 3600
        assert info['hasta_ts'] == ahora - 1 * 3600
        assert info['dias_historial'] == pytest.approx(47 / 24)

    def test_precios_1h_diario_usa_la_columna_fecha(self, db):
        # precios_1h_diario usa `fecha`, no `timestamp`, como columna de
        # tiempo -- confirma que obtener_resumen_datos() no la confunde
        # con las otras 3 tablas.
        ahora = (int(time.time()) // 86400) * 86400
        conn = sqlite3.connect(db.db_path)
        c = conn.cursor()
        c.execute(
            'INSERT INTO precios_1h_diario (item_id, fecha, avg_high_price, avg_low_price, high_volume, low_volume) VALUES (1, ?, 100, 90, 500, 500)',
            (ahora - 5 * 86400,),
        )
        c.execute(
            'INSERT INTO precios_1h_diario (item_id, fecha, avg_high_price, avg_low_price, high_volume, low_volume) VALUES (1, ?, 100, 90, 500, 500)',
            (ahora,),
        )
        conn.commit()
        conn.close()

        info = db.obtener_resumen_datos()['precios_1h_diario']
        assert info['filas'] == 2
        assert info['dias_historial'] == pytest.approx(5.0)

    def test_tabla_con_un_solo_dato_da_cero_dias(self, db):
        conn = sqlite3.connect(db.db_path)
        c = conn.cursor()
        c.execute(
            'INSERT INTO precios_6h (item_id, timestamp, avg_high_price, avg_low_price, high_volume, low_volume) VALUES (1, ?, 100, 90, 10, 10)',
            (int(time.time()),),
        )
        conn.commit()
        conn.close()

        info = db.obtener_resumen_datos()['precios_6h']
        assert info['filas'] == 1
        assert info['dias_historial'] == 0
