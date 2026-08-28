"""
Tests de los helpers de escritorio/paginas/pagina_modelos.py que no
necesitan una QApplication activa (son funciones planas, no métodos de un
QWidget) — _generar_model_id, _formatear_fecha, _resumen_calidad. Se
importan sin problema aunque PySide6 no tenga una ventana corriendo (ver
el plan de la fase "app de escritorio v1", workstream 5/7): definir clases
QWidget no requiere una QApplication, solo instanciarlas.

_generar_model_id y _resumen_calidad tocan una DB SQLite real (temporal,
ver el fixture `db`) porque dependen de OSRSBaseDatos.obtener_modelo_config
y de leer model_metrics respectivamente.
"""
import os
import sqlite3
import tempfile
import time

import pytest

from base_de_datos import OSRSBaseDatos
from escritorio.paginas.pagina_modelos import _formatear_fecha, _generar_model_id, _resumen_calidad


@pytest.fixture
def db():
    path = tempfile.mktemp(suffix='.db')
    base = OSRSBaseDatos(path)
    yield base
    for sufijo in ('', '-wal', '-shm'):
        p = path + sufijo
        if os.path.exists(p):
            os.remove(p)


class TestFormatearFecha:
    def test_none_da_nunca(self):
        assert _formatear_fecha(None) == "Nunca"

    def test_timestamp_valido_se_formatea(self):
        # 2024-01-01 00:00:00 UTC -- el formato exacto depende de la zona
        # horaria local (datetime.fromtimestamp), solo se verifica forma.
        resultado = _formatear_fecha(1704067200)
        assert len(resultado) == len("2024-01-01 00:00")
        assert resultado[:4].isdigit()


class TestGenerarModelId:
    def test_slug_basico(self, db):
        model_id = _generar_model_id(db, "Mis Runas Favoritas")
        assert model_id == 'custom_mis_runas_favoritas'

    def test_caracteres_especiales_se_normalizan(self, db):
        model_id = _generar_model_id(db, "¡Armas de Élite! (top 10)")
        assert model_id.startswith('custom_')
        assert ' ' not in model_id
        assert model_id.isascii()

    def test_nombre_sin_caracteres_validos_usa_fallback(self, db):
        assert _generar_model_id(db, "!!!") == 'custom_modelo'

    def test_colision_agrega_sufijo_numerico(self, db):
        primero = _generar_model_id(db, "Prueba")
        db.crear_modelo_config(
            model_id=primero, nombre="Prueba", tipo='regresor',
            cadencia='manual', modo_seleccion='manual', item_ids=[1],
        )
        segundo = _generar_model_id(db, "Prueba")
        assert segundo != primero
        assert segundo == f"{primero}_2"


class TestResumenCalidad:
    def _insertar_metrica(self, db, model_id, accuracy, train_ts=None):
        conn = sqlite3.connect(db.db_path)
        c = conn.cursor()
        c.execute(
            '''INSERT INTO model_metrics
                (item_id, train_timestamp, model_name, mae, rmse,
                 horizonte_horas, accuracy_direccional, modo_evaluacion)
               VALUES (NULL, ?, ?, 0.01, 0.02, 1, ?, 'holdout')''',
            (train_ts or int(time.time()), model_id, accuracy),
        )
        conn.commit()
        conn.close()

    def test_sin_entrenar(self, db):
        assert _resumen_calidad(db, 'no_entrenado_nunca') == "Sin entrenar todavía"

    def test_accuracy_none(self, db):
        self._insertar_metrica(db, 'modelo_x', None)
        assert _resumen_calidad(db, 'modelo_x') == "Entrenado (sin métrica de dirección)"

    def test_buena_senal(self, db):
        self._insertar_metrica(db, 'modelo_x', 0.60)
        assert "Buena señal" in _resumen_calidad(db, 'modelo_x')

    def test_regular(self, db):
        self._insertar_metrica(db, 'modelo_x', 0.50)
        assert "Regular" in _resumen_calidad(db, 'modelo_x')

    def test_debil(self, db):
        self._insertar_metrica(db, 'modelo_x', 0.40)
        assert "Débil" in _resumen_calidad(db, 'modelo_x')

    def test_usa_la_corrida_mas_reciente(self, db):
        ahora = int(time.time())
        self._insertar_metrica(db, 'modelo_x', 0.30, train_ts=ahora - 3600)
        self._insertar_metrica(db, 'modelo_x', 0.90, train_ts=ahora)
        assert "Buena señal" in _resumen_calidad(db, 'modelo_x')
