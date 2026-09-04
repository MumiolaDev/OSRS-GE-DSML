"""
Tests de los helpers de escritorio/paginas/pagina_modelos.py que no
necesitan una QApplication activa (son funciones planas, no métodos de un
QWidget) — _generar_ids_par, _formatear_fecha, _resumen_calidad. Se
importan sin problema aunque PySide6 no tenga una ventana corriendo (ver
el plan de la fase "app de escritorio v1", workstream 5/7): definir clases
QWidget no requiere una QApplication, solo instanciarlas.

_generar_ids_par y _resumen_calidad tocan una DB SQLite real (temporal,
ver el fixture `db`) porque dependen de OSRSBaseDatos.obtener_modelo_config
y de leer model_metrics respectivamente.
"""
import os
import sqlite3
import tempfile
import time

import pytest

from base_de_datos import OSRSBaseDatos
from escritorio.paginas.pagina_modelos import (
    _formatear_fecha, _generar_ids_par, _margen_error_95, _resumen_calidad,
)


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


class TestGenerarIdsPar:
    def test_slug_basico(self, db):
        id_regresor, id_clasificador = _generar_ids_par(db, "Mis Runas Favoritas")
        assert id_regresor == 'custom_mis_runas_favoritas_regresor'
        assert id_clasificador == 'custom_mis_runas_favoritas_clasificador'

    def test_caracteres_especiales_se_normalizan(self, db):
        id_regresor, id_clasificador = _generar_ids_par(db, "¡Armas de Élite! (top 10)")
        assert id_regresor.startswith('custom_')
        assert ' ' not in id_regresor and ' ' not in id_clasificador
        assert id_regresor.isascii() and id_clasificador.isascii()

    def test_nombre_sin_caracteres_validos_usa_fallback(self, db):
        id_regresor, id_clasificador = _generar_ids_par(db, "!!!")
        assert id_regresor == 'custom_modelo_regresor'
        assert id_clasificador == 'custom_modelo_clasificador'

    def test_colision_agrega_sufijo_numerico_compartido(self, db):
        primero_reg, primero_clas = _generar_ids_par(db, "Prueba")
        db.crear_modelo_config(
            model_id=primero_reg, nombre="Prueba", tipo='regresor',
            cadencia='manual', modo_seleccion='manual', item_ids=[1],
        )
        segundo_reg, segundo_clas = _generar_ids_par(db, "Prueba")
        assert segundo_reg != primero_reg
        assert segundo_reg == f"custom_prueba_1_regresor"
        assert segundo_clas == f"custom_prueba_1_clasificador"

    def test_colision_solo_en_clasificador_tambien_hace_avanzar_el_sufijo(self, db):
        """Si sólo el id de clasificador de un sufijo está ocupado, igual
        hay que saltar ese sufijo entero -- así el par sigue compartiendo
        el mismo número (ver el docstring de _generar_ids_par: quedan
        visiblemente emparejados en la tabla)."""
        primero_reg, primero_clas = _generar_ids_par(db, "Prueba")
        db.crear_modelo_config(
            model_id=primero_clas, nombre="Prueba", tipo='clasificador',
            cadencia='manual', modo_seleccion='manual', item_ids=[1],
        )
        segundo_reg, segundo_clas = _generar_ids_par(db, "Prueba")
        assert segundo_reg == 'custom_prueba_1_regresor'
        assert segundo_clas == 'custom_prueba_1_clasificador'


class TestMargenError95:
    """El veredicto de calidad ya no sale de umbrales fijos sobre la
    accuracy puntual sino de compararla contra el azar con su intervalo de
    confianza — lo que exige saber sobre cuántas observaciones se midió
    (model_metrics.n_evaluado)."""

    def test_sin_muestra_no_hay_margen(self):
        assert _margen_error_95(0.6, 0) is None
        assert _margen_error_95(None, 100) is None

    def test_mas_muestra_achica_el_margen(self):
        chico = _margen_error_95(0.6, 30)
        grande = _margen_error_95(0.6, 30_000)
        assert chico > grande
        # 30 observaciones dan ±17 puntos: 60% no se distingue de una moneda
        assert chico > 0.15
        assert grande < 0.01


class TestResumenCalidad:
    def _insertar_metrica(self, db, model_id, accuracy, train_ts=None, n_evaluado=1000,
                          modo='holdout'):
        conn = sqlite3.connect(db.db_path)
        c = conn.cursor()
        c.execute(
            '''INSERT INTO model_metrics
                (item_id, train_timestamp, model_name, mae, rmse,
                 horizonte_horas, accuracy_direccional, modo_evaluacion, n_evaluado)
               VALUES (NULL, ?, ?, 0.01, 0.02, 1, ?, ?, ?)''',
            (train_ts or int(time.time()), model_id, accuracy, modo, n_evaluado),
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

    def test_al_azar(self, db):
        self._insertar_metrica(db, 'modelo_x', 0.50)
        assert "Indistinguible del azar" in _resumen_calidad(db, 'modelo_x')

    def test_peor_que_el_azar_tambien_es_indistinguible_o_peor(self, db):
        self._insertar_metrica(db, 'modelo_x', 0.40)
        assert "Indistinguible del azar" in _resumen_calidad(db, 'modelo_x')

    def test_accuracy_alta_con_muestra_chica_no_se_declara_buena(self, db):
        """El caso que motivó el cambio: 60% sobre 20 movimientos entra
        dentro del ruido de una moneda, y antes se mostraba igual que 60%
        sobre 20.000."""
        self._insertar_metrica(db, 'modelo_x', 0.60, n_evaluado=20)
        assert "Indistinguible del azar" in _resumen_calidad(db, 'modelo_x')

    def test_usa_la_corrida_mas_reciente(self, db):
        ahora = int(time.time())
        self._insertar_metrica(db, 'modelo_x', 0.30, train_ts=ahora - 3600)
        self._insertar_metrica(db, 'modelo_x', 0.90, train_ts=ahora)
        assert "Buena señal" in _resumen_calidad(db, 'modelo_x')

    def test_walkforward_tiene_prioridad_sobre_holdout(self, db):
        ahora = int(time.time())
        self._insertar_metrica(db, 'modelo_x', 0.90, train_ts=ahora, modo='holdout')
        self._insertar_metrica(db, 'modelo_x', 0.52, train_ts=ahora, modo='walkforward')
        resumen = _resumen_calidad(db, 'modelo_x')
        assert "walk-forward" in resumen
        assert "52%" in resumen
