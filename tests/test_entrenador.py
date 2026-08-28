"""
Tests de _clasificar_retorno (entrenador.py) — convierte el log-retorno
continuo en las 3 clases del clasificador direccional (0=baja, 1=estable,
2=sube). Si el umbral se aplica mal, el clasificador aprende sobre
etiquetas incorrectas sin que nada lo avise.

TestEntrenarModeloGlobalConItemIds toca una DB SQLite real (temporal, ver
el fixture `db_con_precios`) — a diferencia del resto de este archivo,
porque el parámetro item_ids (ver el plan de la fase "app de escritorio
v1", workstream 2) solo se puede verificar corriendo entrenar_modelo_global
de punta a punta: hace falta confirmar que con item_ids explícito
directamente NO se consulta la liquidez, no solo que el resultado final
coincida.
"""
import os
import random
import sqlite3
import tempfile
import time

import numpy as np
import pytest

from base_de_datos import OSRSBaseDatos
from entrenador import _clasificar_retorno, entrenar_modelo_global


def test_valores_por_encima_del_umbral_son_sube():
    resultado = _clasificar_retorno(np.array([0.01, 0.10]), umbral_pct=0.5)
    assert list(resultado) == [2, 2]


def test_valores_por_debajo_del_umbral_negativo_son_baja():
    resultado = _clasificar_retorno(np.array([-0.01, -0.10]), umbral_pct=0.5)
    assert list(resultado) == [0, 0]


def test_valores_dentro_del_umbral_son_estable():
    resultado = _clasificar_retorno(np.array([0.0, 0.001, -0.001]), umbral_pct=0.5)
    assert list(resultado) == [1, 1, 1]


def test_umbral_es_estrictamente_mayor_no_mayor_igual():
    # exactamente en el umbral (0.5% = 0.005 en espacio log-retorno) cuenta
    # como "estable", no "sube" -- la condición es > umbral, no >=.
    umbral = 0.5 / 100
    resultado = _clasificar_retorno(np.array([umbral]), umbral_pct=0.5)
    assert resultado[0] == 1


def test_umbral_personalizado():
    # con un umbral más chico (0.1%), un retorno de 0.2% ya cuenta como sube
    resultado = _clasificar_retorno(np.array([0.002]), umbral_pct=0.1)
    assert resultado[0] == 2


def test_array_mixto():
    target = np.array([0.02, 0.0, -0.02, 0.001])
    resultado = _clasificar_retorno(target, umbral_pct=0.5)
    assert list(resultado) == [2, 1, 0, 1]


@pytest.fixture
def db_con_precios():
    """DB temporal con 3 ítems y 60h de precios sintéticos — suficiente
    para que build_training_set/preprocess_item no descarten todo por
    historia insuficiente (mínimo max(lags, max(ma_windows)) + 2 = 8 filas
    con los defaults de entrenador.py)."""
    random.seed(0)
    path = tempfile.mktemp(suffix='.db')
    db = OSRSBaseDatos(path)

    conn = sqlite3.connect(path)
    c = conn.cursor()
    item_ids = [111, 222, 333]
    c.executemany(
        'INSERT INTO items (item_id, name, members, buy_limit) VALUES (?,?,?,?)',
        [(i, f'Item {i}', 0, 1000) for i in item_ids],
    )
    ahora = (int(time.time()) // 3600) * 3600
    filas = []
    for item_id in item_ids:
        precio = 100.0
        for h in range(60, 0, -1):
            precio *= (1 + random.uniform(-0.02, 0.02))
            low = max(1, precio)
            filas.append((item_id, ahora - h * 3600, round(low * 1.05), round(low), 50, 50))
    c.executemany(
        'INSERT OR IGNORE INTO precios_1h (item_id, timestamp, avg_high_price, avg_low_price, high_volume, low_volume) VALUES (?,?,?,?,?,?)',
        filas,
    )
    conn.commit()
    conn.close()

    yield db, item_ids

    for sufijo in ('', '-wal', '-shm'):
        p = path + sufijo
        if os.path.exists(p):
            os.remove(p)


class TestEntrenarModeloGlobalConItemIds:
    def test_item_ids_explicito_salta_la_consulta_de_liquidez(self, db_con_precios, monkeypatch):
        db, item_ids = db_con_precios

        def _falla_si_se_llama(*args, **kwargs):
            raise AssertionError("no debería consultar liquidez cuando item_ids viene explícito")

        monkeypatch.setattr(db, 'obtener_top_items_liquidez', _falla_si_se_llama)
        monkeypatch.setattr(db, 'obtener_top_items_liquidez_hasta', _falla_si_se_llama)

        exito = entrenar_modelo_global(
            db, model_name='test_manual', item_ids=item_ids, guardar_en_disco=False,
        )
        assert exito is True

    def test_entrena_solo_sobre_los_items_pedidos(self, db_con_precios):
        db, item_ids = db_con_precios
        subconjunto = item_ids[:2]

        exito = entrenar_modelo_global(
            db, model_name='test_subconjunto', item_ids=subconjunto, guardar_en_disco=False,
        )
        assert exito is True

        conn = sqlite3.connect(db.db_path)
        c = conn.cursor()
        c.execute(
            "SELECT DISTINCT item_id FROM predicciones WHERE model_version = 'test_subconjunto'"
        )
        items_predichos = {row[0] for row in c.fetchall()}
        conn.close()
        assert items_predichos <= set(subconjunto)

    def test_sin_item_ids_usa_liquidez_como_siempre(self, db_con_precios):
        db, item_ids = db_con_precios
        # Sin item_ids, entrenar_modelo_global cae al camino de liquidez de
        # siempre -- con la DB de este fixture (sin resumen_actual poblada)
        # eso no tiene items disponibles y debe cortar temprano, no
        # explotar ni entrenar sobre datos inventados.
        exito = entrenar_modelo_global(db, model_name='test_sin_item_ids', guardar_en_disco=False)
        assert exito is False
