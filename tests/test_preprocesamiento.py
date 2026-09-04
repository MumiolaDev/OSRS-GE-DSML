"""
Tests de preprocesamiento.py — las dos correcciones de fondo de la revisión:
la grilla temporal regular y las features relativas.

Son funciones puras sobre DataFrames (no tocan la DB ni la red), así que se
prueban directo con series armadas a mano.
"""
import numpy as np
import pandas as pd
import pytest

from preprocesamiento import (
    FEATURES_VERSION, _reindexar_a_grilla, columnas_feature, construir_features, preprocess_item,
)

PASO = 3600


def serie(precios, ts_inicial=1_700_000_000, paso=PASO, timestamps=None, volumen_alto=10, volumen_bajo=10):
    """DataFrame con el formato de precios_1h para una lista de precios."""
    if timestamps is None:
        timestamps = [ts_inicial + i * paso for i in range(len(precios))]
    return pd.DataFrame({
        'item_id': 2,
        'timestamp': timestamps,
        'avg_low_price': precios,
        'avg_high_price': [p * 1.02 for p in precios],
        'high_volume': volumen_alto,
        'low_volume': volumen_bajo,
    })


class TestReindexarAGrilla:
    def test_serie_completa_no_cambia_de_largo(self):
        df = serie([100, 101, 102, 103])
        assert len(_reindexar_a_grilla(df, PASO)) == 4

    def test_hueco_se_rellena_con_una_fila_nan(self):
        ts = [0, PASO, 3 * PASO]  # falta 2*PASO
        df = serie([100, 101, 103], timestamps=ts)
        reindexado = _reindexar_a_grilla(df, PASO)
        assert len(reindexado) == 4
        assert reindexado['avg_low_price'].isna().sum() == 1
        # item_id es constante de la serie, no un dato del período
        assert reindexado['item_id'].notna().all()

    def test_timestamps_duplicados_gana_el_ultimo(self):
        """El pronóstico recursivo de prediccion.py concatena filas
        sintéticas que pueden caer sobre un timestamp ya presente; sin
        deduplicar, reindex() lanza excepción."""
        df = pd.concat([
            serie([100, 101], timestamps=[0, PASO]),
            serie([999], timestamps=[PASO]),  # pisa el mismo bucket
        ], ignore_index=True)
        reindexado = _reindexar_a_grilla(df, PASO)
        assert len(reindexado) == 2
        assert reindexado.iloc[-1]['avg_low_price'] == 999


class TestFeaturesRelativas:
    def test_no_queda_ninguna_feature_en_nivel_de_precio(self):
        """El bug de fondo: con lags y medias móviles en log-precio absoluto,
        un árbol no puede formar el retorno (no puede restar dos features), y
        encima el precio actual ni siquiera estaba entre las features."""
        feats = construir_features(serie(list(range(100, 140))), paso_segundos=PASO)
        cols = columnas_feature(feats)
        assert not any(c.startswith('price_lag_') for c in cols)
        assert not any(c.startswith('ma_price_') for c in cols)
        assert 'log_price' not in cols
        assert any(c.startswith('ret_lag_') for c in cols)
        assert any(c.startswith('dist_ma_price_') for c in cols)

    def test_ret_lag_1_es_el_retorno_del_periodo_anterior(self):
        precios = [100, 110, 121]  # +10% cada paso
        feats = construir_features(serie(precios), paso_segundos=PASO)
        assert feats['ret_lag_1'].iloc[-1] == pytest.approx(np.log(1.10), rel=1e-9)

    def test_ret_lag_2_acumula_dos_periodos(self):
        feats = construir_features(serie([100, 110, 121]), paso_segundos=PASO)
        assert feats['ret_lag_2'].iloc[-1] == pytest.approx(np.log(1.21), rel=1e-9)

    def test_desbalance_de_volumen_se_calcula(self):
        """high_volume/(high+low): presión compradora dentro del período.
        Estaba en los datos crudos desde siempre pero se perdía al sumar los
        dos volúmenes en un total."""
        feats = construir_features(
            serie([100] * 10, volumen_alto=30, volumen_bajo=10), paso_segundos=PASO,
        )
        assert feats['desbalance_volumen'].iloc[-1] == pytest.approx(0.75)

    def test_sin_volumen_el_desbalance_es_nan_no_un_valor_inventado(self):
        feats = construir_features(
            serie([100] * 10, volumen_alto=0, volumen_bajo=0), paso_segundos=PASO,
        )
        assert feats['desbalance_volumen'].isna().all()


class TestTargetSobreLaGrilla:
    def test_target_es_siempre_de_un_solo_periodo(self):
        """Con un hueco en el medio, la fila anterior al hueco no puede
        generar target: su 'período siguiente' no existe. Antes se
        emparejaba con el próximo dato disponible, que podía estar muchas
        horas después, y el modelo aprendía una mezcla de horizontes."""
        ts = [i * PASO for i in range(30)]
        del ts[20]  # el bucket 20 no existe
        precios = [100 + i for i in range(29)]
        dataset = preprocess_item(serie(precios, timestamps=ts), paso_segundos=PASO)

        assert not dataset.empty
        # El bucket faltante nunca puede ser un target: la fila de 19 no
        # tiene período siguiente. Antes se emparejaba con el bucket 21 y esa
        # fila entraba al entrenamiento como si fuera un movimiento de 1h.
        assert 20 * PASO not in dataset['timestamp_target'].values
        assert 21 * PASO not in dataset['timestamp_target'].values

    def test_serie_sin_huecos_produce_targets_contiguos(self):
        dataset = preprocess_item(serie([100 + i for i in range(40)]), paso_segundos=PASO)
        assert not dataset.empty
        # price_target/price_actual difieren exactamente en un paso de precio
        assert (dataset['price_target'] - dataset['price_actual'] == 1).all()


def test_features_version_esta_definida():
    """El guard de prediccion.py depende de que esta constante exista y
    cambie cuando cambien las features."""
    assert isinstance(FEATURES_VERSION, int)
    assert FEATURES_VERSION >= 2
