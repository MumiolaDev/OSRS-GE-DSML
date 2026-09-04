"""
Tests de las funciones puras de metricas.py — sin DB, sin red. Cubren
específicamente los casos límite que ya causaron ruido real en el screener
(ítems de precio ~0 con ROI absurdo) o que son fáciles de romper sin darse
cuenta al tocar la lógica de impuesto/margen.
"""
import math

import pandas as pd
import pytest

import metricas
from metricas import (
    calcular_impuesto_ge,
    calcular_margen,
    calcular_profit_potencial,
    dimensionar_oportunidad,
    filtrar_screener_liquido,
    margen_neto_proyectado,
    pct_cambio,
    percentil_historico,
    tendencia_volumen,
    volatilidad,
    EXEMPT_ITEM_IDS,
    GE_TAX_CAP,
)


class TestCalcularImpuestoGe:
    def test_2_por_ciento_redondeado_hacia_abajo(self):
        # floor(1000 * 0.02) = 20
        assert calcular_impuesto_ge(1000, item_id=4151) == 20

    def test_venta_bajo_50gp_no_paga_impuesto(self):
        # floor(49 * 0.02) = 0
        assert calcular_impuesto_ge(49, item_id=4151) == 0

    def test_item_exento_no_paga_nunca(self):
        item_exento = next(iter(EXEMPT_ITEM_IDS))
        assert calcular_impuesto_ge(1_000_000, item_id=item_exento) == 0

    def test_tope_de_5_millones(self):
        # 2% de 1000M = 20M, muy por encima del tope
        assert calcular_impuesto_ge(1_000_000_000, item_id=4151) == GE_TAX_CAP

    @pytest.mark.parametrize("precio_invalido", [0, None, -5])
    def test_precio_invalido_da_cero(self, precio_invalido):
        assert calcular_impuesto_ge(precio_invalido, item_id=4151) == 0


class TestCalcularMargen:
    def test_margen_bruto_y_neto_consistentes(self):
        resultado = calcular_margen(avg_high_price=110, avg_low_price=100, item_id=4151)
        impuesto_esperado = math.floor(110 * 0.02)
        assert resultado['margen_bruto'] == 10
        assert resultado['impuesto_ge'] == impuesto_esperado
        assert resultado['margen_neto'] == 10 - impuesto_esperado
        assert resultado['roi_pct'] == pytest.approx((10 - impuesto_esperado) / 100 * 100)

    def test_avg_low_price_cero_da_roi_none(self):
        # El caso real que generaba roi_pct de 141600% en el screener antes
        # del filtro de liquidez — acá al menos el cálculo en sí no debe
        # explotar ni dar un número sin sentido cuando el precio es 0.
        resultado = calcular_margen(avg_high_price=10, avg_low_price=0, item_id=4151)
        assert resultado['roi_pct'] is None


class TestCalcularProfitPotencial:
    def test_normal(self):
        assert calcular_profit_potencial(margen_neto=10, buy_limit=100) == 1000

    @pytest.mark.parametrize("buy_limit,margen_neto", [(None, 10), (100, None)])
    def test_none_propaga_none(self, buy_limit, margen_neto):
        assert calcular_profit_potencial(margen_neto, buy_limit) is None


class TestPctCambio:
    def test_normal(self):
        assert pct_cambio([100, 110]) == pytest.approx(10.0)

    def test_menos_de_dos_valores(self):
        assert pct_cambio([100]) is None
        assert pct_cambio([]) is None

    def test_primer_valor_cero(self):
        assert pct_cambio([0, 100]) is None


class TestVolatilidad:
    def test_normal_no_negativa(self):
        v = volatilidad([100, 110, 90, 105])
        assert v is not None
        assert v >= 0

    def test_serie_constante_da_cero(self):
        assert volatilidad([100, 100, 100]) == 0

    def test_menos_de_dos_valores(self):
        assert volatilidad([100]) is None


class TestPercentilHistorico:
    def test_en_el_medio(self):
        assert percentil_historico(150, [100, 200]) == pytest.approx(50.0)

    def test_en_el_minimo_y_maximo(self):
        assert percentil_historico(100, [100, 200]) == 0.0
        assert percentil_historico(200, [100, 200]) == 100.0

    def test_serie_sin_rango_da_50(self):
        assert percentil_historico(100, [100, 100, 100]) == 50.0

    def test_serie_vacia(self):
        assert percentil_historico(100, []) is None


class TestTendenciaVolumen:
    def test_volumen_subiendo(self):
        # 6 horas "recientes" con volumen 20, resto con volumen 10 -> ratio 2
        serie = [10] * 6 + [20] * 6
        assert tendencia_volumen(serie, ventana_reciente=6) == pytest.approx(2.0)

    def test_serie_muy_corta_da_none(self):
        assert tendencia_volumen([10, 20], ventana_reciente=6) is None

    def test_resto_en_cero_da_none(self):
        assert tendencia_volumen([0, 0, 0, 10, 10, 10, 10], ventana_reciente=4) is None


class TestDimensionarOportunidad:
    def test_acota_por_volumen_no_por_buy_limit(self):
        # buy_limit generoso, pero el volumen esperado es la restricción real
        resultado = dimensionar_oportunidad(
            margen_neto=10, buy_limit=10_000, volumen_1h_promedio=50,
            horas_horizonte=3, fraccion_participacion=0.15,
        )
        volumen_capturable = 50 * 3 * 0.15  # 22.5
        assert resultado == pytest.approx(10 * volumen_capturable)

    def test_acota_por_buy_limit_cuando_el_volumen_sobra(self):
        resultado = dimensionar_oportunidad(
            margen_neto=10, buy_limit=5, volumen_1h_promedio=100_000,
            horas_horizonte=4, fraccion_participacion=0.15,
        )
        assert resultado == pytest.approx(10 * 5)

    @pytest.mark.parametrize("margen_neto,buy_limit", [(None, 100), (10, None)])
    def test_none_propaga_none(self, margen_neto, buy_limit):
        assert dimensionar_oportunidad(margen_neto, buy_limit, 100, 1) is None


class TestMargenNetoProyectado:
    def test_descuenta_impuesto_sobre_precio_de_venta_proyectado(self):
        # spread actual 110/100 = 1.10; predicho low=105 -> high proyectado 115.5
        resultado = margen_neto_proyectado(
            avg_low_price_actual=100, avg_high_price_actual=110,
            avg_low_price_predicho=105, item_id=4151,
        )
        high_proyectado = 105 * (110 / 100)
        impuesto = calcular_impuesto_ge(high_proyectado, 4151)
        assert resultado == pytest.approx(high_proyectado - 100 - impuesto)

    @pytest.mark.parametrize("low,high", [(0, 110), (100, 0), (None, 110)])
    def test_precio_actual_invalido_da_none(self, low, high):
        assert margen_neto_proyectado(low, high, 105, 4151) is None


class TestFiltrarScreenerLiquido:
    def _resumen(self):
        return pd.DataFrame([
            {'item_id': 1, 'volumen_24h': 10, 'margen_neto': 850, 'roi_pct': 42500.0},  # ilíquido
            {'item_id': 2, 'volumen_24h': 500, 'margen_neto': 5, 'roi_pct': 2.0},
            {'item_id': 3, 'volumen_24h': 1000, 'margen_neto': -3, 'roi_pct': -1.0},
        ])

    def test_filtra_por_volumen_minimo(self):
        filtrado = filtrar_screener_liquido(self._resumen(), volumen_24h_minimo=100)
        assert set(filtrado['item_id']) == {2, 3}

    def test_filtra_tambien_por_margen_neto_minimo(self):
        filtrado = filtrar_screener_liquido(
            self._resumen(), volumen_24h_minimo=100, margen_neto_minimo=0,
        )
        assert set(filtrado['item_id']) == {2}

    def _resumen_con_antiguedad(self, ahora):
        return pd.DataFrame([
            {'item_id': 1, 'volumen_24h': 500, 'margen_neto': 5, 'roi_pct': 2.0,
             'ultimo_timestamp': ahora - 3600},          # 1 hora: fresco
            {'item_id': 2, 'volumen_24h': 500, 'margen_neto': 5, 'roi_pct': 2.0,
             'ultimo_timestamp': ahora - 3 * 86400},     # 3 días: viejo
        ])

    def test_filtra_datos_viejos_cuando_se_pide(self):
        """resumen_actual guarda el último dato DISPONIBLE de cada ítem: para
        uno poco líquido, o después de un corte del recolector, el margen que
        muestra está calculado con un precio de hace días y se presenta como
        si fuera el de ahora."""
        ahora = 1_700_000_000
        filtrado = filtrar_screener_liquido(
            self._resumen_con_antiguedad(ahora), volumen_24h_minimo=100,
            antiguedad_maxima_horas=6, ahora_ts=ahora,
        )
        assert set(filtrado['item_id']) == {1}

    def test_sin_el_parametro_no_filtra_por_antiguedad(self):
        ahora = 1_700_000_000
        filtrado = filtrar_screener_liquido(
            self._resumen_con_antiguedad(ahora), volumen_24h_minimo=100,
        )
        assert set(filtrado['item_id']) == {1, 2}


class TestVentanaPorTiempo:
    """_ventana toma horas de RELOJ, no las últimas N filas: para un ítem con
    huecos, 24 filas pueden abarcar días y el 'volumen de 24h' terminaba
    sumando el de varios, sobrestimando la liquidez justo donde el filtro de
    liquidez es lo único que evita mostrar un ROI absurdo."""

    def _serie_con_hueco(self):
        # 3 filas dentro de las últimas 24h y 3 de hace más de una semana
        ultimo = 1_700_000_000
        timestamps = [ultimo - 8 * 86400, ultimo - 7 * 86400, ultimo - 6 * 86400,
                      ultimo - 2 * 3600, ultimo - 3600, ultimo]
        return pd.DataFrame({'timestamp': timestamps, 'volumen_total': [100] * 6}), ultimo

    def test_solo_cuenta_lo_que_cae_en_la_ventana(self):
        df, ultimo = self._serie_con_hueco()
        assert sum(metricas._ventana(df, 'volumen_total', ultimo, 24)) == 300

    def test_la_version_por_filas_habria_sumado_todo(self):
        df, _ = self._serie_con_hueco()
        assert sum(df['volumen_total'].tolist()[-24:]) == 600
