"""
Tests de entrenador.accuracy_direccional y de recolector._ultimo_bucket_cerrado
— las dos correcciones de la revisión que se pueden verificar sin DB ni red.

accuracy_direccional era la métrica que decidía si un modelo "servía", y su
definición estaba rota de dos formas distintas según quién la calculara (ver
el docstring de la función). _ultimo_bucket_cerrado es lo que corrige que la
recolección en vivo fuera dos períodos más vieja de lo necesario.
"""
import numpy as np
import pandas as pd
import pytest

from entrenador import (
    UMBRAL_CLASIF_PCT, UMBRAL_COSTO_PCT, _direcciones_desde_clases, accuracy_direccional,
    rentabilidad_top_k,
)
from metricas import EXEMPT_ITEM_IDS
from preprocesamiento import _agregar_target_margen
from recolector import _ultimo_bucket_cerrado

HORA = 3600


class TestAccuracyDireccional:
    def test_los_periodos_sin_movimiento_no_cuentan(self):
        """El bug: el retorno real es exactamente 0 entre el 7% y el 35% de
        las horas según el ítem (los precios son enteros y muchos ítems no se
        mueven en una hora). Como el regresor nunca predice exactamente 0,
        `sign(pred) == sign(real)` contaba TODOS esos empates como error."""
        real = np.array([0.0, 0.0, 0.0, 0.02, 0.02])
        pred = np.array([0.001, -0.001, 0.001, 0.03, 0.03])

        acc, n = accuracy_direccional(real, pred)

        assert n == 2  # solo los dos movimientos reales
        assert acc == 1.0  # y en los dos acertó la dirección

    def test_formula_vieja_habria_dado_mucho_menos(self):
        """Comparación explícita contra lo que hacía antes, para que quede
        documentado el tamaño del sesgo."""
        real = np.array([0.0] * 8 + [0.02, 0.02])
        pred = np.array([0.001] * 8 + [0.03, 0.03])

        acc_vieja = float((np.sign(pred) == np.sign(real)).mean())
        acc_nueva, _ = accuracy_direccional(real, pred)

        assert acc_vieja == 0.2
        assert acc_nueva == 1.0

    def test_movimiento_por_debajo_del_umbral_no_cuenta(self):
        real = np.array([0.001, -0.001])  # 0.1%, por debajo del umbral
        acc, n = accuracy_direccional(real, np.array([0.05, 0.05]), umbral_pct=1.0)
        assert n == 0
        assert acc is None

    def test_acierto_parcial(self):
        real = np.array([0.02, -0.02, 0.02, -0.02])
        pred = np.array([0.01, -0.01, -0.01, 0.01])
        acc, n = accuracy_direccional(real, pred)
        assert n == 4
        assert acc == 0.5

    def test_abstenerse_no_cuenta_como_error_pero_achica_la_muestra(self):
        """Abstenerse ('estable') no es equivocarse, es no jugar: esas filas
        salen del denominador. Lo que evita que abstenerse sea un truco para
        inflar el numero es n_evaluado, que se desploma y hace explotar el
        intervalo de confianza de quien muestre la metrica."""
        real = np.array([0.02, 0.02, -0.02, -0.02])
        clases = np.array([2, 1, 0, 1])  # acierta 2, se abstiene en 2

        acc, n = accuracy_direccional(real, _direcciones_desde_clases(clases))

        assert n == 2  # solo los movimientos en los que se jugo
        assert acc == 1.0

    def test_el_regresor_siempre_se_juega(self):
        """Para una prediccion continua, sign(pred) nunca es 0, asi que la
        condicion de 'se jugo' no le cambia nada: n_evaluado son todos los
        movimientos reales."""
        real = np.array([0.02, -0.02, 0.02])
        pred = np.array([0.001, -0.001, -0.001])

        acc, n = accuracy_direccional(real, pred)

        assert n == 3
        assert acc == pytest.approx(2 / 3)

    def test_un_clasificador_que_siempre_se_abstiene_no_tiene_metrica(self):
        real = np.array([0.02, -0.02])
        acc, n = accuracy_direccional(real, _direcciones_desde_clases([1, 1]))
        assert n == 0
        assert acc is None

    def test_direcciones_desde_clases(self):
        assert list(_direcciones_desde_clases([0, 1, 2])) == [-1.0, 0.0, 1.0]


class TestUmbralVsCosto:
    def test_el_umbral_de_costo_es_el_impuesto_ge(self):
        assert UMBRAL_COSTO_PCT == 2.0

    def test_el_umbral_por_default_sigue_por_debajo_del_costo(self):
        """Documentado a propósito: con 2% el clasificador colapsa a predecir
        'estable' siempre (entre el 1% y el 10% de las horas superan ese
        movimiento). Que la señal cubra el costo se decide al consumirla,
        cruzándola con el margen del screener — pero entrenador.py loguea un
        warning para que no sea un default silencioso."""
        assert UMBRAL_CLASIF_PCT < UMBRAL_COSTO_PCT


class TestUltimoBucketCerrado:
    def test_hora_en_curso_no_cuenta(self):
        # 13:39:42 UTC -> el último bucket horario cerrado es el de las 12:00
        ahora = 13 * HORA + 39 * 60 + 42
        assert _ultimo_bucket_cerrado(HORA, ahora) == 12 * HORA

    def test_justo_al_cerrar_devuelve_el_recien_cerrado(self):
        ahora = 13 * HORA  # 13:00:00 en punto
        assert _ultimo_bucket_cerrado(HORA, ahora) == 12 * HORA

    def test_paso_de_cinco_minutos(self):
        # 13:39:42 -> el bucket de las 13:35 todavía está EN CURSO (cierra a
        # las 13:40), así que el último cerrado es el de las 13:30.
        ahora = 13 * HORA + 39 * 60 + 42
        assert _ultimo_bucket_cerrado(300, ahora) == 13 * HORA + 30 * 60

    def test_paso_de_cinco_minutos_justo_despues_de_cerrar(self):
        ahora = 13 * HORA + 40 * 60 + 5
        assert _ultimo_bucket_cerrado(300, ahora) == 13 * HORA + 35 * 60

    def test_paso_de_seis_horas(self):
        ahora = 13 * HORA + 40 * 60
        assert _ultimo_bucket_cerrado(6 * HORA, ahora) == 6 * HORA

    def test_siempre_devuelve_un_multiplo_del_paso(self):
        for paso in (300, HORA, 6 * HORA):
            for ahora in (1_788_529_182, 1_788_529_200, 1_788_500_000):
                assert _ultimo_bucket_cerrado(paso, ahora) % paso == 0


class TestRentabilidadTopK:
    """Métrica de calidad de un modelo de spread: de los ítems que eligió
    cada período, ¿cuántos terminaron con margen positivo? Se mide sobre el
    ranking y no sobre el universo entero porque la decisión real es comprar
    unos pocos, no todos."""

    def _panel(self):
        # dos períodos, cuatro ítems cada uno. El margen REAL baja con el
        # item_id; la predicción del "modelo" está bien ordenada en el
        # primer período y al revés en el segundo.
        return pd.DataFrame({
            'timestamp_target': [1, 1, 1, 1, 2, 2, 2, 2],
            'pred_margen':      [9, 8, 7, 6, 6, 7, 8, 9],
            'target_margen':    [0.05, 0.02, -0.01, -0.03, 0.05, 0.02, -0.01, -0.03],
        })

    def test_elige_los_mejores_de_cada_periodo(self):
        fraccion, n, margen_medio = rentabilidad_top_k(self._panel(), top_k=2)
        # periodo 1: elige los dos de arriba (+0.05, +0.02), los dos rentables
        # periodo 2: la prediccion esta invertida, elige (-0.03, -0.01)
        assert n == 4
        assert fraccion == 0.5
        assert margen_medio == pytest.approx((0.05 + 0.02 - 0.03 - 0.01) / 4)

    def test_top_k_mas_grande_que_el_universo_no_rompe(self):
        fraccion, n, _ = rentabilidad_top_k(self._panel(), top_k=99)
        assert n == 8
        assert fraccion == 0.5

    def test_panel_vacio(self):
        vacio = pd.DataFrame(columns=['timestamp_target', 'pred_margen', 'target_margen'])
        assert rentabilidad_top_k(vacio) == (None, 0, None)


class TestTargetMargen:
    """El target que reemplazó al de "¿sube el precio?": el margen neto que
    deja comprar en la punta baja y vender en la alta del período siguiente,
    ya descontado el impuesto del GE."""

    def test_margen_descuenta_el_impuesto(self):
        ds = pd.DataFrame({
            'item_id': [1000],
            'price_target': [1000.0],       # avg_low del periodo siguiente
            'price_target_high': [1100.0],  # avg_high del periodo siguiente
        })
        _agregar_target_margen(ds)
        # impuesto = floor(1100 * 0.02) = 22  ->  (1100 - 1000 - 22) / 1000
        assert ds['target_margen'].iloc[0] == pytest.approx(0.078)

    def test_item_exento_no_paga_impuesto(self):
        exento = sorted(EXEMPT_ITEM_IDS)[0]
        ds = pd.DataFrame({
            'item_id': [exento], 'price_target': [1000.0], 'price_target_high': [1100.0],
        })
        _agregar_target_margen(ds)
        assert ds['target_margen'].iloc[0] == pytest.approx(0.1)

    def test_margen_negativo_cuando_el_spread_no_cubre_el_impuesto(self):
        ds = pd.DataFrame({
            'item_id': [1000], 'price_target': [1000.0], 'price_target_high': [1010.0],
        })
        _agregar_target_margen(ds)
        # spread de 10 gp contra 20 gp de impuesto: perder plata
        assert ds['target_margen'].iloc[0] < 0
