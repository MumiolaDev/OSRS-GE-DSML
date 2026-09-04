"""
Tests de entrenador.accuracy_direccional y de recolector._ultimo_bucket_cerrado
— las dos correcciones de la revisión que se pueden verificar sin DB ni red.

accuracy_direccional era la métrica que decidía si un modelo "servía", y su
definición estaba rota de dos formas distintas según quién la calculara (ver
el docstring de la función). _ultimo_bucket_cerrado es lo que corrige que la
recolección en vivo fuera dos períodos más vieja de lo necesario.
"""
import numpy as np
import pytest

from entrenador import UMBRAL_CLASIF_PCT, UMBRAL_COSTO_PCT, _direcciones_desde_clases, accuracy_direccional
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
