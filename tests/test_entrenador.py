"""
Tests de _clasificar_retorno (entrenador.py) — convierte el log-retorno
continuo en las 3 clases del clasificador direccional (0=baja, 1=estable,
2=sube). Si el umbral se aplica mal, el clasificador aprende sobre
etiquetas incorrectas sin que nada lo avise.
"""
import numpy as np

from entrenador import _clasificar_retorno


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
