"""
Tests del veredicto de estado.py — la lógica que decide si el pipeline está
sano. Importa que sea correcta porque es lo único que se ve desde la barra de
estado: un veredicto 'ok' con los datos atrasados es peor que no tener nada.
"""
import estado as modulo_estado


def _estado(antiguedad_1h, corriendo=True):
    return {
        'tablas': {'precios_1h': {'antiguedad_s': antiguedad_1h}},
        'recolector': {'pid': 1, 'origen': 'servicio', 'inicio': 0} if corriendo else None,
    }


def test_datos_frescos_y_recolector_corriendo_es_ok():
    assert modulo_estado._veredicto(_estado(600))['salud'] == 'ok'


def test_sin_ningun_dato_es_error():
    assert modulo_estado._veredicto(_estado(None))['salud'] == 'error'


def test_datos_muy_viejos_son_error_aunque_el_proceso_este_vivo():
    """El caso que motiva todo esto: `systemctl status` verde no significa
    que el pipeline funcione — un loop colgado en una request sin timeout
    deja el proceso 'activo' sin insertar una fila."""
    veredicto = modulo_estado._veredicto(_estado(10 * 3600, corriendo=True))
    assert veredicto['salud'] == 'error'


def test_recolector_detenido_con_datos_frescos_es_atencion_no_error():
    """Recién apagado no es una emergencia: los datos siguen sirviendo."""
    assert modulo_estado._veredicto(_estado(600, corriendo=False))['salud'] == 'atencion'


def test_atraso_moderado_con_el_proceso_vivo_es_atencion():
    assert modulo_estado._veredicto(_estado(3 * 3600))['salud'] == 'atencion'


def test_una_hora_de_atraso_todavia_es_normal():
    """El bucket de 1h se pide a :01 y la API tarda un rato en agregarlo:
    hasta ~2h de antigüedad es operación normal, no un síntoma."""
    assert modulo_estado._veredicto(_estado(3600))['salud'] == 'ok'


def test_el_motivo_nunca_va_vacio():
    for antiguedad in (None, 600, 3 * 3600, 10 * 3600):
        for corriendo in (True, False):
            assert modulo_estado._veredicto(_estado(antiguedad, corriendo))['motivo']
