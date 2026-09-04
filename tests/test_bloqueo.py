"""
Tests del candado de instancia única (bloqueo.py) — lo que impide que el
servicio de systemd y la app de escritorio recolecten los dos a la vez sobre
la misma DB (duplicando requests a la API y pisándose el .pkl al reentrenar).

No tocan la DB ni la red: el candado es un archivo con flock.
"""
import multiprocessing
import os
import tempfile

import pytest

from bloqueo import (
    ORIGEN_APP, ORIGEN_SERVICIO, BloqueoRecolector, hay_recolector_corriendo,
    leer_info_bloqueo, ruta_bloqueo,
)


@pytest.fixture
def db_path():
    directorio = tempfile.mkdtemp()
    yield os.path.join(directorio, 'osrs_ge.db')


def test_el_candado_vive_al_lado_de_la_db(db_path):
    assert ruta_bloqueo(db_path) == db_path + '.lock'


def test_se_adquiere_si_esta_libre(db_path):
    bloqueo = BloqueoRecolector(db_path, origen=ORIGEN_SERVICIO)
    assert bloqueo.adquirir() is True
    bloqueo.liberar()


def test_un_segundo_recolector_no_puede_adquirirlo(db_path):
    primero = BloqueoRecolector(db_path, origen=ORIGEN_SERVICIO)
    assert primero.adquirir() is True
    segundo = BloqueoRecolector(db_path, origen=ORIGEN_APP)
    try:
        assert segundo.adquirir() is False
        # Y tiene que poder decir QUIÉN lo tiene, para que la app muestre
        # "ya corre como servicio" en vez de un error opaco.
        assert segundo.duenio_previo['origen'] == ORIGEN_SERVICIO
        assert segundo.duenio_previo['pid'] == os.getpid()
    finally:
        primero.liberar()


def test_despues_de_liberar_se_puede_volver_a_tomar(db_path):
    primero = BloqueoRecolector(db_path)
    primero.adquirir()
    primero.liberar()
    segundo = BloqueoRecolector(db_path)
    assert segundo.adquirir() is True
    segundo.liberar()


def test_dos_dbs_distintas_no_se_estorban(db_path):
    """Una DB de pruebas y la productiva son dos recolectores legítimamente
    simultáneos, no un conflicto."""
    otra = db_path.replace('osrs_ge.db', 'otra.db')
    a, b = BloqueoRecolector(db_path), BloqueoRecolector(otra)
    assert a.adquirir() is True
    assert b.adquirir() is True
    a.liberar()
    b.liberar()


def test_sin_archivo_no_hay_recolector(db_path):
    assert hay_recolector_corriendo(db_path) is None


def test_hay_recolector_lo_detecta_sin_robarle_el_candado(db_path):
    bloqueo = BloqueoRecolector(db_path, origen=ORIGEN_SERVICIO)
    bloqueo.adquirir()
    try:
        info = hay_recolector_corriendo(db_path)
        assert info is not None
        assert info['origen'] == ORIGEN_SERVICIO
        # El chequeo no debe soltar el candado del dueño: preguntarlo dos
        # veces tiene que dar lo mismo.
        assert hay_recolector_corriendo(db_path) is not None
    finally:
        bloqueo.liberar()
    assert hay_recolector_corriendo(db_path) is None


def _tomar_y_esperar(db_path, tomado, seguir):
    bloqueo = BloqueoRecolector(db_path, origen=ORIGEN_SERVICIO)
    bloqueo.adquirir()
    tomado.set()
    seguir.wait(timeout=10)
    # Sale SIN liberar a propósito: simula un kill -9 / corte de luz.
    os._exit(0)


def test_el_candado_muere_con_el_proceso(db_path):
    """
    La razón de usar flock y no un archivo de PID: el kernel lo suelta solo
    cuando el proceso muere, aunque muera de la peor forma. Si esto fallara,
    un recolector matado a la fuerza dejaría la app convencida para siempre
    de que hay otro corriendo.
    """
    ctx = multiprocessing.get_context('spawn')
    tomado, seguir = ctx.Event(), ctx.Event()
    proceso = ctx.Process(target=_tomar_y_esperar, args=(db_path, tomado, seguir))
    proceso.start()
    try:
        assert tomado.wait(timeout=15), "el subproceso no llegó a tomar el candado"
        assert hay_recolector_corriendo(db_path) is not None
        assert hay_recolector_corriendo(db_path)['pid'] == proceso.pid
    finally:
        seguir.set()
        proceso.join(timeout=15)
    assert hay_recolector_corriendo(db_path) is None


def test_un_archivo_corrupto_no_rompe_la_lectura(db_path):
    with open(ruta_bloqueo(db_path), 'w', encoding='utf-8') as f:
        f.write('esto no es json')
    assert leer_info_bloqueo(ruta_bloqueo(db_path)) is None
    # Y el candado en sí sigue tomándose sin problema: la exclusión la da el
    # flock, no el contenido del archivo.
    bloqueo = BloqueoRecolector(db_path)
    assert bloqueo.adquirir() is True
    bloqueo.liberar()


def test_intentar_tomarlo_no_le_borra_la_info_al_dueno(db_path):
    """Abrir el archivo en modo 'w' lo truncaría ANTES de saber si el
    candado está libre — el que sí lo tiene se quedaría sin su JSON de
    dueño y la app no podría decir quién está recolectando."""
    dueno = BloqueoRecolector(db_path, origen=ORIGEN_SERVICIO)
    dueno.adquirir()
    try:
        BloqueoRecolector(db_path, origen=ORIGEN_APP).adquirir()
        assert leer_info_bloqueo(ruta_bloqueo(db_path))['origen'] == ORIGEN_SERVICIO
    finally:
        dueno.liberar()


def test_como_context_manager_falla_ruidosamente_si_ya_esta_tomado(db_path):
    primero = BloqueoRecolector(db_path)
    primero.adquirir()
    try:
        with pytest.raises(RuntimeError):
            with BloqueoRecolector(db_path):
                pass
    finally:
        primero.liberar()
