"""
Tests de configuracion.py — resolución de config.json + variables de
entorno para las credenciales de Telegram y la ruta de la DB (ver el plan
de la fase "app de escritorio v1", workstream 6). Formaliza lo que se
probó a mano durante el desarrollo: persistencia, prioridad de variable de
entorno sobre config.json, y que un archivo corrupto no rompa nada.
"""
import json
import os
import tempfile

import pytest

import configuracion


@pytest.fixture
def config_path():
    path = tempfile.mktemp(suffix='.json')
    yield path
    if os.path.exists(path):
        os.remove(path)


class TestCargarConfiguracion:
    def test_archivo_inexistente_da_dict_vacio(self, config_path):
        assert configuracion.cargar_configuracion(config_path) == {}

    def test_lee_lo_guardado(self, config_path):
        configuracion.guardar_configuracion({'telegram_bot_token': 'ABC'}, config_path)
        assert configuracion.cargar_configuracion(config_path) == {'telegram_bot_token': 'ABC'}

    def test_archivo_corrupto_da_dict_vacio_sin_explotar(self, config_path):
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write('{esto no es json valido')
        assert configuracion.cargar_configuracion(config_path) == {}


class TestGuardarConfiguracion:
    def test_sobreescribe_el_dict_completo(self, config_path):
        configuracion.guardar_configuracion({'a': 1, 'b': 2}, config_path)
        configuracion.guardar_configuracion({'a': 99}, config_path)
        assert configuracion.cargar_configuracion(config_path) == {'a': 99}

    def test_formato_json_legible(self, config_path):
        configuracion.guardar_configuracion({'telegram_chat_id': '123'}, config_path)
        with open(config_path, encoding='utf-8') as f:
            contenido = json.load(f)
        assert contenido == {'telegram_chat_id': '123'}


class TestObtener:
    def test_sin_archivo_ni_env_da_none(self, config_path):
        assert configuracion.obtener('telegram_bot_token', config_path) is None

    def test_lee_desde_config_json(self, config_path):
        configuracion.guardar_configuracion({'db_path': 'otra/ruta.db'}, config_path)
        assert configuracion.obtener('db_path', config_path) == 'otra/ruta.db'

    def test_variable_de_entorno_tiene_prioridad_sobre_config_json(self, config_path, monkeypatch):
        configuracion.guardar_configuracion({'telegram_bot_token': 'DESDE_ARCHIVO'}, config_path)
        monkeypatch.setenv('TELEGRAM_BOT_TOKEN', 'DESDE_ENV')
        assert configuracion.obtener('telegram_bot_token', config_path) == 'DESDE_ENV'

    def test_env_vacia_no_pisa_config_json(self, config_path, monkeypatch):
        # Una variable de entorno seteada pero vacía no debe ganarle a un
        # valor real en config.json -- ver el `if valor:` en obtener().
        configuracion.guardar_configuracion({'telegram_bot_token': 'DESDE_ARCHIVO'}, config_path)
        monkeypatch.setenv('TELEGRAM_BOT_TOKEN', '')
        assert configuracion.obtener('telegram_bot_token', config_path) == 'DESDE_ARCHIVO'

    def test_clave_sin_variable_de_entorno_equivalente(self, config_path):
        # db_path no tiene variable de entorno mapeada en CLAVES -- debe
        # resolver directo desde config.json sin intentar os.environ.get(None).
        configuracion.guardar_configuracion({'db_path': 'z/y.db'}, config_path)
        assert configuracion.obtener('db_path', config_path) == 'z/y.db'
