"""
Tests de prediccion.cargar_modelo — resuelve la ruta del bundle guardado
por entrenador.py. Bug real encontrado en la revisión final de esta rama:
indexaba entrenador.MODEL_PATHS directo (`MODEL_PATHS[model_name]`, que
solo tiene 4 entradas fijas — los 2 regresores y 2 variantes del
clasificador F2P), así que cualquier modelo creado por el usuario desde
la app de escritorio (que entrenador.py SÍ guarda bien, vía
`MODEL_PATHS.get(model_name, .../f"model_{model_name}.pkl")` con
fallback) tiraba KeyError al intentar leerlo de vuelta —
`pagina_oportunidades.py` lo mostraba como "no se pudo cargar el modelo
'custom_x': 'custom_x'" (el `str(KeyError(...))` es justo el nombre entre
comillas), no como "todavía no se entrenó". Estos tests cubren los dos
casos (nombre conocido vía MODEL_PATHS, nombre custom vía el fallback) y
el caso sin archivo, para que una regresión futura de este tipo la agarre
un test, no un usuario viendo "Oportunidades" vacío en silencio.
"""
import joblib
import pytest

import entrenador
from preprocesamiento import FEATURES_VERSION
from prediccion import cargar_modelo


@pytest.fixture
def model_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(entrenador, 'MODEL_DIR', str(tmp_path))
    return tmp_path


def _bundle(marca, version=FEATURES_VERSION):
    return {'marca': marca, 'features_version': version}


def test_nombre_conocido_usa_model_paths(model_dir, monkeypatch):
    ruta = model_dir / 'un_bundle_conocido.pkl'
    joblib.dump(_bundle('conocido'), ruta)
    monkeypatch.setitem(entrenador.MODEL_PATHS, 'mi_modelo_conocido', str(ruta))

    bundle = cargar_modelo(model_name='mi_modelo_conocido')

    assert bundle['marca'] == 'conocido'


def test_nombre_custom_usa_el_fallback_de_model_dir(model_dir):
    # Mismo patrón de nombre que usa entrenador.py al guardar: model_{model_name}.pkl
    ruta = model_dir / 'model_custom_x_clasificador.pkl'
    joblib.dump(_bundle('custom'), ruta)

    bundle = cargar_modelo(model_name='custom_x_clasificador')

    assert bundle['marca'] == 'custom'


def test_nombre_custom_sin_pkl_lanza_filenotfound(model_dir):
    with pytest.raises(FileNotFoundError):
        cargar_modelo(model_name='no_se_entreno_todavia')


def test_bundle_de_otra_version_de_features_no_se_carga(model_dir):
    """
    Un .pkl entrenado con el esquema de features viejo (niveles absolutos,
    features_version=1) tiene que fallar con un mensaje claro, no predecir
    ruido en silencio: _preparar_fila_prediccion arma la fila con
    reindex(columns=bundle['feature_cols']), que rellena con NaN las
    columnas que ya no existen, y XGBoost acepta NaN sin protestar.
    """
    ruta = model_dir / 'model_viejo.pkl'
    joblib.dump(_bundle('viejo', version=FEATURES_VERSION - 1), ruta)

    with pytest.raises(ValueError, match='features_version'):
        cargar_modelo(model_name='viejo')


def test_bundle_sin_features_version_se_trata_como_version_1(model_dir):
    """Los bundles anteriores a que existiera la clave no la tienen: se
    asumen versión 1 (features en niveles) y se rechazan igual."""
    ruta = model_dir / 'model_sin_version.pkl'
    joblib.dump({'marca': 'sin_version'}, ruta)

    with pytest.raises(ValueError, match='features_version=1'):
        cargar_modelo(model_name='sin_version')
