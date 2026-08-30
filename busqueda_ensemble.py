"""
busqueda_ensemble.py — variante de busqueda_hiperparametros.py: en vez de
hacer competir regresor vs clasificador (cuál da más plata solo), los
combina. El clasificador aporta la DIRECCIÓN (sube/baja/estable); el
regresor aporta la MAGNITUD estimada del movimiento — algo de lo que el
clasificador no tiene ninguna noción. La señal combinada ("ensemble")
solo dispara una compra cuando AMBOS coinciden: el clasificador predijo
'sube' Y el regresor predijo un retorno por encima del mismo umbral —
la idea es que un modelo confirme al otro, no que reemplace al otro.

Corre sobre ÍTEMS INDIVIDUALES (no un grupo pooleado como la corrida
anterior) con precio > 1000gp y volumen alto — a diferencia de las runas/
ítems F2P baratos de la corrida anterior (donde el impuesto GE se
redondea a 0 y el ROI puede ser ruido), acá el impuesto es real y el
buy_limit/volumen sí restringen de verdad cuánto se puede operar: un test
más representativo de lo que sería operar en la práctica.

Para cada ítem, para cada umbral de UMBRALES, entrena walk-forward el
clasificador (con ese umbral) y el regresor (una sola vez por ítem, no
depende de umbral) sobre las últimas 2 semanas, y evalúa TRES métodos:
'clasificador' (solo), 'regresor' (solo, filtrado por el mismo umbral
como magnitud mínima) y 'ensemble' (intersección de ambos). Cada uno se
prueba con el mismo backtest_capital_constante() de
busqueda_hiperparametros.py.

Mismo criterio que la búsqueda anterior: nunca escribe en modelos_config
del usuario ni toca ningún .pkl productivo, resultados a
resultados_ensemble.csv fila por fila.
"""
import logging
import time

import pandas as pd

from base_de_datos import OSRSBaseDatos
from replay_historico import ejecutar_replay_modelo, ejecutar_replay_clasificador
from busqueda_hiperparametros import (
    _rango_backtest, _senales_regresor, _senales_clasificador,
    backtest_capital_constante, TABLA, CAPITAL_INICIAL,
)

DB_PATH = 'data/osrs_ge.db'
CSV_RESULTADOS = 'resultados_ensemble.csv'
VENTANA_DIAS = 90
UMBRALES = [0.8, 1.0, 1.2]  # "el umbral 0.8 y un par más alto", pedido del usuario

# Ítems individuales con precio > 1000gp y volumen alto (ver el pedido del
# usuario) -- elegidos de resumen_actual real, tres categorías de juego
# distintas (smithing, mining, prayer) para no probar solo un tipo.
ITEMS = [
    {'item_id': 2361, 'nombre': 'Adamantite bar'},
    {'item_id': 451, 'nombre': 'Runite ore'},
    {'item_id': 536, 'nombre': 'Dragon bones'},
]


def _senal_ensemble(senales_clasif, senales_regresor):
    """
    Intersección de ambas señales por (item_id, ts_entrada): solo queda
    una señal de compra si el clasificador dijo 'sube' Y el regresor
    predijo una suba por encima del umbral EN EL MISMO checkpoint. La
    magnitud para priorizar/dimensionar viene del regresor (es la única
    fuente que estima "cuánto", ver el pedido del usuario) -- el
    clasificador solo aporta el filtro de "sí/no".
    """
    if senales_clasif.empty or senales_regresor.empty:
        return pd.DataFrame(columns=['item_id', 'ts_entrada', 'ts_salida', 'magnitud'])
    combinado = senales_clasif.merge(
        senales_regresor[['item_id', 'ts_entrada', 'magnitud']],
        on=['item_id', 'ts_entrada'], suffixes=('', '_regresor'),
    )
    if combinado.empty:
        return pd.DataFrame(columns=['item_id', 'ts_entrada', 'ts_salida', 'magnitud'])
    combinado['magnitud'] = combinado['magnitud_regresor']
    return combinado[['item_id', 'ts_entrada', 'ts_salida', 'magnitud']]


def _entrenar_temporal(db, model_id, item_id, tipo, umbral_pct=None):
    """Crea un modelo temporal de un solo ítem en modelos_config (hace
    falta para ejecutar_replay_modelo), lo entrena walk-forward, y lo
    borra al terminar -- mismo patrón que busqueda_hiperparametros.
    probar_regresor. Devuelve el resultado de ejecutar_replay_modelo."""
    db.crear_modelo_config(
        model_id=model_id, nombre=f"[ensemble] {model_id}", tipo=tipo,
        cadencia='manual', modo_seleccion='manual', item_ids=[item_id],
        tabla=TABLA, ventana_dias=VENTANA_DIAS, umbral_pct=umbral_pct,
    )
    desde_ts, hasta_ts = _rango_backtest(db, TABLA)
    t0 = time.time()
    ejecutar_replay_modelo(db, model_id, desde_ts=desde_ts, hasta_ts=hasta_ts)
    duracion = time.time() - t0
    db.eliminar_modelo_config(model_id)  # limpia el registro, conserva model_metrics/predicciones
    return desde_ts, hasta_ts, duracion


def probar_item(db, item):
    item_id, nombre = item['item_id'], item['nombre']
    desde_ts, hasta_ts = _rango_backtest(db, TABLA)
    filas = []

    # Regresor: un solo entrenamiento por ítem, no depende de umbral.
    model_id_reg = f"ensemble_{item_id}_regresor"
    logging.info(f"--- {nombre} ({item_id}): entrenando regresor ---")
    _, _, dur_reg = _entrenar_temporal(db, model_id_reg, item_id, 'regresor')
    señales_reg_base = _senales_regresor(db, model_id_reg, TABLA, desde_ts, hasta_ts, umbral_subida_pct=0.0)

    for umbral_pct in UMBRALES:
        logging.info(f"--- {nombre} ({item_id}): entrenando clasificador (umbral {umbral_pct}%) ---")
        model_id_clasif = f"ensemble_{item_id}_clasif_u{umbral_pct}"
        test_acumulado = ejecutar_replay_clasificador(
            db, desde_ts, hasta_ts, model_name=model_id_clasif, umbral_pct=umbral_pct,
            tabla=TABLA, ventana_dias=VENTANA_DIAS, item_ids=[item_id],
        )
        señales_clasif = _senales_clasificador(test_acumulado)
        señales_reg_filtradas = señales_reg_base[señales_reg_base['magnitud'] >= umbral_pct]
        señales_ensemble = _senal_ensemble(señales_clasif, señales_reg_filtradas)

        for metodo, señales in (
            ('clasificador', señales_clasif),
            ('regresor', señales_reg_filtradas),
            ('ensemble', señales_ensemble),
        ):
            trades, resumen = backtest_capital_constante(db, TABLA, señales, CAPITAL_INICIAL)
            fila = {
                'item_id': item_id, 'nombre': nombre, 'umbral_pct': umbral_pct, 'metodo': metodo,
                **resumen,
            }
            filas.append(fila)
            logging.info(f"Resultado: {fila}")

    return filas


def ejecutar_busqueda_ensemble():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s |--| %(levelname)s |--| %(message)s')
    db = OSRSBaseDatos(DB_PATH)

    resultados = []
    for item in ITEMS:
        filas = probar_item(db, item)
        resultados.extend(filas)
        pd.DataFrame(resultados).to_csv(CSV_RESULTADOS, index=False)

    df = pd.DataFrame(resultados)
    df.to_csv(CSV_RESULTADOS, index=False)
    logging.info(f"\n=== Búsqueda ensemble completa: {len(df)} filas en {CSV_RESULTADOS} ===")
    if not df.empty:
        logging.info("\n" + df.sort_values('ganancia', ascending=False).to_string(index=False))
    return df


if __name__ == '__main__':
    ejecutar_busqueda_ensemble()
