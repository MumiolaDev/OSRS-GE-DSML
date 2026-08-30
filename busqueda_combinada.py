"""
busqueda_combinada.py — último experimento de la noche (recomendación #5 de
docs/investigacion_calidad_modelos.md): busqueda_calibrada.py probó umbral
calibrado + dimensionamiento por confianza + salida por triple barrera cada
uno POR SEPARADO contra la salida/dimensionamiento de siempre. Acá se prueban
LAS TRES JUNTAS sobre los dos ítems donde alguna mejora individual dio
resultado (Kwuarm: triple barrera casi duplicó el resultado; Bolt of canvas:
confianza y triple barrera mejoraron ambas) — para ver si se combinan o se
pisan entre sí.

Reentrena el clasificador walk-forward una vez más por ítem (con el mismo
umbral calibrado ya encontrado anoche, hardcodeado abajo — no hace falta
recalibrarlo) porque prob_sube solo vive en memoria durante esa corrida, no
se persiste en la DB. Resultados a resultados_combinada.csv.
"""
import logging

import pandas as pd

from base_de_datos import OSRSBaseDatos
from replay_historico import ejecutar_replay_clasificador
from busqueda_hiperparametros import _rango_backtest, backtest_capital_constante, TABLA, CAPITAL_INICIAL
from busqueda_calibrada import (
    VENTANA_DIAS, STOP_LOSS_FRACCION, TIMEOUT_PASOS,
    _senales_triple_barrera, _fraccion_por_confianza,
)

DB_PATH = 'data/osrs_ge.db'
CSV_RESULTADOS = 'resultados_combinada.csv'

# Umbrales calibrados ya encontrados anoche (busqueda_calibrada.py) -- no
# hace falta recalcularlos, solo reentrenar para tener prob_sube en memoria.
ITEMS = [
    {'item_id': 263, 'nombre': 'Kwuarm', 'umbral_pct': 0.9512},
    {'item_id': 31475, 'nombre': 'Bolt of canvas', 'umbral_pct': 6.9437},
]


def probar_item(db, item, desde_ts, hasta_ts):
    item_id, nombre, umbral_pct = item['item_id'], item['nombre'], item['umbral_pct']
    model_id = f"combinada_{item_id}_clasif"
    logging.info(f"--- {nombre} ({item_id}): reentrenando clasificador (umbral {umbral_pct}%) para tener prob_sube ---")
    test_acumulado = ejecutar_replay_clasificador(
        db, desde_ts, hasta_ts, model_name=model_id, umbral_pct=umbral_pct,
        tabla=TABLA, ventana_dias=VENTANA_DIAS, item_ids=[item_id],
    )
    if test_acumulado.empty:
        logging.warning(f"{nombre}: walk-forward vacío, se omite.")
        return []

    candidatos = test_acumulado[test_acumulado['clase_predicha'] == 2].copy()
    if candidatos.empty:
        logging.warning(f"{nombre}: sin señales 'sube', se omite.")
        return []
    candidatos['ts_salida'] = candidatos['timestamp_target'].astype(int)
    candidatos['ts_entrada'] = candidatos['ts_salida'] - 3600
    candidatos['item_id'] = candidatos['item_id'].astype(int)
    señales_base = candidatos[['item_id', 'ts_entrada', 'ts_salida']].copy()
    señales_base['magnitud'] = 1.0  # placeholder, se sobreescribe según la variante

    mapa_prob = candidatos.set_index(candidatos['ts_salida'])['prob_sube']

    filas = []

    # (a) baseline: salida fija, fracción fija (ya conocido de busqueda_calibrada.py, para referencia)
    trades, resumen = backtest_capital_constante(db, TABLA, señales_base, CAPITAL_INICIAL)
    filas.append({'item_id': item_id, 'nombre': nombre, 'variante': 'baseline_fijo_fijo', **resumen})

    # (b) solo confianza (salida fija, fracción por prob_sube)
    señales_conf = señales_base.copy()
    señales_conf['magnitud'] = señales_conf['ts_salida'].map(mapa_prob).fillna(0.5)
    trades, resumen = backtest_capital_constante(db, TABLA, señales_conf, CAPITAL_INICIAL, fraccion_fn=_fraccion_por_confianza)
    filas.append({'item_id': item_id, 'nombre': nombre, 'variante': 'solo_confianza', **resumen})

    # (c) solo triple barrera (fracción fija, salida dinámica)
    señales_tb = _senales_triple_barrera(db, TABLA, señales_base, umbral_pct, STOP_LOSS_FRACCION, TIMEOUT_PASOS)
    trades, resumen = backtest_capital_constante(db, TABLA, señales_tb, CAPITAL_INICIAL)
    filas.append({'item_id': item_id, 'nombre': nombre, 'variante': 'solo_triple_barrera', **resumen})

    # (d) las tres combinadas: triple barrera resuelve ts_salida, y la magnitud (prob_sube, tomada en
    # el momento de ENTRADA -- la única disponible al decidir la compra) dimensiona por confianza.
    señales_comb = señales_tb.copy()
    if not señales_comb.empty:
        # prob_sube está indexado por ts_salida ORIGINAL (ts_entrada+3600) en mapa_prob -- para
        # la variante combinada se busca por ts_entrada+3600 (el momento en que el clasificador
        # emitió la señal), no por el ts_salida ya reresuelto por la barrera.
        señales_comb['magnitud'] = (señales_comb['ts_entrada'] + 3600).map(mapa_prob).fillna(0.5)
    trades, resumen = backtest_capital_constante(db, TABLA, señales_comb, CAPITAL_INICIAL, fraccion_fn=_fraccion_por_confianza)
    filas.append({'item_id': item_id, 'nombre': nombre, 'variante': 'combinado_confianza_triple_barrera', **resumen})

    for f in filas:
        logging.info(f"[{nombre}/{f['variante']}] ganancia={f['ganancia']} ({f['ganancia_pct']:.2f}%) trades={f['n_trades']} win_rate={f['win_rate']}")
    return filas


def ejecutar_busqueda_combinada():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s |--| %(levelname)s |--| %(message)s')
    db = OSRSBaseDatos(DB_PATH)
    desde_ts, hasta_ts = _rango_backtest(db, TABLA)
    logging.info(f"=== Búsqueda combinada: {len(ITEMS)} ítems, backtest {desde_ts}-{hasta_ts} ===")

    resultados = []
    for item in ITEMS:
        try:
            filas = probar_item(db, item, desde_ts, hasta_ts)
            resultados.extend(filas)
            pd.DataFrame(resultados).to_csv(CSV_RESULTADOS, index=False)
        except Exception as e:
            logging.error(f"Ítem {item['nombre']} falló: {e}", exc_info=True)

    df = pd.DataFrame(resultados)
    df.to_csv(CSV_RESULTADOS, index=False)
    logging.info(f"\n=== Búsqueda combinada completa: {len(df)} filas en {CSV_RESULTADOS} ===")
    if not df.empty:
        logging.info("\n" + df.to_string(index=False))
    return df


if __name__ == '__main__':
    ejecutar_busqueda_combinada()
