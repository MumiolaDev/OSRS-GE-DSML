"""
paginas/pagina_oportunidades.py — vista principal de la app: qué ítems
conviene mirar ahora mismo. Fusiona lo que en el dashboard de Streamlit
eran dos pestañas separadas (Screener + Señal direccional F2P) en una
sola tabla curada — ver la corrección del usuario sobre simplificar la
app en vez de portar el dashboard tal cual.

Columnas curadas de resumen_actual (metricas.py) + el MARGEN PREDICHO para
la próxima hora, calculado en vivo con los modelos de tipo 'spread' activos
del usuario (base_de_datos.modelos_config). Esa es la columna por la que la
tabla ordena por default, y es la que responde "qué comprar": el margen de
resumen_actual es el que ya se vio y puede haber desaparecido, mientras que
este es el que el modelo espera que exista cuando efectivamente se pueda
operar. Los ítems que ningún modelo cubre quedan al final con un guión.

Antes esta columna era una señal direccional (sube/estable/baja) de un
clasificador; se reemplazó junto con el tipo de modelo — ver el docstring de
entrenador.entrenar_modelo_spread para la medición que lo justifica.
"""
import logging
import sqlite3

import pandas as pd
from PySide6.QtWidgets import (
    QComboBox, QHBoxLayout, QLabel, QPushButton, QSpinBox, QVBoxLayout, QWidget,
)

from metricas import filtrar_screener_liquido
from prediccion import cargar_modelo, pronosticar_margen_items
from escritorio.widgets.tabla_dataframe import crear_tabla

COLUMNAS = [
    ('name', 'Ítem'),
    ('margen_neto', 'Margen neto'),
    ('roi_pct', 'ROI'),
    ('buy_limit', 'Límite de compra'),
    ('volumen_24h', 'Volumen 24h'),
    ('margen_predicho', 'Margen predicho (próx. hora)'),
]

FORMATOS = {
    'margen_neto': lambda v: f"{int(v):,} gp".replace(',', '.'),
    'roi_pct': lambda v: f"{float(v):.1f}%",
    'buy_limit': lambda v: f"{int(v):,}".replace(',', '.'),
    'volumen_24h': lambda v: f"{int(v):,}".replace(',', '.'),
    'margen_predicho': lambda v: '—' if v != v else f"{v:,.0f} gp".replace(',', '.'),
}

ORDEN_OPCIONES = [
    ('margen_predicho', 'Margen predicho (recomendado)'),
    ('margen_neto', 'Margen neto'),
    ('roi_pct', 'ROI'),
    ('volumen_24h', 'Volumen 24h'),
]

# Un ítem cuyo último dato de precio es más viejo que esto no describe el
# mercado de ahora: su margen/ROI se calcularon con un precio de hace horas
# o días. Mismo criterio que alertas.py.
ANTIGUEDAD_MAXIMA_HORAS = 6


def _calcular_margenes_predichos(db, item_ids_screener):
    """
    Para cada modelo activo de tipo='spread' en modelos_config, resuelve su
    universo real de ítems (cfg['item_ids'] directo si
    modo_seleccion='manual', o db.obtener_top_items_liquidez con sus
    parámetros si 'liquidez') y calcula, en vivo, el margen neto que espera
    del próximo período (prediccion.pronosticar_margen_items) SOLO para los
    ítems que además están en `item_ids_screener` — no tiene sentido correr
    inferencia sobre ítems que ni van a aparecer en la tabla.

    Esto es lo que responde "qué comprar": a diferencia del margen de
    `resumen_actual`, que es el que YA se vio (y para cuando se muestra puede
    haber desaparecido), esta columna es el que el modelo espera que exista
    en la próxima hora, que es cuando efectivamente se puede operar.

    Devuelve {item_id: margen_predicho_gp}. Si un ítem está cubierto por más
    de un modelo activo, gana el más reciente (se procesan del más viejo al
    más nuevo, y el último en escribir pisa).
    """
    item_ids_screener = set(item_ids_screener)
    margenes = {}

    modelos = [m for m in db.listar_modelos_config(estado='activo') if m['tipo'] == 'spread']
    modelos.sort(key=lambda m: m['creado_en'] or 0)

    for cfg in modelos:
        if cfg['modo_seleccion'] == 'manual':
            universo = cfg['item_ids'] or []
        else:
            universo = db.obtener_top_items_liquidez(
                cfg['n_items'], solo_f2p=cfg['solo_f2p'],
                precio_minimo=cfg['precio_minimo'], excluir_item_ids=cfg['excluir_item_ids'],
            )
        candidatos = [i for i in universo if i in item_ids_screener]
        if not candidatos:
            continue

        try:
            bundle = cargar_modelo(model_name=cfg['model_id'])
        except FileNotFoundError:
            continue  # todavía no se entrenó/guardó un .pkl para este modelo
        except Exception as e:
            logging.warning(f"pagina_oportunidades: no se pudo cargar el modelo '{cfg['model_id']}': {e}")
            continue

        pronostico = pronosticar_margen_items(db, candidatos, bundle)
        for _, fila in pronostico.iterrows():
            margenes[int(fila['item_id'])] = float(fila['margen_pred_gp'])

    return margenes


class PaginaOportunidades(QWidget):
    def __init__(self, db, parent=None):
        super().__init__(parent)
        self.db = db

        layout = QVBoxLayout(self)

        fila_filtros = QHBoxLayout()
        fila_filtros.addWidget(QLabel("Volumen 24h mínimo:"))
        self.spin_volumen = QSpinBox()
        self.spin_volumen.setRange(0, 100_000)
        self.spin_volumen.setValue(100)
        self.spin_volumen.setSingleStep(10)
        fila_filtros.addWidget(self.spin_volumen)

        fila_filtros.addWidget(QLabel("Ordenar por:"))
        self.combo_orden = QComboBox()
        for clave, etiqueta in ORDEN_OPCIONES:
            self.combo_orden.addItem(etiqueta, clave)
        fila_filtros.addWidget(self.combo_orden)

        boton_actualizar = QPushButton("Actualizar")
        boton_actualizar.clicked.connect(self._refrescar)
        fila_filtros.addWidget(boton_actualizar)
        fila_filtros.addStretch()
        layout.addLayout(fila_filtros)

        self.label_contador = QLabel("")
        layout.addWidget(self.label_contador)

        self.tabla, self.modelo_tabla = crear_tabla(columnas=COLUMNAS, formatos=FORMATOS)
        layout.addWidget(self.tabla)

        self._refrescar()

    def _refrescar(self):
        """
        Refresco manual, no automático — con datos horarios no se justifica
        un auto-refresh en vivo (mismo criterio que ya usaba dashboard.py).
        Envuelto en try/except: una falla acá (ej. la DB bloqueada un
        instante por el recolector, o un modelo con un .pkl corrupto) no
        debe dejar la pestaña rota, solo avisar y mantener lo que ya
        estaba mostrando.
        """
        try:
            # resumen_actual no tiene buy_limit (vive en items, no es parte
            # del screener calculado) -- mismo JOIN que ya usa alertas.py
            # para anotarlo. Sin este JOIN, la columna "Límite de compra"
            # desaparece en silencio (crear_tabla descarta columnas
            # curadas que no están en el DataFrame en vez de fallar) en
            # vez de mostrar un error.
            conn = sqlite3.connect(self.db.db_path)
            resumen = pd.read_sql_query(
                "SELECT r.*, i.buy_limit FROM resumen_actual r JOIN items i ON i.item_id = r.item_id", conn,
            )
            conn.close()

            if resumen.empty:
                self.label_contador.setText(
                    "resumen_actual está vacía todavía — esperar a que el recolector corra su primer job horario."
                )
                self.modelo_tabla.set_dataframe(resumen)
                return

            # antiguedad_maxima_horas: resumen_actual guarda el último dato
            # DISPONIBLE de cada ítem, que para uno poco líquido puede ser de
            # hace días — mostrar ese margen como si fuera el de ahora es
            # engañoso. Ver metricas.filtrar_screener_liquido.
            filtrado = filtrar_screener_liquido(
                resumen, volumen_24h_minimo=self.spin_volumen.value(),
                antiguedad_maxima_horas=ANTIGUEDAD_MAXIMA_HORAS,
            )
            clave_orden = self.combo_orden.currentData() or 'margen_predicho'

            margenes = _calcular_margenes_predichos(self.db, filtrado['item_id'].tolist())
            filtrado = filtrado.copy()
            filtrado['margen_predicho'] = filtrado['item_id'].map(margenes)

            # El orden se aplica DESPUÉS de calcular el margen predicho: es
            # la columna por la que se ordena por default, y no existe hasta
            # acá. na_position='last' deja abajo los ítems que ningún modelo
            # cubre, en vez de arriba (NaN ordena primero con ascending=False).
            filtrado = filtrado.sort_values(
                clave_orden, ascending=False, na_position='last').reset_index(drop=True)

            cubiertos = int(filtrado['margen_predicho'].notna().sum())
            self.label_contador.setText(
                f"{len(filtrado)} de {len(resumen)} ítems pasan el filtro de liquidez; "
                f"{cubiertos} con margen predicho por algún modelo."
            )
            self.modelo_tabla.set_dataframe(filtrado)
        except Exception as e:
            self.label_contador.setText(f"No se pudo actualizar el screener: {e}")
