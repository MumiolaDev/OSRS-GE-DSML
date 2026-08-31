"""
paginas/pagina_oportunidades.py — vista principal de la app: qué ítems
conviene mirar ahora mismo. Fusiona lo que en el dashboard de Streamlit
eran dos pestañas separadas (Screener + Señal direccional F2P) en una
sola tabla curada — ver la corrección del usuario sobre simplificar la
app en vez de portar el dashboard tal cual.

Columnas curadas de resumen_actual (metricas.py) + una columna de señal
direccional (sube/estable/baja) para los ítems cubiertos por algún modelo
clasificador ACTIVO del usuario (base_de_datos.modelos_config) — ya no
solo la variante F2P fija de antes, ahora puede ser cualquier clasificador
que el usuario haya definido desde "Mis modelos". Si un ítem está cubierto
por más de un clasificador activo, se muestra la señal del más reciente
(creado_en más alto) — mantiene la tabla a una sola columna de señal en
vez de una por modelo, a propósito.
"""
import logging
import sqlite3

import pandas as pd
from PySide6.QtWidgets import (
    QComboBox, QHBoxLayout, QLabel, QPushButton, QSpinBox, QVBoxLayout, QWidget,
)

from metricas import filtrar_screener_liquido
from prediccion import cargar_modelo, pronosticar_clase_item
from escritorio.widgets.tabla_dataframe import crear_tabla

COLUMNAS = [
    ('name', 'Ítem'),
    ('margen_neto', 'Margen neto'),
    ('roi_pct', 'ROI'),
    ('buy_limit', 'Límite de compra'),
    ('volumen_24h', 'Volumen 24h'),
    ('señal', 'Señal'),
]

FORMATOS = {
    'margen_neto': lambda v: f"{int(v):,} gp".replace(',', '.'),
    'roi_pct': lambda v: f"{float(v):.1f}%",
    'buy_limit': lambda v: f"{int(v):,}".replace(',', '.'),
    'volumen_24h': lambda v: f"{int(v):,}".replace(',', '.'),
}

ORDEN_OPCIONES = [
    ('margen_neto', 'Margen neto'),
    ('roi_pct', 'ROI'),
    ('volumen_24h', 'Volumen 24h'),
]


def _calcular_senales(db, item_ids_screener):
    """
    Para cada modelo activo de tipo='clasificador' en modelos_config,
    resuelve su universo real de ítems (cfg['item_ids'] directo si
    modo_seleccion='manual', o db.obtener_top_items_liquidez con sus
    parámetros si 'liquidez') y calcula la señal en vivo
    (prediccion.pronosticar_clase_item) SOLO para los ítems que además
    están en `item_ids_screener` (los que se van a mostrar) — evita correr
    inferencia sobre ítems que ni van a aparecer en la tabla.

    Devuelve {item_id: 'sube'|'estable'|'baja'}. Si un ítem está cubierto
    por más de un clasificador activo, gana el más reciente (se procesan
    del más viejo al más nuevo, y el último en escribir pisa).
    """
    item_ids_screener = set(item_ids_screener)
    señales = {}

    modelos = [m for m in db.listar_modelos_config(estado='activo') if m['tipo'] == 'clasificador']
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

        for item_id in candidatos:
            resultado = pronosticar_clase_item(db, item_id, bundle=bundle)
            if resultado is not None:
                señales[item_id] = resultado['label']

    return señales


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

            filtrado = filtrar_screener_liquido(resumen, volumen_24h_minimo=self.spin_volumen.value())
            clave_orden = self.combo_orden.currentData() or 'margen_neto'
            filtrado = filtrado.sort_values(clave_orden, ascending=False).reset_index(drop=True)

            señales = _calcular_senales(self.db, filtrado['item_id'].tolist())
            filtrado = filtrado.copy()
            filtrado['señal'] = filtrado['item_id'].map(señales).fillna('—')

            self.label_contador.setText(f"{len(filtrado)} de {len(resumen)} ítems pasan el filtro de liquidez.")
            self.modelo_tabla.set_dataframe(filtrado)
        except Exception as e:
            self.label_contador.setText(f"No se pudo actualizar el screener: {e}")
