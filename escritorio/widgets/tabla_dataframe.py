"""
widgets/tabla_dataframe.py — QTableView respaldado por un pandas DataFrame,
reusado por todas las páginas que muestran una tabla (Oportunidades, Mis
modelos). Evita repetir el boilerplate de un QAbstractTableModel en cada
página, y centraliza cómo se ven las tablas de la app (columnas curadas
con título en español, formato de números) en un solo lugar.
"""
import logging

import pandas as pd
from PySide6.QtCore import QAbstractTableModel, Qt
from PySide6.QtWidgets import QHeaderView, QTableView


class ModeloDataFrame(QAbstractTableModel):
    """
    Adaptador de solo lectura entre un DataFrame y QTableView.

    columnas: lista opcional de (nombre_columna_df, título_mostrado) — para
    mostrar solo un subconjunto curado de columnas con encabezados en
    español legibles, en vez de todas las columnas crudas de la tabla de
    origen (ver la corrección del usuario sobre no repetir la sobrecarga
    de información del dashboard actual). Si no se pasa, se muestran todas
    las columnas del DataFrame tal cual.

    formatos: dict opcional {nombre_columna: función(valor) -> str} para
    columnas que necesitan formato especial (separador de miles, %, etc.)
    en vez de str(valor) plano.
    """

    def __init__(self, df, columnas=None, formatos=None, parent=None):
        super().__init__(parent)
        self._columnas = columnas
        self._formatos = formatos or {}
        self._df = pd.DataFrame()
        self.set_dataframe(df)

    def set_dataframe(self, df):
        self.beginResetModel()
        if self._columnas:
            faltantes = [c for c, _ in self._columnas if c not in df.columns]
            if faltantes:
                # No se rompe la tabla por una columna curada ausente (ej.
                # un DataFrame vacío sin ninguna columna todavía), pero se
                # loguea siempre — que una columna pedida desaparezca en
                # silencio (ver el bug real de "Límite de compra" faltando
                # por no haber hecho el JOIN con `items` en
                # pagina_oportunidades.py) es peor que un warning de más.
                logging.warning(f"ModeloDataFrame: columnas curadas ausentes en el DataFrame: {faltantes}")
            presentes = [c for c, _ in self._columnas if c in df.columns]
            self._df = df[presentes].reset_index(drop=True) if presentes else df.reset_index(drop=True)
        else:
            self._df = df.reset_index(drop=True)
        self.endResetModel()

    def rowCount(self, parent=None):
        return len(self._df)

    def columnCount(self, parent=None):
        return len(self._df.columns)

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid() or role != Qt.DisplayRole:
            return None
        columna = self._df.columns[index.column()]
        valor = self._df.iat[index.row(), index.column()]
        formateador = self._formatos.get(columna)
        if formateador is not None and valor is not None and pd.notna(valor):
            try:
                return formateador(valor)
            except Exception:
                return str(valor)
        return "" if valor is None or (isinstance(valor, float) and pd.isna(valor)) else str(valor)

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role != Qt.DisplayRole:
            return None
        if orientation == Qt.Horizontal:
            titulos = dict(self._columnas) if self._columnas else {}
            columna = self._df.columns[section]
            return titulos.get(columna, columna)
        return str(section + 1)


def crear_tabla(df=None, columnas=None, formatos=None):
    """Crea un QTableView + su ModeloDataFrame ya conectado — un solo
    punto de construcción para que todas las tablas de la app se vean y
    comporten igual (solo lectura, filas alternadas, columnas ajustadas al
    contenido)."""
    modelo = ModeloDataFrame(df if df is not None else pd.DataFrame(), columnas, formatos)
    vista = QTableView()
    vista.setModel(modelo)
    vista.setEditTriggers(QTableView.NoEditTriggers)
    vista.setAlternatingRowColors(True)
    vista.setSelectionBehavior(QTableView.SelectRows)
    vista.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
    vista.verticalHeader().setVisible(False)
    return vista, modelo
