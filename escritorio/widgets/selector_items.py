"""
widgets/selector_items.py — buscador con autocompletado + multi-selección
sobre la tabla `items`, usado por pagina_modelos.py para elegir a mano los
ítems de un modelo (modo_seleccion='manual'). Con ~4.600 ítems en el
catálogo real (ver CLAUDE.md), un dropdown plano no sirve — filtra por
nombre a medida que se escribe, máximo 200 resultados a la vez.
"""
import sqlite3

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QLineEdit, QListWidget, QListWidgetItem, QVBoxLayout, QWidget


class SelectorItems(QWidget):
    """
    Caja de búsqueda + lista con checkboxes. La selección lógica vive en
    `self._seleccionados` ({item_id: nombre}), separada de lo que la lista
    muestra en un momento dado — así filtrar el texto de búsqueda no
    pierde ítems ya elegidos que quedan fuera del filtro actual.
    """

    def __init__(self, db_path, max_seleccion=None, parent=None):
        super().__init__(parent)
        self.db_path = db_path
        self.max_seleccion = max_seleccion
        self._seleccionados = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.campo_busqueda = QLineEdit()
        self.campo_busqueda.setPlaceholderText("Buscar ítem por nombre...")
        self.campo_busqueda.textChanged.connect(self._filtrar)
        layout.addWidget(self.campo_busqueda)

        self.lista = QListWidget()
        self.lista.itemChanged.connect(self._on_item_changed)
        layout.addWidget(self.lista)

        self.label_contador = QLabel()
        layout.addWidget(self.label_contador)

        self._actualizar_contador()
        self._filtrar("")

    def _filtrar(self, texto):
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            texto = texto.strip()
            if texto:
                c.execute(
                    "SELECT item_id, name FROM items WHERE name LIKE ? ORDER BY name LIMIT 200",
                    (f"%{texto}%",),
                )
            else:
                c.execute("SELECT item_id, name FROM items WHERE name IS NOT NULL ORDER BY name LIMIT 200")
            filas = c.fetchall()
            conn.close()
        except sqlite3.Error:
            filas = []

        self.lista.blockSignals(True)
        self.lista.clear()
        if not filas:
            # Catálogo vacío (DB recién creada, antes de la primera
            # recolección) o error de lectura -- un QListWidget sin filas
            # y sin ninguna explicación se ve como si el widget estuviera
            # roto, no como "todavía no hay datos".
            aviso = QListWidgetItem(
                "Sin ítems para elegir todavía"
                if not texto
                else f"Ningún ítem coincide con «{texto}»"
            )
            aviso.setFlags(Qt.NoItemFlags)
            self.lista.addItem(aviso)
        for item_id, nombre in filas:
            elemento = QListWidgetItem(nombre or f"(ítem {item_id})")
            elemento.setData(Qt.UserRole, item_id)
            elemento.setFlags(elemento.flags() | Qt.ItemIsUserCheckable)
            elemento.setCheckState(Qt.Checked if item_id in self._seleccionados else Qt.Unchecked)
            self.lista.addItem(elemento)
        self.lista.blockSignals(False)

    def _on_item_changed(self, elemento):
        item_id = elemento.data(Qt.UserRole)
        if elemento.checkState() == Qt.Checked:
            if (
                self.max_seleccion is not None
                and item_id not in self._seleccionados
                and len(self._seleccionados) >= self.max_seleccion
            ):
                # Tope alcanzado — revertir el check en vez de aceptarlo.
                self.lista.blockSignals(True)
                elemento.setCheckState(Qt.Unchecked)
                self.lista.blockSignals(False)
                return
            self._seleccionados[item_id] = elemento.text()
        else:
            self._seleccionados.pop(item_id, None)
        self._actualizar_contador()

    def _actualizar_contador(self):
        texto = f"{len(self._seleccionados)} ítem(s) elegido(s)"
        if self.max_seleccion is not None:
            texto += f" (máx. {self.max_seleccion})"
        self.label_contador.setText(texto)

    def seleccionados(self):
        """Lista de item_id marcados, en el orden en que se eligieron."""
        return list(self._seleccionados.keys())

    def set_seleccionados(self, item_ids_con_nombres):
        """item_ids_con_nombres: dict {item_id: nombre} — para precargar
        una selección existente al editar un modelo."""
        self._seleccionados = dict(item_ids_con_nombres)
        self._actualizar_contador()
        self._filtrar(self.campo_busqueda.text())
