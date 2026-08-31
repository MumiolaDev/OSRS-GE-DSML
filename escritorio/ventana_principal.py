"""
ventana_principal.py — ventana principal de la app de escritorio: arma las
tabs y las conecta a una única conexión de solo-lectura a la DB
(base_de_datos.OSRSBaseDatos abre su propia conexión SQLite por llamada,
así que compartir esta instancia entre páginas es seguro).

Deliberadamente pocas tabs (ver la corrección del usuario sobre no repetir
la sobrecarga de información del dashboard de Streamlit): Inicio (control
del recolector), Oportunidades, Mis modelos y Configuración.

Endurecido contra dos fallas que crashearían la app entera al arrancar
sin ninguna explicación útil para alguien sin conocimientos técnicos: una
ruta de DB inválida guardada en config.json (_abrir_db_con_fallback) y una
excepción inesperada construyendo cualquiera de las páginas
(_agregar_pagina_segura) — ninguna de las dos debería poder tumbar toda
la ventana por un problema acotado a una sola sección.
"""
import logging

from PySide6.QtWidgets import QLabel, QMainWindow, QMessageBox, QTabWidget

import configuracion
from base_de_datos import OSRSBaseDatos
from escritorio.paginas.pagina_configuracion import PaginaConfiguracion
from escritorio.paginas.pagina_inicio import PaginaInicio
from escritorio.paginas.pagina_modelos import PaginaModelos
from escritorio.paginas.pagina_oportunidades import PaginaOportunidades

# db_path en config.json (editable desde la tab Configuración) tiene
# prioridad sobre el default -- aplica recién la próxima vez que se abre
# la app, no en caliente (ver pagina_configuracion.py).
DB_PATH = configuracion.obtener('db_path') or 'data/osrs_ge.db'

DB_PATH_DEFAULT = 'data/osrs_ge.db'


class VentanaPrincipal(QMainWindow):
    def __init__(self, db_path=DB_PATH):
        super().__init__()
        self.setWindowTitle("OSRS GE Predictor")
        self.resize(600, 800)

        self.db = self._abrir_db_con_fallback(db_path)
        db_path_efectivo = self.db.db_path

        tabs = QTabWidget()

        self.pagina_inicio = PaginaInicio(self.db, db_path_efectivo)
        tabs.addTab(self.pagina_inicio, "Inicio")

        self.pagina_oportunidades = self._agregar_pagina_segura(
            tabs, "Oportunidades", lambda: PaginaOportunidades(self.db),
        )
        self.pagina_modelos = self._agregar_pagina_segura(
            tabs, "Mis modelos", lambda: PaginaModelos(self.db, db_path_efectivo),
        )
        self.pagina_configuracion = self._agregar_pagina_segura(
            tabs, "Configuración", lambda: PaginaConfiguracion(),
        )

        self.setCentralWidget(tabs)

    def _abrir_db_con_fallback(self, db_path):
        """
        Abre la DB en `db_path` (típicamente lo que haya en config.json) —
        si falla (ej. una ruta inválida guardada a mano en la pestaña
        Configuración, o un directorio que ya no existe), avisa con un
        mensaje claro y reintenta una vez con DB_PATH_DEFAULT en vez de
        crashear la app entera al arrancar sin ninguna explicación. Si el
        default TAMBIÉN falla, no hay forma de seguir — se relanza la
        excepción original (no hay ventana sin DB).
        """
        try:
            return OSRSBaseDatos(db_path)
        except Exception as e:
            if db_path == DB_PATH_DEFAULT:
                raise
            logging.error(f"No se pudo abrir la DB configurada ('{db_path}'): {e}")
            QMessageBox.warning(
                self, "No se pudo abrir la base de datos",
                f"No se pudo abrir '{db_path}' (configurado en la pestaña Configuración): {e}\n\n"
                f"Se va a usar la ruta por default '{DB_PATH_DEFAULT}' en su lugar — "
                "podés corregir la ruta desde la pestaña Configuración.",
            )
            return OSRSBaseDatos(DB_PATH_DEFAULT)

    def _agregar_pagina_segura(self, tabs, titulo, constructor):
        """
        Construye una página con `constructor()` y la agrega como tab. Si
        la construcción falla (ej. una query inesperada rompe en
        __init__), no debe tirar abajo TODA la ventana — se deja un
        mensaje de error en el lugar de esa tab y el resto de la app
        sigue funcionando. Devuelve la página real, o el QLabel de error
        si falló (el llamador no debe asumir el tipo).
        """
        try:
            pagina = constructor()
        except Exception as e:
            logging.error(f"No se pudo cargar la pestaña '{titulo}': {e}")
            pagina = QLabel(f"No se pudo cargar esta sección — {e}")
            pagina.setWordWrap(True)
        tabs.addTab(pagina, titulo)
        return pagina

    def closeEvent(self, event):
        """Al cerrar, espera a que terminen limpio tanto el hilo del
        recolector (pagina_inicio) como uno de entrenamiento en curso
        (pagina_modelos) — ninguno se mata a la fuerza a mitad de camino.
        hasattr() porque cualquiera de las dos puede ser el QLabel de
        error de _agregar_pagina_segura si esa pestaña no cargó."""
        if hasattr(self.pagina_inicio, 'detener_para_cerrar'):
            self.pagina_inicio.detener_para_cerrar()
        if hasattr(self.pagina_modelos, 'esperar_para_cerrar'):
            self.pagina_modelos.esperar_para_cerrar()
        event.accept()
