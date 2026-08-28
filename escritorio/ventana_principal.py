"""
ventana_principal.py — ventana principal de la app de escritorio: arma las
tabs y las conecta a una única conexión de solo-lectura a la DB
(base_de_datos.OSRSBaseDatos abre su propia conexión SQLite por llamada,
así que compartir esta instancia entre páginas es seguro).

Deliberadamente pocas tabs (ver la corrección del usuario sobre no repetir
la sobrecarga de información del dashboard de Streamlit): Inicio (control
del recolector), Oportunidades, Mis modelos y Configuración.
"""
from PySide6.QtWidgets import QMainWindow, QTabWidget

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


class VentanaPrincipal(QMainWindow):
    def __init__(self, db_path=DB_PATH):
        super().__init__()
        self.setWindowTitle("OSRS GE Predictor")
        self.resize(1000, 700)

        self.db = OSRSBaseDatos(db_path)

        self.pagina_inicio = PaginaInicio(db_path)
        self.pagina_oportunidades = PaginaOportunidades(self.db)
        self.pagina_modelos = PaginaModelos(self.db, db_path)
        self.pagina_configuracion = PaginaConfiguracion()

        tabs = QTabWidget()
        tabs.addTab(self.pagina_inicio, "Inicio")
        tabs.addTab(self.pagina_oportunidades, "Oportunidades")
        tabs.addTab(self.pagina_modelos, "Mis modelos")
        tabs.addTab(self.pagina_configuracion, "Configuración")
        self.setCentralWidget(tabs)

    def closeEvent(self, event):
        self.pagina_inicio.detener_para_cerrar()
        event.accept()
