"""
paginas/pagina_inicio.py — estado del recolector (on/off, último mensaje)
y control de arranque/parada. Antes vivía inline en ventana_principal.py;
se movió a su propia página al agregar la tab "Oportunidades", para que
ventana_principal.py se quede solo con el armado de las tabs.
"""
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from escritorio.hilo_recolector import HiloRecolector


class PaginaInicio(QWidget):
    def __init__(self, db_path, parent=None):
        super().__init__(parent)
        self.db_path = db_path
        self._hilo = None

        layout = QVBoxLayout(self)

        fila_estado = QHBoxLayout()
        self.label_estado = QLabel("Recolector detenido")
        self.boton_toggle = QPushButton("Iniciar recolector")
        self.boton_toggle.clicked.connect(self._alternar_recolector)
        fila_estado.addWidget(self.label_estado, stretch=1)
        fila_estado.addWidget(self.boton_toggle)
        layout.addLayout(fila_estado)
        layout.addStretch()

    def _alternar_recolector(self):
        if self._hilo is None or not self._hilo.isRunning():
            self._iniciar_recolector()
        else:
            self._detener_recolector()

    def _iniciar_recolector(self):
        self._hilo = HiloRecolector(self.db_path)
        self._hilo.estado_cambio.connect(self.label_estado.setText)
        self._hilo.error.connect(self._mostrar_error)
        self._hilo.finished.connect(self._al_terminar_hilo)
        self._hilo.start()
        self.boton_toggle.setText("Detener recolector")

    def _detener_recolector(self):
        if self._hilo is not None:
            self.boton_toggle.setEnabled(False)
            self.label_estado.setText("Deteniendo...")
            self._hilo.detener()

    def _al_terminar_hilo(self):
        self.boton_toggle.setText("Iniciar recolector")
        self.boton_toggle.setEnabled(True)

    def _mostrar_error(self, mensaje):
        self.label_estado.setText(f"Error: {mensaje}")

    def hilo_activo(self):
        return self._hilo is not None and self._hilo.isRunning()

    def detener_para_cerrar(self):
        """Llamado desde VentanaPrincipal.closeEvent — pide que el hilo
        termine limpio (hasta 5s de espera) antes de salir, para no cortar
        una recolección/entrenamiento a mitad de camino."""
        if self.hilo_activo():
            self._hilo.detener()
            self._hilo.wait(5000)
