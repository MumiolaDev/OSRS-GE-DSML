"""
ventana_principal.py — ventana principal de la app de escritorio.

Cascarón (workstream 3 del plan de la fase "app de escritorio v1"): por
ahora solo tiene el control start/stop del recolector y un indicador de
estado. Las páginas de contenido (Oportunidades, Mis modelos,
Configuración) se agregan como tabs en los workstreams siguientes —
mantener esta ventana simple a propósito (ver la corrección del usuario
sobre no repetir la sobrecarga de información del dashboard actual).
"""
from PySide6.QtWidgets import (
    QHBoxLayout, QLabel, QMainWindow, QPushButton, QVBoxLayout, QWidget,
)

from escritorio.hilo_recolector import HiloRecolector

DB_PATH = 'data/osrs_ge.db'


class VentanaPrincipal(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("OSRS GE Predictor")
        self.resize(1000, 700)

        self._hilo = None

        central = QWidget()
        layout = QVBoxLayout(central)

        fila_estado = QHBoxLayout()
        self.label_estado = QLabel("Recolector detenido")
        self.boton_toggle = QPushButton("Iniciar recolector")
        self.boton_toggle.clicked.connect(self._alternar_recolector)
        fila_estado.addWidget(self.label_estado, stretch=1)
        fila_estado.addWidget(self.boton_toggle)
        layout.addLayout(fila_estado)

        # Placeholder — reemplazado por las tabs Oportunidades/Mis modelos/
        # Configuración en los próximos workstreams del plan.
        layout.addWidget(QLabel("Páginas de contenido: próximo workstream."))

        self.setCentralWidget(central)

    def _alternar_recolector(self):
        if self._hilo is None or not self._hilo.isRunning():
            self._iniciar_recolector()
        else:
            self._detener_recolector()

    def _iniciar_recolector(self):
        self._hilo = HiloRecolector(DB_PATH)
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

    def closeEvent(self, event):
        """Al cerrar la ventana, pedir que el hilo del recolector termine
        limpio antes de salir — evita matar una recolección/entrenamiento
        de golpe a mitad de camino."""
        if self._hilo is not None and self._hilo.isRunning():
            self._hilo.detener()
            self._hilo.wait(5000)
        event.accept()
