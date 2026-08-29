"""
paginas/pagina_inicio.py — estado del recolector (on/off, qué cadencia
está corriendo, progreso del relleno de huecos/replay al arrancar) y un
panel de "Datos disponibles" (filas y rango de fechas por tabla de
precios). Control de arranque/parada del hilo del recolector.

Pedido explícito del usuario tras probar la app: antes esta pestaña solo
mostraba una línea de texto genérica — ahora distingue qué cadencia se
está descargando en cada momento, muestra una barra de progreso real
durante el relleno de huecos/replay (los únicos procesos con un total
conocido de antemano — ver escritorio/hilo_recolector.py), y expone
cuánto historial hay acumulado sin tener que abrir la DB a mano.
"""
import logging
from datetime import datetime

import pandas as pd
from PySide6.QtWidgets import (
    QGroupBox, QHBoxLayout, QLabel, QProgressBar, QPushButton, QVBoxLayout, QWidget,
)

from escritorio.hilo_recolector import HiloRecolector
from escritorio.widgets.tabla_dataframe import crear_tabla

# Orden y nombre legible de cada tabla en el panel "Datos disponibles".
NOMBRES_TABLAS = {
    'precios_5m': '5 minutos',
    'precios_1h': '1 hora',
    'precios_6h': '6 horas',
    'precios_1h_diario': '1 hora (archivado)',
}

COLUMNAS_DATOS = [
    ('tabla', 'Intervalo'),
    ('filas', 'Filas'),
    ('desde', 'Desde'),
    ('hasta', 'Hasta'),
    ('dias', 'Días de historial'),
]

# Mensajes de estado_cambio que marcan "recolector en régimen normal, nada
# más que mostrar como progreso" -- ocultan la barra y salen del modo
# "arranque" (ver _on_estado_cambio).
ESTADOS_SIN_PROGRESO = ("Recolectando (activo)", "Detenido")


def _formatear_fecha_corta(ts):
    return "—" if ts is None else datetime.fromtimestamp(ts).strftime('%Y-%m-%d')


class PaginaInicio(QWidget):
    def __init__(self, db, db_path, parent=None):
        super().__init__(parent)
        self.db = db
        self.db_path = db_path
        self._hilo = None
        self._en_arranque = False

        layout = QVBoxLayout(self)

        fila_estado = QHBoxLayout()
        self.label_estado = QLabel("Recolector detenido")
        self.boton_toggle = QPushButton("Iniciar recolector")
        self.boton_toggle.clicked.connect(self._alternar_recolector)
        fila_estado.addWidget(self.label_estado, stretch=1)
        fila_estado.addWidget(self.boton_toggle)
        layout.addLayout(fila_estado)

        self.barra_progreso = QProgressBar()
        self.barra_progreso.setVisible(False)
        layout.addWidget(self.barra_progreso)

        grupo_datos = QGroupBox("Datos disponibles")
        layout_datos = QVBoxLayout(grupo_datos)
        self.tabla_datos, self.modelo_tabla_datos = crear_tabla(columnas=COLUMNAS_DATOS)
        self.tabla_datos.setMaximumHeight(160)  # 4 filas fijas, no hace falta más alto
        layout_datos.addWidget(self.tabla_datos)
        layout.addWidget(grupo_datos)

        layout.addStretch()

        self._refrescar_datos_disponibles()

    def _alternar_recolector(self):
        if self._hilo is None or not self._hilo.isRunning():
            self._iniciar_recolector()
        else:
            self._detener_recolector()

    def _iniciar_recolector(self):
        self._en_arranque = True
        self._hilo = HiloRecolector(self.db_path)
        self._hilo.estado_cambio.connect(self._on_estado_cambio)
        self._hilo.progreso.connect(self._on_progreso)
        self._hilo.error.connect(self._mostrar_error)
        self._hilo.finished.connect(self._al_terminar_hilo)
        self._hilo.start()
        self.boton_toggle.setText("Detener recolector")

    def _detener_recolector(self):
        if self._hilo is not None:
            self.boton_toggle.setEnabled(False)
            self.label_estado.setText("Deteniendo...")
            self.barra_progreso.setVisible(False)
            self._hilo.detener()

    def _al_terminar_hilo(self):
        self.boton_toggle.setText("Iniciar recolector")
        self.boton_toggle.setEnabled(True)
        self.barra_progreso.setVisible(False)
        self._en_arranque = False

    def _on_estado_cambio(self, texto):
        self.label_estado.setText(texto)
        if texto in ESTADOS_SIN_PROGRESO:
            self._en_arranque = False
            self.barra_progreso.setVisible(False)
        elif self._en_arranque:
            # Cada cambio de fase durante el arranque (catálogo,
            # recolección inicial, relleno de huecos, catch-up, primer job
            # horario) reinicia la barra a modo indeterminado -- si la fase
            # trae progreso numérico real (relleno de huecos/replay), el
            # primer tick de _on_progreso la pasa a modo determinado en
            # seguida. Fuera del arranque (ticks periódicos del régimen
            # normal, ver hilo_recolector._con_notificacion) no se toca la
            # barra -- ahí no hay "total" que mostrar.
            self.barra_progreso.setVisible(True)
            self.barra_progreso.setRange(0, 0)
        self._refrescar_datos_disponibles()

    def _on_progreso(self, fase, actual, total):
        if total > 0:
            self.barra_progreso.setVisible(True)
            self.barra_progreso.setRange(0, total)
            self.barra_progreso.setValue(actual)
            # Ojo: en el mini-lenguaje de QProgressBar.setFormat, %p YA es
            # solo el número (sin el signo %) y %% NO es un escape de "%
            # literal" (Qt lo deja tal cual, duplicado) -- un solo % suelto
            # después de %p alcanza para el signo de porcentaje. Verificado
            # a mano: %p%% rendía "(7%%)" en vez de "(7%)".
            self.barra_progreso.setFormat(f"{fase}: %v/%m (%p%)")
        self._refrescar_datos_disponibles()

    def _mostrar_error(self, mensaje):
        self.label_estado.setText(f"Error: {mensaje}")
        self.barra_progreso.setVisible(False)
        self._en_arranque = False

    def _refrescar_datos_disponibles(self):
        """
        Envuelto en try/except (mismo criterio de hardening del resto de
        la app): una falla acá (ej. la DB bloqueada un instante por una
        escritura concurrente) no debe romper la pestaña, solo loguearse.
        """
        try:
            resumen = self.db.obtener_resumen_datos()
            filas = [
                {
                    'tabla': nombre,
                    'filas': resumen[tabla]['filas'],
                    'desde': _formatear_fecha_corta(resumen[tabla]['desde_ts']),
                    'hasta': _formatear_fecha_corta(resumen[tabla]['hasta_ts']),
                    'dias': (
                        f"{resumen[tabla]['dias_historial']:.1f}"
                        if resumen[tabla]['dias_historial'] is not None else "—"
                    ),
                }
                for tabla, nombre in NOMBRES_TABLAS.items()
            ]
            self.modelo_tabla_datos.set_dataframe(pd.DataFrame(filas))
        except Exception as e:
            logging.error(f"No se pudo refrescar 'Datos disponibles': {e}")

    def hilo_activo(self):
        return self._hilo is not None and self._hilo.isRunning()

    def detener_para_cerrar(self):
        """Llamado desde VentanaPrincipal.closeEvent — pide que el hilo
        termine limpio (hasta 5s de espera) antes de salir, para no cortar
        una recolección/entrenamiento a mitad de camino."""
        if self.hilo_activo():
            self._hilo.detener()
            self._hilo.wait(5000)
