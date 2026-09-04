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
import os
import time
from datetime import datetime

import pandas as pd
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (
    QGroupBox, QHBoxLayout, QLabel, QProgressBar, QPushButton, QVBoxLayout, QWidget,
)

import bloqueo
from escritorio.hilo_recolector import HiloRecolector
from escritorio.widgets.tabla_dataframe import crear_tabla

# Cada cuánto se chequea si hay un recolector corriendo AFUERA de la app
# (típicamente el servicio de systemd, ver docs/despliegue_24_7.md). Barato:
# es un flock no bloqueante sobre un archivo, no toca la DB.
MS_CHEQUEO_EXTERNO = 5000

# Cada cuántos chequeos se refresca además el panel de datos, para que la app
# sirva de monitor en vivo del servicio. Más espaciado porque esto sí hace
# COUNT(*) sobre tablas de millones de filas (ver _on_progreso).
CHEQUEOS_POR_REFRESCO_DATOS = 12  # 12 * 5s = 1 minuto

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


def _antiguedad_legible(segundos):
    """'hace 3 min' / 'hace 2 h' / 'hace 4 días' — para que el estado se lea
    de un vistazo sin tener que restar timestamps mentalmente."""
    if segundos is None:
        return "—"
    if segundos < 60:
        return "recién"
    if segundos < 3600:
        return f"hace {int(segundos // 60)} min"
    if segundos < 86400:
        return f"hace {int(segundos // 3600)} h"
    return f"hace {int(segundos // 86400)} días"


class PaginaInicio(QWidget):
    def __init__(self, db, db_path, parent=None):
        super().__init__(parent)
        self.db = db
        self.db_path = db_path
        self._hilo = None
        self._en_arranque = False
        self._externo = None      # info del recolector de otro proceso, si lo hay
        self._ticks = 0

        layout = QVBoxLayout(self)

        fila_estado = QHBoxLayout()
        self.label_estado = QLabel("Recolector detenido")
        # setWordWrap(True): un mensaje de error largo (ej. una excepción
        # completa vía _mostrar_error) no debe forzar el ancho mínimo de
        # toda la ventana -- ver el mismo problema real ya encontrado y
        # corregido en pagina_configuracion.py.
        self.label_estado.setWordWrap(True)
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

        # Antigüedad del dato más reciente: es la métrica que de verdad
        # dice si el pipeline está sano — un recolector "corriendo" que hace
        # tres horas que no inserta nada está roto igual.
        self.label_frescura = QLabel("")
        layout.addWidget(self.label_frescura)

        layout.addStretch()

        self._refrescar_datos_disponibles()

        # Chequeo periódico de un recolector externo (servicio de systemd):
        # sin esto la app ofrecería "Iniciar recolector" con el servicio ya
        # andando, y el usuario vería un error recién al apretar el botón.
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._chequear_recolector_externo)
        self._timer.start(MS_CHEQUEO_EXTERNO)
        self._chequear_recolector_externo()

    def _alternar_recolector(self):
        if self._externo is not None:
            return  # el botón ya está desactivado, pero por las dudas
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
        # NO se refresca "Datos disponibles" acá a propósito -- este
        # handler corre en CADA descarga individual durante el backfill
        # (potencialmente cientos de veces), y obtener_resumen_datos() hace
        # COUNT(*) sobre precios_1h/5m, que en una DB real de millones de
        # filas mide ~1 segundo por sí solo (medido: 4.4M filas en
        # precios_1h). Sumado al costo ya real del propio INSERT OR IGNORE
        # sobre una tabla de ese tamaño (documentado en CLAUDE.md) y al
        # segundo de pausa entre pedidos a la API, este refresco de más
        # era el único de los tres costos que no aportaba nada a cambio y
        # sí se podía evitar sin más — se sigue refrescando en cada cambio
        # de fase (_on_estado_cambio), que alcanza para ver el avance real
        # sin pagar el costo en cada descarga individual.

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
            self._actualizar_frescura(resumen)
        except Exception as e:
            logging.error(f"No se pudo refrescar 'Datos disponibles': {e}")

    def _actualizar_frescura(self, resumen):
        """Antigüedad del último dato horario, que es el que alimenta el
        screener y todos los modelos — si esto se atrasa, todo lo demás
        muestra el pasado sin avisar."""
        hasta = (resumen.get('precios_1h') or {}).get('hasta_ts')
        if hasta is None:
            self.label_frescura.setText("Sin datos horarios todavía.")
            return
        antiguedad = time.time() - hasta
        texto = f"Último dato horario: {_antiguedad_legible(antiguedad)}."
        # Un bucket de 1h cerrado se pide a :01 y tarda unos minutos en
        # aparecer, así que "hace menos de 2 horas" es lo normal; más que eso
        # es un síntoma (proceso caído, sin red, o la API devolviendo vacío).
        if antiguedad > 2 * 3600:
            texto += "  ⚠ El pipeline está atrasado."
        self.label_frescura.setText(texto)

    def _chequear_recolector_externo(self):
        """
        Detecta un recolector corriendo en OTRO proceso (el servicio de
        systemd, típicamente) y convierte esta pestaña en un monitor de solo
        lectura: sin esto, el botón "Iniciar recolector" seguiría ofreciendo
        arrancar un segundo recolector sobre la misma DB, que es justo lo que
        el candado de bloqueo.py existe para impedir.
        """
        self._ticks += 1
        try:
            info = bloqueo.hay_recolector_corriendo(self.db_path)
        except Exception as e:
            logging.error(f"No se pudo chequear el candado del recolector: {e}")
            return

        # El candado tomado por el hilo de esta misma app no cuenta como
        # "externo": ahí manda el flujo normal del botón.
        if info is not None and info.get('pid') == os.getpid():
            info = None

        if info is not None and self._externo is None:
            self._entrar_modo_monitor(info)
        elif info is None and self._externo is not None:
            self._salir_modo_monitor()
        elif info is not None:
            self._actualizar_texto_monitor(info)

        if self._externo is not None and self._ticks % CHEQUEOS_POR_REFRESCO_DATOS == 0:
            self._refrescar_datos_disponibles()

    def _entrar_modo_monitor(self, info):
        self._externo = info
        self.boton_toggle.setEnabled(False)
        self.barra_progreso.setVisible(False)
        self._actualizar_texto_monitor(info)
        self._refrescar_datos_disponibles()

    def _salir_modo_monitor(self):
        self._externo = None
        self.boton_toggle.setEnabled(True)
        self.boton_toggle.setText("Iniciar recolector")
        self.label_estado.setText("Recolector detenido")

    def _actualizar_texto_monitor(self, info):
        self._externo = info
        origen = info.get('origen', 'otro proceso')
        nombre = "servicio del sistema" if origen == bloqueo.ORIGEN_SERVICIO else origen
        inicio = info.get('inicio')
        desde = f", activo desde {_antiguedad_legible(time.time() - inicio)}" if inicio else ""
        self.boton_toggle.setText("Lo maneja el servicio")
        self.label_estado.setText(
            f"Recolector corriendo como {nombre} (PID {info.get('pid', '?')}){desde}. "
            f"Esta pestaña lo muestra en vivo; para pararlo, usá el servicio."
        )

    def hilo_activo(self):
        return self._hilo is not None and self._hilo.isRunning()

    def detener_para_cerrar(self):
        """Llamado desde VentanaPrincipal.closeEvent — pide que el hilo
        termine limpio (hasta 5s de espera) antes de salir, para no cortar
        una recolección/entrenamiento a mitad de camino."""
        if self.hilo_activo():
            self._hilo.detener()
            self._hilo.wait(5000)
