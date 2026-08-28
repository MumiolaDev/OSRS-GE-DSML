"""
hilo_recolector.py — corre el loop del recolector (recolección + jobs
programados) en un hilo de Qt aparte, para no bloquear la ventana
principal mientras recolecta/entrena.

Reusa las funciones de recolector.py tal cual (collect_5min/1h/6h,
job_horario/job_diario/job_semanal, rellenar_huecos_al_inicio,
verificar_catchup_semanal, _programar_cada_n_minutos_alineado) — no
duplica ninguna lógica de recolección/entrenamiento, solo reimplementa de
forma controlable (start/stop) el mismo scheduling loop que hoy vive en
`if __name__ == "__main__":` de recolector.py, e informa el estado a la
UI vía señales de Qt en vez de solo loguear a archivo/consola.

Cada función de base_de_datos.py abre su propia conexión SQLite por
llamada (sqlite3.connect() dentro de cada método) — ese patrón ya existente
es lo que hace seguro llamarlas desde este hilo sin cambios adicionales
(SQLite no permite compartir una misma conexión entre hilos, pero acá
nunca se comparte una: cada llamada abre y cierra la suya).
"""
import logging

import schedule
from PySide6.QtCore import QThread, Signal

import recolector
from base_de_datos import OSRSBaseDatos


class HiloRecolector(QThread):
    """
    QThread que corre la recolección 24/7. Emite señales para que la UI
    (que vive en el hilo principal) se entere de cambios de estado sin
    tocar widgets desde este hilo — Qt no permite tocar widgets fuera del
    hilo de UI; toda comunicación hacia la ventana principal pasa por
    señales/slots (Qt las entrega en el hilo del receptor automáticamente
    cuando emisor y receptor viven en hilos distintos).
    """
    estado_cambio = Signal(str)  # texto legible del estado actual, para un QLabel
    error = Signal(str)          # error no recuperable (ej. no se pudo abrir la DB)

    def __init__(self, db_path='data/osrs_ge.db', parent=None):
        super().__init__(parent)
        self.db_path = db_path
        self._detener = False

    def run(self):
        try:
            db = OSRSBaseDatos(self.db_path)
        except Exception as e:
            self.error.emit(f"No se pudo abrir la base de datos: {e}")
            return

        self.estado_cambio.emit("Actualizando catálogo de ítems...")
        try:
            mapping = recolector.api.get_item_mapping()
            db.guardar_items(mapping)
        except Exception as e:
            logging.error(f"Error actualizando catálogo de ítems: {e}")

        self.estado_cambio.emit("Recolectando datos iniciales...")
        recolector.collect_5min(db)
        recolector.collect_1h(db)
        recolector.collect_6h(db)

        self.estado_cambio.emit("Rellenando huecos de datos...")
        try:
            recolector.rellenar_huecos_al_inicio(db)
        except Exception as e:
            logging.error(f"Error rellenando huecos: {e}")

        recolector.verificar_catchup_semanal(db)

        self.estado_cambio.emit("Actualizando resumen y modelos...")
        recolector.job_horario(db)

        # schedule.clear() antes de registrar: defensivo ante un
        # arranque/parada/reinicio dentro del mismo proceso (la app crea un
        # HiloRecolector nuevo en cada "Iniciar", pero todos comparten el
        # scheduler global por default de la librería `schedule`, el mismo
        # que usa recolector.py.__main__).
        schedule.clear()
        recolector._programar_cada_n_minutos_alineado(5, recolector.collect_5min, db)
        schedule.every().hour.at(":00", "UTC").do(recolector.collect_1h, db)
        for h in (0, 6, 12, 18):
            schedule.every().day.at(f"{h:02d}:00", "UTC").do(recolector.collect_6h, db)
        schedule.every().hour.at(":05", "UTC").do(recolector.job_horario, db)
        schedule.every().day.at("03:00", "UTC").do(recolector.job_diario, db)
        schedule.every().sunday.at("04:00", "UTC").do(recolector.job_semanal, db)

        self.estado_cambio.emit("Recolectando (activo)")
        while not self._detener:
            schedule.run_pending()
            self.msleep(1000)

        schedule.clear()
        self.estado_cambio.emit("Detenido")

    def detener(self):
        """
        Pide que el loop termine en la próxima vuelta (hasta 1s de
        latencia, ver el `msleep(1000)` de run()) — no mata el hilo a la
        fuerza, para no cortar una recolección/entrenamiento a mitad de
        camino. El llamador debe esperar la señal `finished` (heredada de
        QThread) o usar `wait()` para saber cuándo terminó de verdad.
        """
        self._detener = True
