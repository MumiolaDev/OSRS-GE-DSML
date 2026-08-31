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
    estado_cambio = Signal(str)       # texto legible del estado actual, para un QLabel
    progreso = Signal(str, int, int)  # (fase, actual, total) -- solo durante backfill/replay
    error = Signal(str)               # error no recuperable (ej. no se pudo abrir la DB)

    def __init__(self, db_path='data/osrs_ge.db', parent=None):
        super().__init__(parent)
        self.db_path = db_path
        self._detener = False

    def _con_notificacion(self, nombre_legible, func):
        """
        Envuelve `func` (una de collect_5min/1h/6h, job_horario, etc.) para
        que, al dispararse desde el scheduler, avise por estado_cambio qué
        cadencia/job está corriendo en ESE momento y vuelva al texto
        "activo" al terminar — sin esto, una vez alcanzado el régimen
        normal la UI se queda con el mismo texto genérico para siempre, sin
        distinguir un tick de 5m de uno de 1h o de un job_horario.
        """
        def _wrapped(*args, **kwargs):
            self.estado_cambio.emit(f"Recolectando {nombre_legible}...")
            resultado = func(*args, **kwargs)
            if not self._detener:
                self.estado_cambio.emit("Recolectando (activo)")
            return resultado
        return _wrapped

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

        # Recolección inicial, un mensaje por cadencia (antes era un solo
        # "Recolectando datos iniciales..." que no distinguía cuál de las
        # tres estaba en curso).
        self.estado_cambio.emit("Recolectando 5m inicial...")
        recolector.collect_5min(db)
        self.estado_cambio.emit("Recolectando 1h inicial...")
        recolector.collect_1h(db)
        self.estado_cambio.emit("Recolectando 6h inicial...")
        recolector.collect_6h(db)

        if self._detener:
            self.estado_cambio.emit("Detenido")
            return

        self.estado_cambio.emit("Rellenando huecos de datos...")
        try:
            recolector.rellenar_huecos_al_inicio(
                db, on_progreso=self.progreso.emit, debe_detener=lambda: self._detener,
            )
        except Exception as e:
            logging.error(f"Error rellenando huecos: {e}")

        # Chequear _detener entre cada fase del arranque (además de dentro
        # de rellenar_huecos_al_inicio/backfill_faltantes/ejecutar_replay,
        # que ahora sí lo respetan a mitad de su propio loop) -- sin esto,
        # pedir "Detener" durante el backfill lo cortaba ahí pero la app
        # igual seguía con el catch-up semanal y el reentrenamiento de
        # job_horario antes de terminar, con la misma sensación de "no
        # respeta el botón".
        if self._detener:
            self.estado_cambio.emit("Detenido")
            return

        recolector.verificar_catchup_semanal(db)

        if self._detener:
            self.estado_cambio.emit("Detenido")
            return

        self.estado_cambio.emit("Actualizando resumen y modelos...")
        recolector.job_horario(db)

        if self._detener:
            self.estado_cambio.emit("Detenido")
            return

        # schedule.clear() antes de registrar: defensivo ante un
        # arranque/parada/reinicio dentro del mismo proceso (la app crea un
        # HiloRecolector nuevo en cada "Iniciar", pero todos comparten el
        # scheduler global por default de la librería `schedule`, el mismo
        # que usa recolector.py.__main__).
        schedule.clear()
        recolector._programar_cada_n_minutos_alineado(5, self._con_notificacion("5m", recolector.collect_5min), db)
        schedule.every().hour.at(":00", "UTC").do(self._con_notificacion("1h", recolector.collect_1h), db)
        for h in (0, 6, 12, 18):
            schedule.every().day.at(f"{h:02d}:00", "UTC").do(self._con_notificacion("6h", recolector.collect_6h), db)
        schedule.every().hour.at(":05", "UTC").do(self._con_notificacion("resumen y modelos (job horario)", recolector.job_horario), db)
        schedule.every().day.at("03:00", "UTC").do(self._con_notificacion("modelos (job diario)", recolector.job_diario), db)
        schedule.every().sunday.at("04:00", "UTC").do(self._con_notificacion("mantenimiento semanal", recolector.job_semanal), db)

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
