"""
hilo_entrenamiento.py — QThread de un solo uso para el botón "Entrenar
ahora" de la página Mis modelos: entrenar (XGBoost, de segundos a minutos
según cuántos ítems) bloquearía la ventana si se hiciera en el hilo
principal.

Deliberadamente independiente de HiloRecolector: no le encola la tarea al
hilo del recolector (si está corriendo) para no acoplar un entrenamiento
disparado a mano al ciclo de vida de ese loop — funciona igual esté o no
corriendo el recolector. Cada conexión SQLite se abre por llamada (mismo
patrón de siempre en base_de_datos.py, nunca se comparte una conexión
entre hilos), así que esta escritura ocasional concurrente con el
recolector, si está corriendo, es segura.
"""
from PySide6.QtCore import QThread, Signal

from base_de_datos import OSRSBaseDatos
from recolector import _entrenar_desde_config


class HiloEntrenamiento(QThread):
    terminado = Signal(bool, str)  # (éxito, model_id)

    def __init__(self, db_path, cfg, calcular_metricas_horizonte=False, parent=None):
        super().__init__(parent)
        self.db_path = db_path
        self.cfg = cfg
        self.calcular_metricas_horizonte = calcular_metricas_horizonte

    def run(self):
        db = OSRSBaseDatos(self.db_path)
        try:
            exito = _entrenar_desde_config(
                db, self.cfg, guardar_en_disco=True,
                calcular_metricas_horizonte=self.calcular_metricas_horizonte,
            )
        except Exception:
            exito = False
        self.terminado.emit(exito, self.cfg['model_id'])
