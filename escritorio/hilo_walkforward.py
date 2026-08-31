"""
hilo_walkforward.py — QThread para correr el walk-forward de un modelo
(replay_historico.ejecutar_replay_modelo) en segundo plano, sin bloquear
la ventana. Se usa tanto para el walk-forward inicial al crear un modelo
como para rehacerlo después sobre una ventana elegida por el usuario (ver
pagina_modelos.PaginaModelos._rehacer_walkforward) — mismo hilo, mismo
mecanismo de reanudación (existe_checkpoint salta lo ya corrido), solo
cambia si se le pasa desde_ts/hasta_ts o no.

Un walk-forward de varios meses a resolución horaria son miles de
checkpoints, cada uno un fit de XGBoost — puede tardar minutos. Mismo
patrón que hilo_entrenamiento.py (independiente de HiloRecolector, no le
encola la tarea al hilo del recolector); acá además emite progreso real
(vía replay_historico's on_progreso) y es interrumpible (debe_detener),
igual que el relleno de huecos de pagina_inicio.py.
"""
from PySide6.QtCore import QThread, Signal

from base_de_datos import OSRSBaseDatos
from replay_historico import ejecutar_replay_modelo


class HiloWalkForward(QThread):
    progreso = Signal(str, int, int)  # (fase, actual, total) -- fase='walkforward' siempre acá
    terminado = Signal(str, int, int, int)  # (model_id, n_corridos, n_saltados, n_fallidos)

    def __init__(self, db_path, model_id, desde_ts=None, hasta_ts=None, parent=None):
        """
        desde_ts/hasta_ts (opcional): ver replay_historico.ejecutar_replay_modelo
        -- None (default, el comportamiento de siempre) recorre TODO el
        historial disponible para los ítems del modelo; un rango explícito
        acota el walk-forward a esa ventana (lo que usa "Rehacer
        walk-forward" para no tener que recorrer meses de historia de
        nuevo solo para cubrir los últimos días).
        """
        super().__init__(parent)
        self.db_path = db_path
        self.model_id = model_id
        self.desde_ts = desde_ts
        self.hasta_ts = hasta_ts
        self._detener = False

    def run(self):
        db = OSRSBaseDatos(self.db_path)
        try:
            n_corridos, n_saltados, n_fallidos = ejecutar_replay_modelo(
                db, self.model_id, desde_ts=self.desde_ts, hasta_ts=self.hasta_ts,
                on_progreso=self.progreso.emit, debe_detener=lambda: self._detener,
            )
        except Exception:
            n_corridos = n_saltados = n_fallidos = 0
        self.terminado.emit(self.model_id, n_corridos, n_saltados, n_fallidos)

    def detener(self):
        """Pide que el walk-forward se corte en el próximo checkpoint (no
        a mitad de un fit en curso) — mismo patrón que HiloRecolector."""
        self._detener = True
