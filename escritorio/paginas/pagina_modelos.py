"""
paginas/pagina_modelos.py — alta/gestión de modelos definidos por el
usuario (base_de_datos.modelos_config, ver workstream 1/2 del plan). Es la
pieza central de "libertad de elegir los ítems y ventanas deseadas" del
plan: acá es donde el usuario arma sus propios modelos. No hay ningún
modelo sembrado por default (ver base_de_datos._migrar_esquema) ni
protegido contra borrado — todo lo que aparece en la tabla lo creó el
usuario, y todo se puede eliminar desde acá.

Deliberadamente simple (ver la corrección del usuario sobre no repetir la
sobrecarga de información del dashboard actual): sin selector de modo de
evaluación, sin tabla de MAE/RMSE cruda — el estado de calidad de cada
modelo se resume en una frase corta (_resumen_calidad), y los parámetros
técnicos (lags, medias móviles, umbral del clasificador) no se exponen en
el formulario de alta, quedan en los defaults sensatos de entrenador.py.

Revertido a un flujo único y más simple (pedido explícito del usuario):
"+ Nuevo modelo" crea SIEMPRE un par (un regresor + un clasificador) sobre
los mismos ítems, cadencia horaria, tabla precios_1h. Antes existía
también "+ Nuevo grupo" (regresor+clasificador × 3 granularidades, 6
modelos) y un selector de tipo/cadencia en el alta simple; se sacaron de
la UI. Lo único configurable ahora es la ventana de historial y los
ítems — el resto queda fijo en `_crear_par_modelos`, no porque
base_de_datos.crear_modelo_config/entrenador.py hayan perdido esa
flexibilidad (siguen aceptando tipo/cadencia/tabla arbitrarios, nada de
eso se tocó), sino porque esta pantalla ya no la expone.
"""
import re
import sqlite3
from datetime import datetime

import pandas as pd
from PySide6.QtWidgets import (
    QDialog, QDialogButtonBox, QDoubleSpinBox, QFormLayout,
    QHBoxLayout, QLabel, QLineEdit, QMessageBox, QProgressBar, QPushButton,
    QTableView, QVBoxLayout, QWidget,
)

from base_de_datos import MAX_ITEMS_POR_MODELO
from escritorio.hilo_entrenamiento import HiloEntrenamiento
from escritorio.hilo_walkforward import HiloWalkForward
from escritorio.widgets.selector_items import SelectorItems
from escritorio.widgets.tabla_dataframe import crear_tabla

# Fijo para todo modelo creado desde acá -- ver el docstring del módulo.
# 90d en 1h coincide con la retención real de precios_1h (ver
# mantenimiento.RETENCION_DIAS), no es un número arbitrario.
TABLA_FIJA = 'precios_1h'
CADENCIA_FIJA = 'horaria'
VENTANA_DIAS_DEFAULT = 90.0

# No hay ningún modelo protegido/no-eliminable (pedido explícito del
# usuario: "no quiero que haya ningún modelo por default. todos tienen que
# poder eliminarse") — modelos_config ya no viene sembrada con nada al
# arrancar (ver base_de_datos._migrar_esquema), así que no hace falta
# distinguir "modelos del sistema" de "modelos del usuario": todos son del
# usuario, y cualquier fila de la tabla se puede eliminar desde acá.

# Sin columnas de 'tipo'/'cadencia' (pedido explícito del usuario: no
# aportan nada ahora que son valores fijos para todo lo creado desde acá
# -- cadencia siempre 'horaria', y tipo ya se distingue en el propio
# nombre, "... (regresor)"/"... (clasificador)", ver _crear_par_modelos).
COLUMNAS_TABLA = [
    ('nombre', 'Nombre'),
    ('estado', 'Estado'),
    ('calidad', 'Calidad'),
    ('ultimo_entrenamiento', 'Último entrenamiento'),
]


def _resumen_calidad(db, model_id):
    """
    Frase corta sobre qué tan bien viene funcionando el modelo, en vez de
    una tabla de métricas crudas. Se basa en accuracy_direccional de
    model_metrics (item_id IS NULL, horizonte_horas=1) -- comparable entre
    regresor y clasificador porque entrenador.py calcula esa métrica para
    los dos (ver entrenar_modelo_global/entrenar_clasificador_direccional).

    Prioriza el PROMEDIO de todas las corridas walkforward sobre la última
    holdout: el walk-forward inicial (replay_historico.ejecutar_replay_modelo,
    disparado automáticamente al crear un modelo, ver
    PaginaModelos._encolar_walkforward) deja una fila de métricas agregada
    POR CHECKPOINT -- promediarlas es una lectura mucho más representativa
    que un solo split holdout, y para un modelo recién creado puede ser lo
    único que exista todavía (holdout solo se genera al usar "Entrenar
    ahora" o en una corrida programada). Umbrales de partida, no
    validados exhaustivamente -- mismo criterio que otros números mágicos
    ya documentados en el proyecto (ej. UMBRAL_CLASIF_PCT en entrenador.py).
    """
    conn = sqlite3.connect(db.db_path)
    c = conn.cursor()
    c.execute(
        '''SELECT AVG(accuracy_direccional), COUNT(*) FROM model_metrics
           WHERE model_name = ? AND item_id IS NULL AND modo_evaluacion = 'walkforward'
             AND horizonte_horas = 1 AND accuracy_direccional IS NOT NULL''',
        (model_id,),
    )
    acc_wf, n_wf = c.fetchone()

    if acc_wf is not None:
        conn.close()
        etiqueta = f" (walk-forward, {n_wf} checkpoints)"
    else:
        c.execute(
            '''SELECT accuracy_direccional FROM model_metrics
               WHERE model_name = ? AND item_id IS NULL AND modo_evaluacion = 'holdout'
                 AND horizonte_horas = 1
               ORDER BY train_timestamp DESC LIMIT 1''',
            (model_id,),
        )
        fila = c.fetchone()
        conn.close()
        if fila is None:
            return "Sin entrenar todavía"
        acc_wf = fila[0]
        etiqueta = "" if acc_wf is None else " (holdout)"

    if acc_wf is None:
        return "Entrenado (sin métrica de dirección)"
    if acc_wf >= 0.55:
        return f"Buena señal ({acc_wf:.0%}){etiqueta}"
    if acc_wf >= 0.48:
        return f"Regular ({acc_wf:.0%}){etiqueta}"
    return f"Débil, cerca del azar ({acc_wf:.0%}){etiqueta}"


def _formatear_fecha(ts):
    return "Nunca" if ts is None else datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M')


def _generar_ids_par(db, nombre):
    """
    Slugs 'model_id_regresor'/'model_id_clasificador' a partir del nombre
    elegido por el usuario, con un sufijo numérico compartido si ya
    existe alguno de los dos -- model_id es el mismo string que
    entrenador.py usa como model_name/model_version en todo el resto del
    pipeline (ver base_de_datos.crear_modelo_config), así que tiene que
    ser único. Prefijo 'custom_' para no poder chocar nunca con los 3
    model_id productivos reservados (MODELOS_PROTEGIDOS). Devuelve
    (id_regresor, id_clasificador), siempre con el mismo sufijo -- así
    quedan visiblemente emparejados en la tabla ("mi_modelo_regresor" /
    "mi_modelo_clasificador"), no con sufijos independientes por tipo.
    """
    base = 'custom_' + re.sub(r'[^a-z0-9]+', '_', nombre.strip().lower()).strip('_')
    if base == 'custom_':
        base = 'custom_modelo'
    sufijo = 0
    while True:
        candidato = base if sufijo == 0 else f"{base}_{sufijo}"
        id_regresor, id_clasificador = f"{candidato}_regresor", f"{candidato}_clasificador"
        if db.obtener_modelo_config(id_regresor) is None and db.obtener_modelo_config(id_clasificador) is None:
            return id_regresor, id_clasificador
        sufijo += 1


class _DialogoNuevoModelo(QDialog):
    """
    Único diálogo de alta: nombre, ítems y ventana de historial son lo
    único configurable (ver el docstring del módulo) -- tipo, cadencia y
    tabla quedan fijos (regresor + clasificador, horaria, precios_1h) y no
    se muestran acá.
    """
    def __init__(self, db_path, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Nuevo modelo")
        self.resize(480, 560)

        layout = QVBoxLayout(self)
        aviso = QLabel(
            "Crea un par de modelos (un regresor + un clasificador) sobre los ítems elegidos, "
            "reentrenados cada hora sobre precios de 1h. Elegí un nombre, los ítems y la ventana "
            "de historial a usar."
        )
        aviso.setWordWrap(True)
        layout.addWidget(aviso)

        layout.addWidget(QLabel("Nombre:"))
        self.campo_nombre = QLineEdit()
        layout.addWidget(self.campo_nombre)

        layout.addWidget(QLabel(f"Ítems (máximo {MAX_ITEMS_POR_MODELO}):"))
        self.selector = SelectorItems(db_path, max_seleccion=MAX_ITEMS_POR_MODELO)
        layout.addWidget(self.selector, stretch=1)

        form = QFormLayout()
        self.campo_ventana = QDoubleSpinBox()
        self.campo_ventana.setRange(0.1, 365.0)
        self.campo_ventana.setDecimals(1)
        self.campo_ventana.setValue(VENTANA_DIAS_DEFAULT)
        form.addRow("Ventana de historial (días):", self.campo_ventana)
        layout.addLayout(form)

        botones = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        botones.accepted.connect(self._validar_y_aceptar)
        botones.rejected.connect(self.reject)
        layout.addWidget(botones)

    def _validar_y_aceptar(self):
        if not self.campo_nombre.text().strip():
            QMessageBox.warning(self, "Falta el nombre", "Ponele un nombre al modelo.")
            return
        if not self.selector.seleccionados():
            QMessageBox.warning(self, "Sin ítems", "Elegí al menos un ítem.")
            return
        self.accept()

    def valores(self):
        return {
            'nombre': self.campo_nombre.text().strip(),
            'item_ids': self.selector.seleccionados(),
            'ventana_dias': self.campo_ventana.value(),
        }


def _crear_par_modelos(db, nombre, item_ids, ventana_dias):
    """
    Crea el par regresor+clasificador de un modelo (ver
    _DialogoNuevoModelo) -- tipo/cadencia/tabla fijos (ver el docstring
    del módulo), item_ids/ventana_dias los que eligió el usuario. No
    aborta ante el primer error (ej. el segundo choca con algo
    inesperado): intenta los dos y junta lo que falló, para que el
    llamador pueda avisar exactamente cuál se creó.

    Devuelve (creados, errores): `creados` es la lista de model_id dados
    de alta; `errores` es una lista de (model_id, excepción) para los que
    no se pudieron crear.
    """
    id_regresor, id_clasificador = _generar_ids_par(db, nombre)
    creados = []
    errores = []
    for model_id, tipo, etiqueta_tipo in (
        (id_regresor, 'regresor', 'regresor'),
        (id_clasificador, 'clasificador', 'clasificador'),
    ):
        try:
            db.crear_modelo_config(
                model_id=model_id, nombre=f"{nombre} ({etiqueta_tipo})",
                tipo=tipo, cadencia=CADENCIA_FIJA, modo_seleccion='manual',
                item_ids=item_ids, tabla=TABLA_FIJA, ventana_dias=ventana_dias,
            )
            creados.append(model_id)
        except (ValueError, sqlite3.IntegrityError) as e:
            errores.append((model_id, e))
    return creados, errores


class PaginaModelos(QWidget):
    def __init__(self, db, db_path, parent=None):
        super().__init__(parent)
        self.db = db
        self.db_path = db_path
        self._modelos = []  # cfg dicts, sincronizados 1:1 con las filas de la tabla
        self._hilo_entrenamiento = None
        self._hilo_walkforward = None
        self._cola_walkforward = []  # model_id pendientes, se procesan de a uno

        layout = QVBoxLayout(self)

        fila_botones = QHBoxLayout()
        boton_nuevo = QPushButton("+ Nuevo modelo")
        boton_nuevo.clicked.connect(self._nuevo_modelo)
        self.boton_entrenar = QPushButton("Entrenar ahora")
        self.boton_entrenar.clicked.connect(self._entrenar_seleccionado)
        self.boton_pausar = QPushButton("Pausar / activar")
        self.boton_pausar.clicked.connect(self._alternar_estado_seleccionado)
        self.boton_eliminar = QPushButton("Eliminar")
        self.boton_eliminar.clicked.connect(self._eliminar_seleccionado)
        for boton in (boton_nuevo, self.boton_entrenar, self.boton_pausar, self.boton_eliminar):
            fila_botones.addWidget(boton)
        fila_botones.addStretch()
        layout.addLayout(fila_botones)

        self.label_estado = QLabel("")
        self.label_estado.setWordWrap(True)  # ver pagina_configuracion.py: un mensaje largo no debe fijar el ancho mínimo de la ventana
        layout.addWidget(self.label_estado)

        self.barra_walkforward = QProgressBar()
        self.barra_walkforward.setVisible(False)
        layout.addWidget(self.barra_walkforward)

        self.tabla, self.modelo_tabla = crear_tabla(columnas=COLUMNAS_TABLA)
        self.tabla.setSelectionMode(QTableView.SingleSelection)
        self.tabla.setSelectionBehavior(QTableView.SelectRows)
        layout.addWidget(self.tabla)

        self._refrescar()

    def _refrescar(self):
        """
        Envuelto en try/except: una falla acá (ej. la DB quedó bloqueada
        un instante por una escritura concurrente del recolector) no debe
        dejar la pestaña en un estado roto — se avisa y se mantiene lo que
        ya se estaba mostrando, en vez de propagar la excepción hacia
        arriba (podría interrumpir un click del usuario sin explicación).
        """
        try:
            self._modelos = self.db.listar_modelos_config()
            filas = [
                {
                    'nombre': cfg['nombre'],
                    'estado': 'Activo' if cfg['estado'] == 'activo' else 'Pausado',
                    'calidad': _resumen_calidad(self.db, cfg['model_id']),
                    'ultimo_entrenamiento': _formatear_fecha(cfg['ultimo_entrenamiento_ts']),
                }
                for cfg in self._modelos
            ]
            self.modelo_tabla.set_dataframe(pd.DataFrame(filas))
        except Exception as e:
            self.label_estado.setText(f"No se pudo actualizar la lista de modelos: {e}")

    def _fila_seleccionada(self):
        indices = self.tabla.selectionModel().selectedRows()
        if not indices:
            return None
        fila = indices[0].row()
        return self._modelos[fila] if 0 <= fila < len(self._modelos) else None

    def _nuevo_modelo(self):
        dialogo = _DialogoNuevoModelo(self.db_path, self)
        if dialogo.exec() != QDialog.Accepted:
            return
        valores = dialogo.valores()
        creados, errores = _crear_par_modelos(
            self.db, valores['nombre'], valores['item_ids'], valores['ventana_dias'],
        )

        self._refrescar()
        if creados:
            self.label_estado.setText(
                f"Modelo '{valores['nombre']}' creado: {len(creados)} de 2 "
                f"({'; '.join(creados)}) — arrancando walk-forward inicial de cada uno..."
            )
            self._encolar_walkforward(creados)
        if errores:
            detalle = "\n".join(f"- {model_id}: {e}" for model_id, e in errores)
            QMessageBox.warning(
                self, "Algo no se pudo crear",
                f"{len(errores)} de 2 fallaron:\n{detalle}",
            )
        if not creados and not errores:
            self.label_estado.setText("No se creó ningún modelo.")

    def _entrenar_seleccionado(self):
        cfg = self._fila_seleccionada()
        if cfg is None:
            QMessageBox.information(self, "Elegí un modelo", "Seleccioná una fila de la tabla primero.")
            return
        if self._hilo_entrenamiento is not None and self._hilo_entrenamiento.isRunning():
            QMessageBox.information(self, "Entrenamiento en curso", "Esperá a que termine el actual antes de lanzar otro.")
            return

        self.boton_entrenar.setEnabled(False)
        self.label_estado.setText(f"Entrenando '{cfg['nombre']}'... esto puede tardar unos minutos.")
        self._hilo_entrenamiento = HiloEntrenamiento(self.db_path, cfg)
        self._hilo_entrenamiento.terminado.connect(self._al_terminar_entrenamiento)
        self._hilo_entrenamiento.start()

    def _al_terminar_entrenamiento(self, exito, model_id, mensaje):
        self.boton_entrenar.setEnabled(True)
        if exito:
            self.label_estado.setText(f"'{model_id}' entrenado correctamente.")
        else:
            self.label_estado.setText(f"'{model_id}' no se pudo entrenar todavía — {mensaje}.")
        self._refrescar()

    def _encolar_walkforward(self, model_ids):
        """
        Agrega `model_ids` a la cola de walk-forward inicial (ver
        replay_historico.ejecutar_replay_modelo) y arranca el
        procesamiento si no hay uno corriendo ya. Se procesan de a UNO por
        vez, no en paralelo — varios fits de XGBoost compitiendo por CPU
        al mismo tiempo no aporta nada, solo hace que todos tarden más.
        """
        self._cola_walkforward.extend(model_ids)
        if self._hilo_walkforward is None or not self._hilo_walkforward.isRunning():
            self._procesar_siguiente_walkforward()

    def _procesar_siguiente_walkforward(self):
        if not self._cola_walkforward:
            self.barra_walkforward.setVisible(False)
            return
        model_id = self._cola_walkforward.pop(0)
        self.barra_walkforward.setVisible(True)
        self.barra_walkforward.setRange(0, 0)  # indeterminado hasta el primer tick de progreso real
        self.label_estado.setText(f"Walk-forward inicial de '{model_id}'...")
        self._hilo_walkforward = HiloWalkForward(self.db_path, model_id)
        self._hilo_walkforward.progreso.connect(self._on_progreso_walkforward)
        self._hilo_walkforward.terminado.connect(self._on_terminado_walkforward)
        self._hilo_walkforward.start()

    def _on_progreso_walkforward(self, fase, actual, total):
        if total > 0:
            self.barra_walkforward.setRange(0, total)
            self.barra_walkforward.setValue(actual)
            # Mismo detalle de QProgressBar.setFormat que pagina_inicio.py:
            # %p ya es solo el número, un %% no es un escape de "%
            # literal" -- un solo % suelto después de %p alcanza.
            self.barra_walkforward.setFormat(f"walk-forward: %v/%m (%p%)")

    def _on_terminado_walkforward(self, model_id, n_corridos, n_saltados, n_fallidos):
        total = n_corridos + n_saltados + n_fallidos
        self.label_estado.setText(
            f"Walk-forward inicial de '{model_id}' completo: {n_corridos} corridos, "
            f"{n_saltados} ya existentes, {n_fallidos} fallidos, de {total} checkpoints."
        )
        self._refrescar()
        self._procesar_siguiente_walkforward()

    def hilo_activo(self):
        entrenamiento_activo = self._hilo_entrenamiento is not None and self._hilo_entrenamiento.isRunning()
        walkforward_activo = self._hilo_walkforward is not None and self._hilo_walkforward.isRunning()
        return entrenamiento_activo or walkforward_activo

    def esperar_para_cerrar(self):
        """Llamado desde VentanaPrincipal.closeEvent. Un entrenamiento en
        curso se espera hasta 30s (un fit real puede tardar más que el
        hilo del recolector) — un walk-forward en curso puede ser de
        miles de checkpoints, esperar a que termine solo no es razonable
        al cerrar la ventana, así que se le pide que pare en el próximo
        checkpoint (HiloWalkForward.detener()) en vez de eso."""
        if self._hilo_entrenamiento is not None and self._hilo_entrenamiento.isRunning():
            self._hilo_entrenamiento.wait(30000)
        if self._hilo_walkforward is not None and self._hilo_walkforward.isRunning():
            self._hilo_walkforward.detener()
            self._hilo_walkforward.wait(10000)

    def _alternar_estado_seleccionado(self):
        cfg = self._fila_seleccionada()
        if cfg is None:
            QMessageBox.information(self, "Elegí un modelo", "Seleccioná una fila de la tabla primero.")
            return
        nuevo_estado = 'pausado' if cfg['estado'] == 'activo' else 'activo'
        try:
            self.db.actualizar_modelo_config(cfg['model_id'], estado=nuevo_estado)
        except ValueError as e:
            QMessageBox.warning(self, "No se pudo activar", str(e))
            return
        self._refrescar()

    def _eliminar_seleccionado(self):
        cfg = self._fila_seleccionada()
        if cfg is None:
            QMessageBox.information(self, "Elegí un modelo", "Seleccioná una fila de la tabla primero.")
            return
        respuesta = QMessageBox.question(
            self, "Confirmar",
            f"¿Eliminar el modelo '{cfg['nombre']}'? Se conserva su historial en model_metrics, "
            "pero deja de reentrenarse.",
        )
        if respuesta != QMessageBox.Yes:
            return
        self.db.eliminar_modelo_config(cfg['model_id'])
        self._refrescar()
