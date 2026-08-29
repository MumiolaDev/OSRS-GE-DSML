"""
paginas/pagina_modelos.py — alta/gestión de modelos definidos por el
usuario (base_de_datos.modelos_config, ver workstream 1/2 del plan). Es la
pieza central de "libertad de elegir los ítems y ventanas deseadas" del
plan: acá es donde el usuario arma un modelo propio, no solo mira los
productivos.

Deliberadamente simple (ver la corrección del usuario sobre no repetir la
sobrecarga de información del dashboard actual): sin selector de modo de
evaluación, sin tabla de MAE/RMSE cruda — el estado de calidad de cada
modelo se resume en una frase corta (_resumen_calidad), y los parámetros
técnicos (lags, medias móviles, umbral del clasificador) no se exponen en
el formulario de alta, quedan en los defaults sensatos de entrenador.py.
"""
import re
import sqlite3
from datetime import datetime

import pandas as pd
from PySide6.QtWidgets import (
    QComboBox, QDialog, QDialogButtonBox, QHBoxLayout, QLabel, QLineEdit,
    QMessageBox, QPushButton, QTableView, QVBoxLayout, QWidget,
)

from base_de_datos import MAX_ITEMS_POR_MODELO
from entrenador import MODEL_NAME_HORARIO, MODEL_NAME_DIARIO, MODEL_NAME_CLASIF_F2P_100GP
from escritorio.hilo_entrenamiento import HiloEntrenamiento
from escritorio.widgets.selector_items import SelectorItems
from escritorio.widgets.tabla_dataframe import crear_tabla

# Los 3 modelos productivos del proyecto (sembrados en la migración, ver
# base_de_datos._sembrar_modelos_productivos) no se pueden eliminar desde
# acá -- pausarlos alcanza para dejar de reentrenarlos, y borrarlos
# rompería recolector.job_horario/job_diario o las alertas si alguien lo
# hace por error.
MODELOS_PROTEGIDOS = {MODEL_NAME_HORARIO, MODEL_NAME_DIARIO, MODEL_NAME_CLASIF_F2P_100GP}

COLUMNAS_TABLA = [
    ('nombre', 'Nombre'),
    ('tipo', 'Tipo'),
    ('cadencia', 'Cadencia'),
    ('estado', 'Estado'),
    ('calidad', 'Calidad'),
    ('ultimo_entrenamiento', 'Último entrenamiento'),
]


def _resumen_calidad(db, model_id):
    """
    Frase corta sobre qué tan bien viene funcionando el modelo, en vez de
    una tabla de métricas crudas. Se basa en accuracy_direccional de la
    última corrida agregada (item_id IS NULL, modo_evaluacion='holdout',
    horizonte_horas=1) -- comparable entre regresor y clasificador porque
    entrenador.py calcula esa métrica para los dos (ver
    entrenar_modelo_global/entrenar_clasificador_direccional). Umbrales de
    partida, no validados exhaustivamente -- mismo criterio que otros
    números mágicos ya documentados en el proyecto (ej. UMBRAL_CLASIF_PCT
    en entrenador.py).
    """
    conn = sqlite3.connect(db.db_path)
    c = conn.cursor()
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
    acc = fila[0]
    if acc is None:
        return "Entrenado (sin métrica de dirección)"
    if acc >= 0.55:
        return f"Buena señal ({acc:.0%})"
    if acc >= 0.48:
        return f"Regular ({acc:.0%})"
    return f"Débil, cerca del azar ({acc:.0%})"


def _formatear_fecha(ts):
    return "Nunca" if ts is None else datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M')


def _generar_model_id(db, nombre):
    """
    Slug legible a partir del nombre elegido por el usuario, con un
    sufijo numérico si ya existe -- model_id es el mismo string que
    entrenador.py usa como model_name/model_version en todo el resto del
    pipeline (ver base_de_datos.crear_modelo_config), así que tiene que
    ser único. Prefijo 'custom_' para no poder chocar nunca con los 3
    model_id productivos reservados (MODELOS_PROTEGIDOS).
    """
    base = 'custom_' + re.sub(r'[^a-z0-9]+', '_', nombre.strip().lower()).strip('_')
    if base == 'custom_':
        base = 'custom_modelo'
    candidato = base
    sufijo = 1
    while db.obtener_modelo_config(candidato) is not None:
        sufijo += 1
        candidato = f"{base}_{sufijo}"
    return candidato


class _DialogoNuevoModelo(QDialog):
    def __init__(self, db_path, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Nuevo modelo")
        self.resize(480, 520)

        layout = QVBoxLayout(self)

        layout.addWidget(QLabel("Nombre:"))
        self.campo_nombre = QLineEdit()
        layout.addWidget(self.campo_nombre)

        layout.addWidget(QLabel("Tipo:"))
        self.combo_tipo = QComboBox()
        self.combo_tipo.addItem("Regresor (predice el precio)", 'regresor')
        self.combo_tipo.addItem("Clasificador (predice si sube, baja o queda estable)", 'clasificador')
        layout.addWidget(self.combo_tipo)

        layout.addWidget(QLabel(f"Ítems (máximo {MAX_ITEMS_POR_MODELO}):"))
        self.selector = SelectorItems(db_path, max_seleccion=MAX_ITEMS_POR_MODELO)
        layout.addWidget(self.selector, stretch=1)

        layout.addWidget(QLabel("Cadencia:"))
        self.combo_cadencia = QComboBox()
        self.combo_cadencia.addItem("Horaria (se reentrena cada hora)", 'horaria')
        self.combo_cadencia.addItem("Diaria (se reentrena una vez al día)", 'diaria')
        self.combo_cadencia.addItem("Manual (solo cuando yo lo pida)", 'manual')
        layout.addWidget(self.combo_cadencia)

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
            'tipo': self.combo_tipo.currentData(),
            'item_ids': self.selector.seleccionados(),
            'cadencia': self.combo_cadencia.currentData(),
        }


class PaginaModelos(QWidget):
    def __init__(self, db, db_path, parent=None):
        super().__init__(parent)
        self.db = db
        self.db_path = db_path
        self._modelos = []  # cfg dicts, sincronizados 1:1 con las filas de la tabla
        self._hilo_entrenamiento = None

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
                    'tipo': 'Regresor' if cfg['tipo'] == 'regresor' else 'Clasificador',
                    'cadencia': cfg['cadencia'].capitalize(),
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
        model_id = _generar_model_id(self.db, valores['nombre'])
        try:
            self.db.crear_modelo_config(
                model_id=model_id, nombre=valores['nombre'], tipo=valores['tipo'],
                cadencia=valores['cadencia'], modo_seleccion='manual', item_ids=valores['item_ids'],
            )
        except ValueError as e:
            # Topes técnicos (MAX_ITEMS_POR_MODELO / MAX_MODELOS_ACTIVOS).
            QMessageBox.warning(self, "No se pudo crear el modelo", str(e))
            return
        except sqlite3.IntegrityError as e:
            # model_id duplicado -- no debería pasar en la práctica
            # (_generar_model_id ya chequea unicidad justo antes), pero
            # sqlite3.IntegrityError está documentado como posible en
            # crear_modelo_config y no había ningún catch para eso.
            QMessageBox.warning(self, "No se pudo crear el modelo", f"Ya existe un modelo con ese identificador: {e}")
            return
        self._refrescar()
        self.label_estado.setText(f"Modelo '{valores['nombre']}' creado — usá \"Entrenar ahora\" para la primera corrida.")

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

    def hilo_activo(self):
        return self._hilo_entrenamiento is not None and self._hilo_entrenamiento.isRunning()

    def esperar_para_cerrar(self):
        """Llamado desde VentanaPrincipal.closeEvent — espera (hasta 30s,
        un entrenamiento real puede tardar más que el hilo del recolector)
        a que termine un entrenamiento en curso antes de dejar cerrar la
        app, para no matar el hilo de XGBoost a mitad de un fit."""
        if self.hilo_activo():
            self._hilo_entrenamiento.wait(30000)

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
        if cfg['model_id'] in MODELOS_PROTEGIDOS:
            QMessageBox.warning(
                self, "No se puede eliminar",
                "Este es uno de los modelos productivos del proyecto — pausalo en vez de eliminarlo.",
            )
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
