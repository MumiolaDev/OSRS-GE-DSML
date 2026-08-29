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
    QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFormLayout,
    QHBoxLayout, QLabel, QLineEdit, QMessageBox, QProgressBar, QPushButton,
    QTableView, QVBoxLayout, QWidget,
)

from base_de_datos import MAX_ITEMS_POR_MODELO
from entrenador import MODEL_NAME_HORARIO, MODEL_NAME_DIARIO, MODEL_NAME_CLASIF_F2P_100GP
from escritorio.hilo_entrenamiento import HiloEntrenamiento
from escritorio.hilo_walkforward import HiloWalkForward
from escritorio.widgets.selector_items import SelectorItems
from escritorio.widgets.tabla_dataframe import crear_tabla

# (tabla, etiqueta, ventana_dias sugerida por default) para el diálogo de
# grupo -- 90d en 1h coincide con la retención real de precios_1h (ver
# mantenimiento.RETENCION_DIAS), 30d en 6h y 1d en 5m son puntos de
# partida razonables, no valores validados; el usuario los puede cambiar.
# 5m con una ventana muy corta (ej. 1 día = 288 filas) da un dataset chico
# -- el modelo de esa granularidad va a ser el menos confiable de los 3,
# no hay forma de evitarlo sin alargar la ventana.
GRANULARIDADES = [
    ('precios_5m', '5 minutos', 1.0),
    ('precios_1h', '1 hora', 90.0),
    ('precios_6h', '6 horas', 30.0),
]
SUFIJO_GRANULARIDAD = {'precios_5m': '5m', 'precios_1h': '1h', 'precios_6h': '6h'}

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


class _DialogoNuevoGrupo(QDialog):
    """
    Crea, de una sola vez, un regresor + un clasificador para cada una de
    las 3 granularidades (5m/1h/6h) sobre el mismo grupo de ítems -- el
    flujo que pidió el usuario ("el par de runas cosmic y nature"): 6
    modelos en total, cadencia 'manual' por default (crear un grupo ya usa
    6 modelos; si fueran horaria/diaria un solo grupo casi agotaría
    MAX_MODELOS_ACTIVOS) — cada uno arranca su walk-forward inicial apenas
    se crea (ver PaginaModelos._encolar_walkforward).
    """
    def __init__(self, db_path, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Nuevo grupo de ítems relacionados")
        self.resize(520, 600)

        layout = QVBoxLayout(self)
        aviso = QLabel(
            "Elegí un grupo de ítems relacionados (ej. Nature rune + Cosmic rune) — se crean 6 "
            "modelos (regresor + clasificador para 5 minutos, 1 hora y 6 horas), cada uno con su "
            "propia ventana de historial. Todos arrancan con cadencia manual y un walk-forward "
            "inicial sobre los datos ya disponibles."
        )
        aviso.setWordWrap(True)
        layout.addWidget(aviso)

        layout.addWidget(QLabel("Nombre del grupo:"))
        self.campo_nombre = QLineEdit()
        layout.addWidget(self.campo_nombre)

        layout.addWidget(QLabel(f"Ítems (máximo {MAX_ITEMS_POR_MODELO}):"))
        self.selector = SelectorItems(db_path, max_seleccion=MAX_ITEMS_POR_MODELO)
        layout.addWidget(self.selector, stretch=1)

        layout.addWidget(QLabel("Ventana de historial por granularidad (días):"))
        form = QFormLayout()
        self.campos_ventana = {}
        for tabla, etiqueta, default in GRANULARIDADES:
            spin = QDoubleSpinBox()
            spin.setRange(0.1, 365.0)
            spin.setDecimals(1)
            spin.setValue(default)
            form.addRow(f"{etiqueta}:", spin)
            self.campos_ventana[tabla] = spin
        layout.addLayout(form)

        botones = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        botones.accepted.connect(self._validar_y_aceptar)
        botones.rejected.connect(self.reject)
        layout.addWidget(botones)

    def _validar_y_aceptar(self):
        if not self.campo_nombre.text().strip():
            QMessageBox.warning(self, "Falta el nombre", "Ponele un nombre al grupo.")
            return
        if not self.selector.seleccionados():
            QMessageBox.warning(self, "Sin ítems", "Elegí al menos un ítem.")
            return
        self.accept()

    def valores(self):
        return {
            'nombre': self.campo_nombre.text().strip(),
            'item_ids': self.selector.seleccionados(),
            'ventanas': {tabla: self.campos_ventana[tabla].value() for tabla, _, _ in GRANULARIDADES},
        }


def _crear_modelos_grupo(db, nombre_grupo, item_ids, ventanas):
    """
    Crea los 6 modelos de un grupo (ver _DialogoNuevoGrupo) — un
    regresor y un clasificador por cada granularidad. No aborta ante el
    primer error (ej. el nombre ya generó un model_id que choca con algo
    inesperado): intenta los 6 y junta lo que falló, para que el
    llamador pueda avisar exactamente cuáles se crearon.

    Devuelve (creados, errores): `creados` es la lista de model_id dados
    de alta; `errores` es una lista de (model_id, excepción) para los que
    no se pudieron crear.
    """
    base = _generar_model_id(db, nombre_grupo)
    creados = []
    errores = []
    for tabla, etiqueta, _ in GRANULARIDADES:
        sufijo_tabla = SUFIJO_GRANULARIDAD[tabla]
        ventana_dias = ventanas[tabla]
        for tipo, sufijo_tipo in (('regresor', 'regresor'), ('clasificador', 'clasif')):
            model_id = f"{base}_{sufijo_tabla}_{sufijo_tipo}"
            try:
                db.crear_modelo_config(
                    model_id=model_id, nombre=f"{nombre_grupo} ({etiqueta}, {tipo})",
                    tipo=tipo, cadencia='manual', modo_seleccion='manual',
                    item_ids=item_ids, tabla=tabla, ventana_dias=ventana_dias,
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
        boton_nuevo_grupo = QPushButton("+ Nuevo grupo")
        boton_nuevo_grupo.clicked.connect(self._nuevo_grupo)
        self.boton_entrenar = QPushButton("Entrenar ahora")
        self.boton_entrenar.clicked.connect(self._entrenar_seleccionado)
        self.boton_pausar = QPushButton("Pausar / activar")
        self.boton_pausar.clicked.connect(self._alternar_estado_seleccionado)
        self.boton_eliminar = QPushButton("Eliminar")
        self.boton_eliminar.clicked.connect(self._eliminar_seleccionado)
        for boton in (boton_nuevo, boton_nuevo_grupo, self.boton_entrenar, self.boton_pausar, self.boton_eliminar):
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
        self.label_estado.setText(f"Modelo '{valores['nombre']}' creado — arrancando walk-forward inicial...")
        self._encolar_walkforward([model_id])

    def _nuevo_grupo(self):
        dialogo = _DialogoNuevoGrupo(self.db_path, self)
        if dialogo.exec() != QDialog.Accepted:
            return
        valores = dialogo.valores()
        creados, errores = _crear_modelos_grupo(self.db, valores['nombre'], valores['item_ids'], valores['ventanas'])

        self._refrescar()
        if creados:
            self.label_estado.setText(
                f"Grupo '{valores['nombre']}' creado: {len(creados)} modelo(s) "
                f"({'; '.join(creados)}) — arrancando walk-forward inicial de cada uno..."
            )
            self._encolar_walkforward(creados)
        if errores:
            detalle = "\n".join(f"- {model_id}: {e}" for model_id, e in errores)
            QMessageBox.warning(
                self, "Algunos modelos del grupo no se pudieron crear",
                f"{len(errores)} de 6 fallaron:\n{detalle}",
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
