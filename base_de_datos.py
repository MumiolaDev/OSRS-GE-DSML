import sqlite3
import pandas as pd
import time

DB_PATH = "data/osrs_ge.db"

# Topes técnicos para modelos_config, aplicados en crear_modelo_config —
# no son solo una sugerencia de la UI, viven acá para que cualquier
# llamador (la app de escritorio, o un script futuro) quede protegido por
# igual. MAX_ITEMS_POR_MODELO: build_training_set + el fit de XGBoost sobre
# más ítems empieza a tardar minutos en una PC hogareña (el modelo global
# productivo ya usa 200, pero corre una sola vez por cadencia — con varios
# modelos de usuario del mismo tamaño conviviendo, job_horario/job_diario
# se alargarían demasiado). MAX_MODELOS_ACTIVOS: cada modelo con cadencia
# horaria/diaria se reentrena en serie dentro del mismo job (ver
# recolector.job_horario/job_diario) — más modelos activos simultáneos
# alarga esos jobs proporcionalmente. Ninguno de los dos es un límite
# validado contra hardware real, son puntos de partida conservadores;
# ajustar si en la práctica resultan muy restrictivos u optimistas.
MAX_ITEMS_POR_MODELO = 50
MAX_MODELOS_ACTIVOS = 8

class OSRSBaseDatos:

    def __init__(self, db_path = DB_PATH ):
        
        self.db_path = db_path

        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        # WAL permite leer (notebook / futura app) mientras el recolector escribe.
        c.execute('PRAGMA journal_mode=WAL')

        # Crea las tablas si no existen.
        # Tabla items: catálogo de ítems (id -> nombre, members, límite de compra)
        c.execute('''CREATE TABLE IF NOT EXISTS items
                    (item_id INTEGER PRIMARY KEY,
                    name TEXT,
                    members INTEGER,
                    buy_limit INTEGER)''')
        # Tabla precios_5m
        # PRIMARY KEY compuesta (item_id, timestamp): evita duplicados si un
        # ciclo de recolección se solapa con el anterior, y sirve de índice
        # para las consultas por ítem.
        c.execute(
            '''CREATE TABLE IF NOT EXISTS precios_5m (
                item_id INT,
                timestamp INT,
                avg_high_price INTEGER,
                avg_low_price INTEGER,
                high_volume INTEGER,
                low_volume INTEGER,
                PRIMARY KEY (item_id, timestamp)
                )'''
                )
        c.execute(
            '''CREATE INDEX IF NOT EXISTS idx_precios_5m_timestamp
                ON precios_5m (timestamp)'''
                )
        # Tabla precios_1h
        c.execute(
            '''CREATE TABLE IF NOT EXISTS precios_1h (
                item_id INT,
                timestamp INT,
                avg_high_price INTEGER,
                avg_low_price INTEGER,
                high_volume INTEGER,
                low_volume INTEGER,
                PRIMARY KEY (item_id, timestamp)
                )'''
                )
        c.execute(
            '''CREATE INDEX IF NOT EXISTS idx_precios_1h_timestamp
                ON precios_1h (timestamp)'''
                )
        # Tabla precios_6h
        c.execute(
            '''CREATE TABLE IF NOT EXISTS precios_6h (
                item_id INT,
                timestamp INT,
                avg_high_price INTEGER,
                avg_low_price INTEGER,
                high_volume INTEGER,
                low_volume INTEGER,
                PRIMARY KEY (item_id, timestamp)
                )'''
                )
        c.execute(
            '''CREATE INDEX IF NOT EXISTS idx_precios_6h_timestamp
                ON precios_6h (timestamp)'''
                )

        # Tabla precios_1h_diario: agregados diarios (avg de precios, suma de
        # volumen) de precios_1h. Destino de los datos horarios que
        # mantenimiento.archivar_datos_antiguos() purga después de la
        # ventana de retención — conserva la tendencia de largo plazo sin
        # el volumen de la resolución horaria. `fecha` es el timestamp de
        # inicio del día (UTC) al que corresponde la fila agregada.
        c.execute(
            '''CREATE TABLE IF NOT EXISTS precios_1h_diario (
                item_id INT,
                fecha INT,
                avg_high_price INTEGER,
                avg_low_price INTEGER,
                high_volume INTEGER,
                low_volume INTEGER,
                PRIMARY KEY (item_id, fecha)
                )'''
                )
        c.execute(
            '''CREATE INDEX IF NOT EXISTS idx_precios_1h_diario_fecha
                ON precios_1h_diario (fecha)'''
                )

        # Tabla resumen_actual: "screener" materializado, una fila por ítem
        # con las métricas de metricas.py ya calculadas (margen, ROI,
        # percentil, volatilidad, etc.). Se recalcula completa cada vez que
        # corre metricas.py — no es una serie de tiempo, es el estado actual.
        # Pensada para que la futura web app la lea directo sin tener que
        # agregar sobre millones de filas de precios_1h en cada visita.
        c.execute('''CREATE TABLE IF NOT EXISTS resumen_actual (
                    item_id INTEGER PRIMARY KEY,
                    name TEXT,
                    ultimo_timestamp INTEGER,
                    avg_high_price INTEGER,
                    avg_low_price INTEGER,
                    margen_bruto INTEGER,
                    impuesto_ge INTEGER,
                    margen_neto INTEGER,
                    roi_pct REAL,
                    profit_potencial_4h INTEGER,
                    pct_cambio_24h REAL,
                    pct_cambio_7d REAL,
                    volatilidad_30d_pct REAL,
                    percentil_30d REAL,
                    volumen_24h INTEGER,
                    tendencia_volumen REAL,
                    n_horas_historial INTEGER,
                    calculado_en INTEGER)''')
        c.execute(
            '''CREATE INDEX IF NOT EXISTS idx_resumen_margen_neto
                ON resumen_actual (margen_neto)'''
                )
        c.execute(
            '''CREATE INDEX IF NOT EXISTS idx_resumen_roi
                ON resumen_actual (roi_pct)'''
                )

        # Tabla predicciones. PRIMARY KEY (item_id, timestamp, model_version):
        # mismo patrón idempotente que las tablas de precios — reentrenar y
        # volver a guardar predicciones sobre el mismo set de test (ej. el
        # job diario reentrenando con datos que no cambiaron) no debe
        # duplicar filas, solo reemplazar el valor con INSERT OR REPLACE.
        c.execute('''CREATE TABLE IF NOT EXISTS predicciones
                    (
                    item_id INTEGER,
                    timestamp INTEGER,
                    predicted_price REAL,
                    actual_price REAL,
                    error REAL,
                    model_version TEXT,
                    PRIMARY KEY (item_id, timestamp, model_version))''')
        # Métricas del modelo (por ítem y fecha de entrenamiento)
        c.execute('''CREATE TABLE IF NOT EXISTS model_metrics
                    (id INTEGER PRIMARY KEY AUTOINCREMENT,
                    item_id INTEGER,
                    train_timestamp INTEGER,
                    model_name TEXT,
                    mae REAL,
                    rmse REAL)''')
        conn.commit()

        self._migrar_esquema(conn)
        conn.close()

    def _migrar_esquema(self, conn):
        """
        Agrega columnas/tablas nuevas a un esquema que ya pudo haber sido
        creado por una versión anterior de este archivo. Los CREATE TABLE
        IF NOT EXISTS de arriba no alcanzan para esto: no tocan una tabla
        que ya existe, aunque le falten columnas nuevas, y SQLite no
        soporta 'ALTER TABLE ... ADD COLUMN IF NOT EXISTS' — hay que
        chequear con PRAGMA table_info() antes de cada ALTER para que
        correr esto dos veces seguidas (cada vez que arranca el proceso)
        no falle.
        """
        c = conn.cursor()

        # model_metrics: horizonte_horas/accuracy_direccional/modo_evaluacion
        # (evaluacion.py, entrenador.py) — separan métricas de 1 paso de las
        # de horizontes 2..6, y el holdout clásico del walk-forward del
        # replay histórico (replay_historico.py), que no deben mezclarse en
        # el mismo gráfico del dashboard.
        columnas_model_metrics = {row[1] for row in c.execute('PRAGMA table_info(model_metrics)')}
        if 'horizonte_horas' not in columnas_model_metrics:
            c.execute('ALTER TABLE model_metrics ADD COLUMN horizonte_horas INTEGER DEFAULT 1')
        if 'accuracy_direccional' not in columnas_model_metrics:
            c.execute('ALTER TABLE model_metrics ADD COLUMN accuracy_direccional REAL')
        if 'modo_evaluacion' not in columnas_model_metrics:
            c.execute("ALTER TABLE model_metrics ADD COLUMN modo_evaluacion TEXT DEFAULT 'holdout'")

        # Único por (item_id o agregado, train_timestamp, model_name,
        # horizonte_horas, modo_evaluacion) — sin esto, correr job_horario/
        # job_diario dos veces sobre el mismo período (ej. el proceso se
        # reinicia y todavía no llegó dato nuevo, así que el dataset queda
        # igual) duplica silenciosamente la fila agregada Y cada fila por
        # ítem, en vez de reemplazarla (guardar_metricas_modelo pasa a usar
        # INSERT OR REPLACE apoyado en este índice). COALESCE(item_id, -1):
        # SQLite trata cada NULL como "distinto de cualquier otro NULL" en
        # un índice único — sin el COALESCE, las filas agregadas (item_id
        # IS NULL) nunca chocarían entre sí y seguirían duplicándose.
        indices_model_metrics = {row[1] for row in c.execute('PRAGMA index_list(model_metrics)')}
        if 'idx_model_metrics_unico' not in indices_model_metrics:
            # Deduplicar lo que ya exista antes de crear el índice único —
            # CREATE UNIQUE INDEX falla si hay duplicados. Se conserva la
            # fila más reciente (id más alto) de cada grupo.
            c.execute('''
                DELETE FROM model_metrics
                WHERE id NOT IN (
                    SELECT MAX(id) FROM model_metrics
                    GROUP BY COALESCE(item_id, -1), train_timestamp, model_name, horizonte_horas, modo_evaluacion
                )
            ''')
            c.execute('''
                CREATE UNIQUE INDEX idx_model_metrics_unico
                ON model_metrics (COALESCE(item_id, -1), train_timestamp, model_name, horizonte_horas, modo_evaluacion)
            ''')

        # predicciones: agregar modo_evaluacion a la PK. Sin esto, una
        # corrida de replay walk-forward (replay_historico.py) y el holdout
        # en vivo pueden predecir el mismo (item_id, timestamp,
        # model_version) — ej. el replay corriendo sobre las últimas horas,
        # que caen dentro del test set holdout en vivo — y el INSERT OR
        # REPLACE de una pisa silenciosamente las filas de la otra. SQLite
        # no permite ALTER TABLE para cambiar la PRIMARY KEY, así que hay
        # que recrear la tabla; las filas existentes (todas de corridas en
        # vivo, de antes de que existiera el modo walkforward) se migran
        # con modo_evaluacion='holdout'.
        columnas_predicciones = {row[1] for row in c.execute('PRAGMA table_info(predicciones)')}
        if 'modo_evaluacion' not in columnas_predicciones:
            c.execute('''CREATE TABLE predicciones_nueva (
                        item_id INTEGER,
                        timestamp INTEGER,
                        predicted_price REAL,
                        actual_price REAL,
                        error REAL,
                        model_version TEXT,
                        modo_evaluacion TEXT DEFAULT 'holdout',
                        PRIMARY KEY (item_id, timestamp, model_version, modo_evaluacion))''')
            c.execute('''INSERT INTO predicciones_nueva
                        (item_id, timestamp, predicted_price, actual_price, error, model_version, modo_evaluacion)
                        SELECT item_id, timestamp, predicted_price, actual_price, error, model_version, 'holdout'
                        FROM predicciones''')
            c.execute('DROP TABLE predicciones')
            c.execute('ALTER TABLE predicciones_nueva RENAME TO predicciones')

        # alertas_enviadas: cooldown de alertas.py — evita reavisar el mismo
        # ítem en cada corrida de job_horario si no cambió sustancialmente.
        c.execute('''CREATE TABLE IF NOT EXISTS alertas_enviadas (
                    item_id INTEGER PRIMARY KEY,
                    ultima_alerta INTEGER,
                    ultimo_roi_pct REAL,
                    ultimo_margen_neto INTEGER)''')

        # mantenimiento_estado: última corrida de cada job de mantenimiento
        # (ej. job_semanal), para poder hacer catch-up si el proceso estuvo
        # apagado justo en la ventana programada (ver recolector.py).
        c.execute('''CREATE TABLE IF NOT EXISTS mantenimiento_estado (
                    tarea TEXT PRIMARY KEY,
                    ultima_corrida INTEGER)''')

        # modelos_config: registro de modelos definidos por el usuario (app de
        # escritorio) — reemplaza los 3 modelos que hasta ahora vivían
        # hardcodeados en recolector.job_horario/job_diario. model_id es
        # literalmente el mismo string que entrenador.py usa como
        # model_name/model_version en model_metrics/predicciones/el nombre del
        # .pkl — no es un id aparte, así que un modelo de acá se referencia
        # igual en todo el resto del pipeline sin ninguna tabla puente.
        #
        # modo_seleccion distingue cómo se arma el universo de ítems:
        # 'manual' (item_ids: JSON con la lista elegida a mano por el usuario
        # en el selector de la app) o 'liquidez' (n_items/solo_f2p/
        # precio_minimo/excluir_item_ids: los mismos parámetros que ya
        # acepta obtener_top_items_liquidez[_hasta] — no expuesto desde la
        # app de escritorio hoy, ver escritorio/paginas/pagina_modelos.py,
        # pero sigue siendo un modo válido para crear un modelo a mano vía
        # crear_modelo_config).
        #
        # cadencia ('horaria'|'diaria'|'manual') decide en qué job del
        # recolector se reentrena solo (o nunca, en 'manual' — para que
        # explorar configuraciones nuevas no cargue el scheduler). estado
        # ('activo'|'pausado') permite desactivar un modelo sin borrar su
        # historial en model_metrics/predicciones ni su .pkl.
        c.execute('''CREATE TABLE IF NOT EXISTS modelos_config (
                    model_id TEXT PRIMARY KEY,
                    nombre TEXT,
                    tipo TEXT,
                    modo_seleccion TEXT,
                    item_ids TEXT,
                    n_items INTEGER,
                    solo_f2p INTEGER,
                    precio_minimo INTEGER,
                    excluir_item_ids TEXT,
                    lags INTEGER DEFAULT 5,
                    ma_windows TEXT DEFAULT '[3, 6]',
                    horizonte_horas INTEGER DEFAULT 1,
                    umbral_pct REAL,
                    cadencia TEXT,
                    estado TEXT DEFAULT 'activo',
                    creado_en INTEGER,
                    ultimo_entrenamiento_ts INTEGER,
                    tabla TEXT DEFAULT 'precios_1h',
                    ventana_dias REAL)''')

        # tabla/ventana_dias: agregados después de la primera versión de
        # modelos_config (ver el bullet de más abajo, mismo patrón que el
        # resto de _migrar_esquema) -- tabla es la granularidad de precios
        # sobre la que entrena el modelo (precios_5m/1h/6h, antes siempre
        # implícitamente precios_1h); ventana_dias acota cuánto historial
        # hacia atrás usa cada entrenamiento (None = todo el disponible,
        # el comportamiento de siempre). CREATE TABLE IF NOT EXISTS no
        # alcanza para una DB que ya tenía la tabla sin estas columnas.
        columnas_modelos_config = {row[1] for row in c.execute('PRAGMA table_info(modelos_config)')}
        if 'tabla' not in columnas_modelos_config:
            c.execute("ALTER TABLE modelos_config ADD COLUMN tabla TEXT DEFAULT 'precios_1h'")
        if 'ventana_dias' not in columnas_modelos_config:
            c.execute('ALTER TABLE modelos_config ADD COLUMN ventana_dias REAL')

        conn.commit()

        # No se siembra ningún modelo por default acá (a propósito, pedido
        # explícito del usuario: "no quiero que haya ningún modelo por
        # default. todos tienen que poder eliminarse"). Versiones previas
        # de este método sembraban 3 modelos productivos hardcodeados
        # (global_horario/global_diario/f2p10_100gp_clasif) que
        # job_horario/job_diario reentrenaban automáticamente y que la app
        # de escritorio no dejaba eliminar (ver el historial de
        # escritorio/paginas/pagina_modelos.py) — modelos_config arranca
        # vacía ahora, y job_horario/job_diario (recolector.py) simplemente
        # no reentrenan nada hasta que el usuario cree uno desde "Mis
        # modelos" en la app.

    @staticmethod
    def _fila_a_modelo_config(fila):
        """Convierte una fila cruda de modelos_config (tupla, orden de
        columnas de la tabla) en un dict con item_ids/excluir_item_ids/
        ma_windows ya deserializados de JSON — así el resto del código
        (recolector.py, la UI) no repite json.loads en cada lugar que lee
        un modelo."""
        import json

        (
            model_id, nombre, tipo, modo_seleccion, item_ids, n_items, solo_f2p,
            precio_minimo, excluir_item_ids, lags, ma_windows, horizonte_horas,
            umbral_pct, cadencia, estado, creado_en, ultimo_entrenamiento_ts,
            tabla, ventana_dias,
        ) = fila
        return {
            'model_id': model_id,
            'nombre': nombre,
            'tipo': tipo,
            'modo_seleccion': modo_seleccion,
            'item_ids': json.loads(item_ids) if item_ids else None,
            'n_items': n_items,
            'solo_f2p': bool(solo_f2p),
            'precio_minimo': precio_minimo,
            'excluir_item_ids': json.loads(excluir_item_ids) if excluir_item_ids else None,
            'lags': lags,
            'ma_windows': json.loads(ma_windows) if ma_windows else [3, 6],
            'horizonte_horas': horizonte_horas,
            'umbral_pct': umbral_pct,
            'cadencia': cadencia,
            'estado': estado,
            'creado_en': creado_en,
            'ultimo_entrenamiento_ts': ultimo_entrenamiento_ts,
            'tabla': tabla or 'precios_1h',
            'ventana_dias': ventana_dias,
        }

    def crear_modelo_config(
        self, model_id, nombre, tipo, cadencia, modo_seleccion='manual',
        item_ids=None, n_items=None, solo_f2p=False, precio_minimo=None,
        excluir_item_ids=None, lags=5, ma_windows=None, horizonte_horas=1,
        umbral_pct=None, tabla='precios_1h', ventana_dias=None,
    ):
        """
        Da de alta un modelo definido por el usuario. model_id es elegido
        por el llamador (ver escritorio/paginas/pagina_modelos.py: se arma
        a partir del nombre) porque es el mismo string que después se usa
        como model_name/model_version en todo entrenador.py/prediccion.py —
        no hay un id autoincremental separado. Lanza sqlite3.IntegrityError
        si el model_id ya existe (el llamador debe elegir uno libre antes
        de llamar, ver obtener_modelo_config).

        modo_seleccion='manual' (el caso nuevo, ítems elegidos a mano en la
        UI): item_ids es la lista completa, n_items/solo_f2p/precio_minimo/
        excluir_item_ids quedan en None/False, no se usan.
        modo_seleccion='liquidez' (mismo comportamiento que los modelos
        productivos preexistentes): item_ids queda en None, se arma en
        entrenamiento vía obtener_top_items_liquidez[_hasta] con estos
        parámetros.

        tabla ('precios_5m'|'precios_1h'|'precios_6h', default 'precios_1h'
        — el comportamiento de siempre): granularidad de precios sobre la
        que entrena el modelo. ventana_dias (opcional, None = todo el
        historial disponible, el comportamiento de siempre): acota cada
        entrenamiento a los últimos `ventana_dias` días relativos al
        momento de entrenar — ver entrenador.entrenar_modelo_global/
        entrenar_clasificador_direccional. Ninguno de los dos se valida acá
        (los valores razonables dependen de qué tan corta puede ser una
        ventana antes de quedarse sin datos suficientes para lags/medias
        móviles — eso ya lo maneja entrenador.py devolviendo False/None si
        no hay suficiente).

        Lanza ValueError si modo_seleccion='manual' pide más de
        MAX_ITEMS_POR_MODELO ítems, o si cadencia != 'manual' y ya hay
        MAX_MODELOS_ACTIVOS modelos activos con cadencia horaria/diaria
        (ver esas constantes arriba) — el modelo NO se crea en ese caso.
        Un modelo con cadencia='manual' no cuenta contra ese tope (nunca
        se reentrena solo, no carga el scheduler).
        """
        import json

        if modo_seleccion == 'manual' and item_ids and len(item_ids) > MAX_ITEMS_POR_MODELO:
            raise ValueError(
                f"Máximo {MAX_ITEMS_POR_MODELO} ítems por modelo (se pidieron {len(item_ids)})."
            )
        if cadencia != 'manual' and self.contar_modelos_config_activos() >= MAX_MODELOS_ACTIVOS:
            raise ValueError(
                f"Ya hay {MAX_MODELOS_ACTIVOS} modelos activos con cadencia horaria/diaria — "
                "pausá o eliminá alguno antes de activar uno nuevo (o creá este con cadencia 'manual')."
            )

        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        ahora = int(time.time())
        c.execute(
            '''INSERT INTO modelos_config (
                model_id, nombre, tipo, modo_seleccion, item_ids, n_items, solo_f2p,
                precio_minimo, excluir_item_ids, lags, ma_windows, horizonte_horas,
                umbral_pct, cadencia, estado, creado_en, ultimo_entrenamiento_ts,
                tabla, ventana_dias
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,'activo',?,NULL,?,?)''',
            (
                model_id, nombre, tipo, modo_seleccion,
                json.dumps(item_ids) if item_ids is not None else None,
                n_items, int(bool(solo_f2p)), precio_minimo,
                json.dumps(excluir_item_ids) if excluir_item_ids else None,
                lags, json.dumps(ma_windows or [3, 6]), horizonte_horas, umbral_pct,
                cadencia, ahora, tabla, ventana_dias,
            ),
        )
        conn.commit()
        conn.close()
        return model_id

    def listar_modelos_config(self, cadencia=None, estado=None):
        """Lista modelos_config como dicts (ver _fila_a_modelo_config),
        opcionalmente filtrados por cadencia y/o estado — usado por
        recolector.job_horario/job_diario (cadencia='horaria'/'diaria',
        estado='activo') para saber qué reentrenar en cada corrida, y por
        la UI para mostrar el listado completo (sin filtro)."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        query = 'SELECT * FROM modelos_config'
        condiciones, params = [], []
        if cadencia is not None:
            condiciones.append('cadencia = ?')
            params.append(cadencia)
        if estado is not None:
            condiciones.append('estado = ?')
            params.append(estado)
        if condiciones:
            query += ' WHERE ' + ' AND '.join(condiciones)
        query += ' ORDER BY creado_en'
        c.execute(query, params)
        filas = c.fetchall()
        conn.close()
        return [self._fila_a_modelo_config(f) for f in filas]

    def obtener_modelo_config(self, model_id):
        """Un modelo de modelos_config como dict, o None si no existe."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('SELECT * FROM modelos_config WHERE model_id = ?', (model_id,))
        fila = c.fetchone()
        conn.close()
        return self._fila_a_modelo_config(fila) if fila else None

    def contar_modelos_config_activos(self):
        """Cantidad de modelos en estado='activo' (cadencia horaria o
        diaria) — usado para aplicar el tope de modelos simultáneos antes
        de activar uno nuevo (ver escritorio/paginas/pagina_modelos.py),
        sin contar los de cadencia='manual' (no cargan el scheduler)."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute(
            "SELECT COUNT(*) FROM modelos_config WHERE estado = 'activo' AND cadencia != 'manual'"
        )
        n = c.fetchone()[0]
        conn.close()
        return n

    def actualizar_modelo_config(self, model_id, **campos):
        """
        Actualiza columnas arbitrarias de un modelo existente (ej.
        estado='pausado', o ultimo_entrenamiento_ts=... después de
        reentrenar). `campos` son nombres de columna de modelos_config tal
        cual — item_ids/excluir_item_ids/ma_windows se serializan a JSON
        automáticamente si vienen como list. No valida que el model_id
        exista: un UPDATE sobre un id inexistente simplemente no toca
        ninguna fila (rowcount 0).

        Si `campos` reactiva un modelo (estado='activo') aplica el mismo
        tope de MAX_MODELOS_ACTIVOS que crear_modelo_config — sin este
        chequeo, pausar y reactivar modelos sería una forma de eludir el
        tope. No se valida al pausar ni al tocar otros campos.
        """
        import json

        if not campos:
            return 0
        if campos.get('estado') == 'activo':
            modelo_actual = self.obtener_modelo_config(model_id)
            cadencia = campos.get('cadencia', modelo_actual['cadencia'] if modelo_actual else 'manual')
            ya_activo = modelo_actual is not None and modelo_actual['estado'] == 'activo'
            if cadencia != 'manual' and not ya_activo and self.contar_modelos_config_activos() >= MAX_MODELOS_ACTIVOS:
                raise ValueError(
                    f"Ya hay {MAX_MODELOS_ACTIVOS} modelos activos con cadencia horaria/diaria — "
                    "pausá o eliminá alguno antes de reactivar este."
                )
        columnas_json = {'item_ids', 'excluir_item_ids', 'ma_windows'}
        valores = []
        sets = []
        for columna, valor in campos.items():
            if columna in columnas_json and isinstance(valor, list):
                valor = json.dumps(valor)
            sets.append(f'{columna} = ?')
            valores.append(valor)
        valores.append(model_id)

        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute(f'UPDATE modelos_config SET {", ".join(sets)} WHERE model_id = ?', valores)
        filas_afectadas = c.rowcount
        conn.commit()
        conn.close()
        return filas_afectadas

    def eliminar_modelo_config(self, model_id):
        """
        Borra un modelo de modelos_config. Deliberadamente NO borra su
        historial en model_metrics/predicciones ni su .pkl en models/ — se
        conserva como registro histórico (mismo criterio que
        mantenimiento.podar_metricas_por_item, que también conserva el
        agregado); si el llamador quiere limpiar eso también, es una acción
        aparte y explícita del lado de la UI (escritorio/paginas/
        pagina_modelos.py), no un efecto secundario silencioso de borrar la
        configuración.
        """
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('DELETE FROM modelos_config WHERE model_id = ?', (model_id,))
        filas_afectadas = c.rowcount
        conn.commit()
        conn.close()
        return filas_afectadas

    def insertar_precios(self, table, data):

        """
        Inserta múltiples registros en la tabla de precios (prices_5min o prices_1h).
        data: lista de tuplas (item_id, timestamp, avg_high_price, avg_low_price, high_volume, low_volume)

        executemany() en vez de un execute() por fila en un loop de Python:
        con tablas de millones de filas (backfills largos), el loop fila-a-fila
        se vuelve el cuello de botella real de la recolección — cada snapshot
        trae ~2000-2500 filas, y el overhead de Python por execute() individual
        domina sobre el costo real del INSERT en SQLite. Mismo resultado
        (INSERT OR IGNORE, una sola transacción), mucho más rápido.
        """
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        filas = [
            (
                row.item_id,
                row.timestamp,
                row.avg_high_price,
                row.avg_low_price,
                row.high_volume,
                row.low_volume,
            )
            for row in data.itertuples()
        ]

        c.executemany(
            f'''INSERT OR IGNORE INTO {table} (
            item_id,
            timestamp,
            avg_high_price,
            avg_low_price,
            high_volume,
            low_volume
            )
            VALUES (?,?,?,?,?,?)''', filas)

        conn.commit()
        conn.close()

    def guardar_items(self, items_mapping):
        """
        Guarda/actualiza el catálogo de ítems.
        items_mapping: lista de dicts tal como los entrega
        OSRSGeAPI.get_item_mapping() (claves: id, name, members, limit).
        """
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        filas = [
            (
                item['id'],
                item.get('name'),
                int(bool(item.get('members'))),
                item.get('limit'),
            )
            for item in items_mapping
        ]

        c.executemany(
            '''INSERT INTO items (item_id, name, members, buy_limit)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(item_id) DO UPDATE SET
                    name=excluded.name,
                    members=excluded.members,
                    buy_limit=excluded.buy_limit''',
            filas,
        )

        conn.commit()
        conn.close()
        return len(filas)

    def guardar_resumen(self, filas):
        """
        Reemplaza el contenido de resumen_actual con las filas dadas.
        filas: lista de dicts con las llaves calculadas por
        metricas.calcular_resumen_todos(). Es un snapshot del estado actual,
        no histórico, así que se limpia y se vuelve a llenar completo en
        cada corrida (no tiene sentido acumular resúmenes viejos acá).
        """
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        calculado_en = int(time.time())
        tuplas = [
            (
                f['item_id'], f['name'], f['ultimo_timestamp'],
                f['avg_high_price'], f['avg_low_price'],
                f['margen_bruto'], f['impuesto_ge'], f['margen_neto'], f['roi_pct'],
                f['profit_potencial_4h'], f['pct_cambio_24h'], f['pct_cambio_7d'],
                f['volatilidad_30d_pct'], f['percentil_30d'], f['volumen_24h'],
                f['tendencia_volumen'], f['n_horas_historial'], calculado_en,
            )
            for f in filas
        ]

        c.execute('DELETE FROM resumen_actual')
        c.executemany(
            '''INSERT INTO resumen_actual (
                item_id, name, ultimo_timestamp, avg_high_price, avg_low_price,
                margen_bruto, impuesto_ge, margen_neto, roi_pct,
                profit_potencial_4h, pct_cambio_24h, pct_cambio_7d,
                volatilidad_30d_pct, percentil_30d, volumen_24h,
                tendencia_volumen, n_horas_historial, calculado_en
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
            tuplas,
        )

        conn.commit()
        conn.close()
        return len(tuplas)

    def obtener_precios_id(self, item_id, table, desde_timestamp=None, hasta_timestamp=None):
        """
        Obtiene datos de precios para un ítem.
        desde_timestamp/hasta_timestamp (opcionales): si se pasan, filtran a
        desde_timestamp <= timestamp <= hasta_timestamp en la query SQL en
        vez de traer todo el historial y recortar después en pandas —
        importante en tablas grandes para no cargar meses de datos que
        después se descartan. hasta_timestamp es lo que permite simular
        "qué se sabía hasta este momento" durante el replay histórico
        (replay_historico.py): sin él, entrenar con `hasta_timestamp=X`
        igual vería datos posteriores a X ya backfilleados en la tabla.
        Devuelve DataFrame con columnas: timestamp, avg_high_price, avg_low_price, high_volume, low_volume
        """
        conn = sqlite3.connect(self.db_path)
        query = f"SELECT item_id, timestamp, avg_high_price, avg_low_price, high_volume, low_volume FROM {table} WHERE item_id = ? "

        params = [item_id]
        if desde_timestamp is not None:
            query += "AND timestamp >= ? "
            params.append(desde_timestamp)
        if hasta_timestamp is not None:
            query += "AND timestamp <= ? "
            params.append(hasta_timestamp)

        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        return df

    def obtener_precios_multi(self, item_ids, table, desde_timestamp=None, hasta_timestamp=None):
        """
        Como obtener_precios_id, pero para varios ítems en UNA sola query
        (WHERE item_id IN (...)) en vez de una conexión+query por ítem.
        Usada por preprocesamiento.build_training_set(), que arma el
        dataset de ~200 ítems para entrenar: abrir 200 conexiones SQLite
        separadas (una por obtener_precios_id) es el cuello de botella real
        de entrenar_modelo_global — sobre todo en el replay histórico
        (replay_historico.py), donde eso se repite en cada checkpoint.
        Devuelve un único DataFrame con columnas item_id, timestamp,
        avg_high_price, avg_low_price, high_volume, low_volume — separar
        por ítem con .groupby('item_id') del lado del llamador.
        """
        if not item_ids:
            return pd.DataFrame(columns=['item_id', 'timestamp', 'avg_high_price', 'avg_low_price', 'high_volume', 'low_volume'])

        conn = sqlite3.connect(self.db_path)
        placeholders = ','.join('?' * len(item_ids))
        query = (
            f"SELECT item_id, timestamp, avg_high_price, avg_low_price, high_volume, low_volume "
            f"FROM {table} WHERE item_id IN ({placeholders}) "
        )
        params = [int(i) for i in item_ids]
        if desde_timestamp is not None:
            query += "AND timestamp >= ? "
            params.append(desde_timestamp)
        if hasta_timestamp is not None:
            query += "AND timestamp <= ? "
            params.append(hasta_timestamp)

        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        return df

    def obtener_resumen_datos(self):
        """
        Cuánto historial hay disponible AHORA MISMO en cada tabla de
        precios — pensado para mostrarse en la app de escritorio
        (escritorio/paginas/pagina_inicio.py) sin tener que abrir la DB a
        mano. Incluye precios_1h_diario (el agregado que
        mantenimiento.archivar_datos_antiguos produce al purgar
        precios_1h) además de las 3 tablas crudas — sin esa cuarta fila,
        la cantidad real de historial acumulado (incluido lo ya
        archivado) quedaría subestimada.

        Devuelve {tabla: {filas, desde_ts, hasta_ts, dias_historial}}.
        desde_ts/hasta_ts/dias_historial quedan en None si la tabla está
        vacía. COUNT/MIN/MAX son consultas rápidas acá (columna de tiempo
        indexada en las 4 tablas), incluso sobre una DB de cientos de MB.
        `precios_1h_diario` usa `fecha` en vez de `timestamp` como su
        columna de tiempo (ver el esquema en __init__).
        """
        columna_tiempo = {
            'precios_5m': 'timestamp',
            'precios_1h': 'timestamp',
            'precios_6h': 'timestamp',
            'precios_1h_diario': 'fecha',
        }
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        resumen = {}
        for tabla, col in columna_tiempo.items():
            c.execute(f'SELECT COUNT(*), MIN({col}), MAX({col}) FROM {tabla}')
            filas, desde_ts, hasta_ts = c.fetchone()
            dias_historial = (hasta_ts - desde_ts) / 86400 if desde_ts is not None and hasta_ts is not None else None
            resumen[tabla] = {
                'filas': filas,
                'desde_ts': desde_ts,
                'hasta_ts': hasta_ts,
                'dias_historial': dias_historial,
            }
        conn.close()
        return resumen

    def obtener_top_items_liquidez(self, n=200, solo_f2p=False, precio_minimo=None, excluir_item_ids=None):
        """
        Devuelve los `n` item_id más líquidos según `volumen_24h` en
        resumen_actual (poblada por metricas.py). Pensado para elegir el
        subconjunto de ítems con el que entrenar un modelo global: los
        ítems de bajo volumen tienen poco historial útil y agregan ruido.

        solo_f2p: si True, restringe a ítems free-to-play (items.members=0)
        — para acotar el universo a un conjunto mucho más chico y
        específico (ej. entrenar_clasificador_direccional con n=10 sobre
        solo runas/materiales F2P clásicos, en vez de los 200 ítems más
        líquidos del juego completo, mezclando members y F2P).

        precio_minimo: si no es None, descarta ítems con avg_low_price por
        debajo de ese piso — ítems muy baratos (ej. runas de 2-5 gp) tienen
        margen neto por unidad tan chico que el impuesto GE (2%, redondeado
        hacia abajo) se lo come en la mitad de los trades, aunque el modelo
        acierte la dirección (validado en backtest: con ítems así, seguir
        la señal del clasificador no le gana a comprar a ciegas — ver
        backtest.simular_clasificador_walkforward).

        excluir_item_ids: lista opcional de item_id a descartar del
        universo aunque califiquen por liquidez/precio — ej. Steel bar
        (2353), que en el backtest walk-forward de 90 días perdió plata
        tanto con la señal del modelo como comprando a ciegas.
        """
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        if solo_f2p:
            prefix = 'r.'
            condiciones = ['i.members = 0']
            select_from = '''SELECT r.item_id FROM resumen_actual r
                              JOIN items i ON i.item_id = r.item_id'''
        else:
            prefix = ''
            condiciones = []
            select_from = 'SELECT item_id FROM resumen_actual'
        params = []
        if precio_minimo is not None:
            condiciones.append(f'{prefix}avg_low_price >= ?')
            params.append(precio_minimo)
        if excluir_item_ids:
            condiciones.append(f"{prefix}item_id NOT IN ({','.join('?' * len(excluir_item_ids))})")
            params.extend(excluir_item_ids)
        where = f' WHERE {" AND ".join(condiciones)}' if condiciones else ''
        query = f'{select_from}{where} ORDER BY {prefix}volumen_24h DESC LIMIT ?'
        params.append(n)
        c.execute(query, params)
        item_ids = [row[0] for row in c.fetchall()]
        conn.close()
        return item_ids

    def guardar_alerta(self, item_id, ts, roi_pct, margen_neto):
        """
        Registra/actualiza la última alerta enviada para un ítem (tabla
        alertas_enviadas) — usado por alertas.evaluar_alertas() para el
        cooldown: no reavisar el mismo ítem en cada corrida de job_horario
        si no cambió sustancialmente desde la última vez.
        """
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute(
            '''INSERT INTO alertas_enviadas (item_id, ultima_alerta, ultimo_roi_pct, ultimo_margen_neto)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(item_id) DO UPDATE SET
                    ultima_alerta=excluded.ultima_alerta,
                    ultimo_roi_pct=excluded.ultimo_roi_pct,
                    ultimo_margen_neto=excluded.ultimo_margen_neto''',
            (item_id, ts, roi_pct, margen_neto),
        )
        conn.commit()
        conn.close()

    def obtener_ultima_alerta(self, item_id):
        """(ultima_alerta, ultimo_roi_pct, ultimo_margen_neto) o None si
        nunca se alertó este ítem todavía."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute(
            '''SELECT ultima_alerta, ultimo_roi_pct, ultimo_margen_neto
               FROM alertas_enviadas WHERE item_id = ?''',
            (item_id,),
        )
        fila = c.fetchone()
        conn.close()
        return fila

    def registrar_corrida(self, tarea, ts):
        """Registra el timestamp de la última corrida exitosa de `tarea`
        (tabla mantenimiento_estado) — usado por recolector.verificar_catchup_semanal
        para hacer catch-up si job_semanal no corrió en su ventana programada."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute(
            '''INSERT INTO mantenimiento_estado (tarea, ultima_corrida) VALUES (?, ?)
               ON CONFLICT(tarea) DO UPDATE SET ultima_corrida=excluded.ultima_corrida''',
            (tarea, ts),
        )
        conn.commit()
        conn.close()

    def obtener_ultima_corrida(self, tarea):
        """Timestamp de la última corrida registrada de `tarea`, o None si
        nunca corrió (o corrió antes de que existiera este registro)."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('SELECT ultima_corrida FROM mantenimiento_estado WHERE tarea = ?', (tarea,))
        fila = c.fetchone()
        conn.close()
        return fila[0] if fila else None

    def obtener_top_items_liquidez_hasta(self, ahora_ts, n=200, tabla='precios_1h', ventana_horas=24, solo_f2p=False, precio_minimo=None, excluir_item_ids=None):
        """
        Igual que obtener_top_items_liquidez, pero calculado directamente
        por SQL sobre `tabla` (SUM(high_volume+low_volume) agrupado por
        item_id, en la ventana [ahora_ts - ventana_horas, ahora_ts]) en vez
        de leer resumen_actual. Necesaria para el replay histórico
        (replay_historico.py): resumen_actual es el estado EN VIVO que lee
        el dashboard, y escribirle un resumen calculado en un momento
        histórico intermedio lo dejaría mostrando datos viejos/incorrectos.

        solo_f2p: ver obtener_top_items_liquidez — mismo filtro, para que
        el replay walk-forward de un modelo acotado a F2P use el mismo
        universo de ítems en cada checkpoint.

        precio_minimo: ver obtener_top_items_liquidez — acá se compara
        contra el promedio de avg_low_price DENTRO de la misma ventana de
        volumen (no hay un "precio actual" en un punto histórico), vía
        HAVING sobre el AVG agrupado.

        excluir_item_ids: ver obtener_top_items_liquidez.
        """
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        desde = ahora_ts - ventana_horas * 3600
        prefix = 'p.' if solo_f2p else ''

        condiciones = [f'{prefix}timestamp BETWEEN ? AND ?']
        params = [desde, ahora_ts]
        if solo_f2p:
            condiciones.append('i.members = 0')
        if excluir_item_ids:
            condiciones.append(f"{prefix}item_id NOT IN ({','.join('?' * len(excluir_item_ids))})")
            params.extend(excluir_item_ids)
        where = ' AND '.join(condiciones)

        having = f' HAVING AVG({prefix}avg_low_price) >= ?' if precio_minimo is not None else ''
        having_params = [precio_minimo] if precio_minimo is not None else []

        if solo_f2p:
            from_clause = f'{tabla} p JOIN items i ON i.item_id = p.item_id'
        else:
            from_clause = tabla

        query = f'''SELECT {prefix}item_id, SUM({prefix}high_volume + {prefix}low_volume) AS vol
                    FROM {from_clause}
                    WHERE {where}
                    GROUP BY {prefix}item_id{having}
                    ORDER BY vol DESC
                    LIMIT ?'''
        c.execute(query, [*params, *having_params, n])
        item_ids = [row[0] for row in c.fetchall()]
        conn.close()
        return item_ids

    def guardar_metricas_modelo(self, filas):
        """
        Guarda métricas de una corrida de entrenamiento en model_metrics.
        filas: lista de tuplas (item_id, train_timestamp, model_name, mae,
        rmse, horizonte_horas, accuracy_direccional, modo_evaluacion).
        item_id puede ser None para representar una métrica agregada (todo
        el modelo global), no una fila por ítem. horizonte_horas=1 es la
        métrica clásica de 1 paso (entrenador.py); horizontes mayores vienen
        de evaluacion.evaluar_horizontes(). modo_evaluacion distingue el
        holdout 80/20 en vivo del walk-forward del replay histórico
        (replay_historico.py) — no deben mezclarse en el mismo gráfico.

        INSERT OR REPLACE, apoyado en idx_model_metrics_unico (ver
        _migrar_esquema): reentrenar sobre el mismo período (mismo
        train_timestamp — ej. job_horario corriendo dos veces seguidas sin
        que haya llegado dato nuevo todavía) reemplaza la fila en vez de
        duplicarla.
        """
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.executemany(
            '''INSERT OR REPLACE INTO model_metrics
                (item_id, train_timestamp, model_name, mae, rmse,
                 horizonte_horas, accuracy_direccional, modo_evaluacion)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)''', filas
        )
        conn.commit()
        conn.close()
        return len(filas)

    def existe_checkpoint(self, model_name, train_timestamp, modo_evaluacion):
        """
        True si ya existe una fila agregada (item_id IS NULL) en
        model_metrics para este (model_name, train_timestamp,
        modo_evaluacion) — permite que replay_historico.py sea reanudable
        sin duplicar filas: guardar_metricas_modelo() no tiene una
        restricción de unicidad a nivel de esquema (ya tenía filas de
        corridas anteriores antes de este cambio, así que migrar a un
        índice UNIQUE requeriría deduplicar primero), así que el chequeo se
        hace acá, antes de decidir si reentrenar un checkpoint.
        """
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute(
            '''SELECT 1 FROM model_metrics
               WHERE item_id IS NULL AND model_name = ? AND train_timestamp = ?
                 AND modo_evaluacion = ? LIMIT 1''',
            (model_name, train_timestamp, modo_evaluacion),
        )
        existe = c.fetchone() is not None
        conn.close()
        return existe

    def guardar_predicciones(self, filas):
        """
        Guarda predicciones (típicamente del set de test) en predicciones.
        filas: lista de tuplas (item_id, timestamp, predicted_price,
        actual_price, error, model_version, modo_evaluacion).
        INSERT OR REPLACE: reentrenar sobre el mismo (item_id, timestamp,
        model_version, modo_evaluacion) actualiza la fila en vez de
        duplicarla. modo_evaluacion es parte de la PK (no solo de
        model_metrics) para que el holdout en vivo y el walk-forward del
        replay histórico (replay_historico.py) no se pisen entre sí cuando
        predicen el mismo (item_id, timestamp, model_version) — pasa
        seguido, porque el replay suele cubrir justo el tramo más reciente,
        que también cae dentro del test set holdout en vivo.
        """
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.executemany(
            '''INSERT OR REPLACE INTO predicciones
                (item_id, timestamp, predicted_price, actual_price, error, model_version, modo_evaluacion)
               VALUES (?, ?, ?, ?, ?, ?, ?)''', filas
        )
        conn.commit()
        conn.close()
        return len(filas)
