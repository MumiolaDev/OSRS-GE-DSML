import sqlite3
import pandas as pd
import time

DB_PATH = "data/osrs_ge.db"

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

        conn.commit()


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
