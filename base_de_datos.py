import sqlite3
import pandas as pd
import time

DB_PATH = "osrs_data_test.db"

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
        conn.close()


    def insertar_precios(self, table, data):

        """
        Inserta múltiples registros en la tabla de precios (prices_5min o prices_1h).
        data: lista de tuplas (item_id, timestamp, avg_high_price, avg_low_price, high_volume, low_volume)
        """
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        for row in data.itertuples():

        
            filas = (
                row.item_id,
                row.timestamp, 
                row.avg_high_price, 
                row.avg_low_price, 
                row.high_volume, 
                row.low_volume
                )

            c.execute(
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

    def obtener_precios_id(self, item_id, table, desde_timestamp=None):
        """
        Obtiene datos de precios para un ítem.
        desde_timestamp (opcional): si se pasa, filtra a timestamp >= desde_timestamp
        en la query SQL en vez de traer todo el historial y recortar después en
        pandas — importante en tablas grandes para no cargar meses de datos que
        después se descartan.
        Devuelve DataFrame con columnas: timestamp, avg_high_price, avg_low_price, high_volume, low_volume
        """
        conn = sqlite3.connect(self.db_path)
        query = f"SELECT item_id, timestamp, avg_high_price, avg_low_price, high_volume, low_volume FROM {table} WHERE item_id = ? "

        params = [item_id]
        if desde_timestamp is not None:
            query += "AND timestamp >= ? "
            params.append(desde_timestamp)

        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        return df

    def obtener_top_items_liquidez(self, n=200):
        """
        Devuelve los `n` item_id más líquidos según `volumen_24h` en
        resumen_actual (poblada por metricas.py). Pensado para elegir el
        subconjunto de ítems con el que entrenar un modelo global: los
        ítems de bajo volumen tienen poco historial útil y agregan ruido.
        """
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute(
            '''SELECT item_id FROM resumen_actual
               ORDER BY volumen_24h DESC LIMIT ?''', (n,)
        )
        item_ids = [row[0] for row in c.fetchall()]
        conn.close()
        return item_ids

    def guardar_metricas_modelo(self, filas):
        """
        Guarda métricas de una corrida de entrenamiento en model_metrics.
        filas: lista de tuplas (item_id, train_timestamp, model_name, mae, rmse).
        item_id puede ser None para representar una métrica agregada (todo
        el modelo global), no una fila por ítem.
        """
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.executemany(
            '''INSERT INTO model_metrics
                (item_id, train_timestamp, model_name, mae, rmse)
               VALUES (?, ?, ?, ?, ?)''', filas
        )
        conn.commit()
        conn.close()
        return len(filas)

    def guardar_predicciones(self, filas):
        """
        Guarda predicciones (típicamente del set de test) en predicciones.
        filas: lista de tuplas
        (item_id, timestamp, predicted_price, actual_price, error, model_version).
        INSERT OR REPLACE: reentrenar sobre el mismo (item_id, timestamp,
        model_version) actualiza la fila en vez de duplicarla.
        """
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.executemany(
            '''INSERT OR REPLACE INTO predicciones
                (item_id, timestamp, predicted_price, actual_price, error, model_version)
               VALUES (?, ?, ?, ?, ?, ?)''', filas
        )
        conn.commit()
        conn.close()
        return len(filas)
