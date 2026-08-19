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

        # Tabla predicciones
        c.execute('''CREATE TABLE IF NOT EXISTS predicciones
                    (
                    item_id INTEGER,
                    timestamp INTEGER,
                    predicted_price REAL,
                    actual_price REAL,
                    error REAL,
                    model_version TEXT)''')
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

    def obtener_precios_id(self, item_id, table):
        """
        Obtiene datos de precios para un ítem.
        Devuelve DataFrame con columnas: timestamp, avg_high_price, avg_low_price, high_volume, low_volume
        """
        conn = sqlite3.connect(self.db_path)
        query = f"SELECT item_id, timestamp, avg_high_price, avg_low_price, high_volume, low_volume FROM {table} WHERE item_id = ? "
        
        params = [item_id]
    
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        return df
        