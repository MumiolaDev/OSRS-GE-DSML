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
        