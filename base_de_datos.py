import sqlite3
import pandas as pd
import time

DB_PATH = "osrs_data_test.db"

class OSRSBaseDatos:

    def __init__(self, db_path = DB_PATH ):
        
        self.db_path = db_path

        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()


        # Crea las tablas si no existen.
        # Tabla items
        c.execute('''CREATE TABLE IF NOT EXISTS items
                    (item_id INTEGER PRIMARY KEY,
                    name TEXT,
                    members INTEGER,
                    buy_limit INTEGER)''')
        # Tabla precios_5m
        c.execute(
            '''CREATE TABLE IF NOT EXISTS precios_5m (
                item_id INT,
                timestamp INT,
                avg_high_price INTEGER,
                avg_low_price INTEGER,
                high_volume INTEGER,
                low_volume INTEGER
                )'''
                )
        # Tabla precios_1h
        c.execute(
            '''CREATE TABLE IF NOT EXISTS precios_1h (
                item_id INT,
                timestamp INT,
                avg_high_price INTEGER,
                avg_low_price INTEGER,
                high_volume INTEGER,
                low_volume INTEGER
                )'''
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
                f'''INSERT INTO {table} (
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
        