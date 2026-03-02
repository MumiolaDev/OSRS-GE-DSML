import schedule
import time
import logging
from datetime import datetime
from osrs_ge_api import OSRSGeAPI 
from base_de_datos import OSRSBaseDatos


# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s |--| %(levelname)s |--| %(message)s')

# Lista de ítems a monitorear (puedes ampliarla)
ITEM_IDS = [377, 440, 563, 564, 561]
api = OSRSGeAPI()

def collect(table, func, interval_name ,db, timestamp = None):
    """Función genérica para recolectar datos."""
 
    try:
        print(f"Recolectando datos {interval_name}")
        df = func(timestamp=timestamp)  # obtener prices ( id, intervalo)
 
       
        if df.empty:
            logging.warning(f"No se obtuvieron datos para {interval_name}")
            return
     
       
        OSRSBaseDatos.insertar_precios(db, table, df)

        logging.info(f"Insertados {len(df)} registros en {table}")

        return df

    except Exception as e:
        logging.error(f"Error en collect_{interval_name}: {e}")

def collect_5min(db, timestamp=None):
    return collect('precios_5m', api.get_historical_5min, '5m', db, timestamp=timestamp)

def collect_1h(db, timestamp=None):
    return collect('precios_1h', api.get_historical_1h, '1h',db, timestamp=timestamp)

if __name__ == "__main__":
    
    
    db = OSRSBaseDatos('data//testesito.db')
    logging.info("Iniciando recolector...")
    # Ejecutar inmediatamente al arrancar
    data_5m = collect_5min(db)
    data_1h = collect_1h(db)
    # Programar tareas
    schedule.every(5).minutes.do(collect_5min, db)
    schedule.every(1).hour.do(collect_1h, db)

    while True:
        
        schedule.run_pending()
        time.sleep(1)

        