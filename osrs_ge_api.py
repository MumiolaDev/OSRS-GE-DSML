import requests
import pandas as pd
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Timeout de cada request, en segundos. NUNCA sacar esto: sin `timeout`,
# `requests` espera indefinidamente si el servidor acepta la conexión y no
# responde nunca — en un proceso 24/7 (recolector.py) eso no se manifiesta
# como un error sino como el scheduler entero congelado para siempre, sin
# ningún log ni excepción que lo delate.
TIMEOUT_SEGUNDOS = 30

# Reintentos con backoff exponencial para errores transitorios (5xx, 429 de
# rate limit, cortes de conexión). Sin esto, un hipo puntual de la API deja
# un hueco permanente en la tabla: recolector.collect() captura la excepción,
# loguea y sigue, y ese timestamp solo se vuelve a pedir si alguien reinicia
# el proceso (rellenar_huecos_al_inicio). backoff_factor=1 da esperas de
# 0s/2s/4s entre intentos.
REINTENTOS = Retry(
    total=3,
    backoff_factor=1,
    status_forcelist=(429, 500, 502, 503, 504),
    allowed_methods=frozenset(['GET']),
)


class OSRSGeAPI:
    def __init__(self):
        self.base_url = "https://prices.runescape.wiki/api/v1/osrs"
        self.headers = {
            'User-Agent': 'DataScience proyect - dorlandopg@gmail.com'
        }
        # Una sola Session reutilizada: mantiene viva la conexión TCP entre
        # llamadas (un backfill son miles de requests seguidos al mismo host)
        # y es donde se cuelga la política de reintentos de arriba.
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        adaptador = HTTPAdapter(max_retries=REINTENTOS)
        self.session.mount('https://', adaptador)
        self.session.mount('http://', adaptador)

    def _get(self, url):
        """GET con timeout y reintentos, punto único de salida a la red —
        antes cada método hacía su propio `requests.get` sin timeout."""
        response = self.session.get(url, timeout=TIMEOUT_SEGUNDOS)
        response.raise_for_status()
        return response.json()

    def get_latest_prices(self):
        """Obtiene los precios más recientes de todos los items"""
        return self._procesar_prices(self._get(f"{self.base_url}/latest"))

    def get_historical_5min(self, timestamp=None):
        """
        Obtiene datos históricos de 5 minutos.
        Si no se especifica timestamp, obtiene los más recientes.

        OJO: "los más recientes" NO es el último bucket cerrado — la API
        devuelve el ANTEPENÚLTIMO (verificado empíricamente: a las 13:40 UTC
        devuelve el bucket de las 13:30, pese a que el de las 13:35 ya está
        disponible pidiéndolo explícitamente). Ver
        recolector._ultimo_bucket_cerrado: la recolección programada pasa
        siempre un timestamp explícito justamente por esto.
        """
        url = f"{self.base_url}/5m"
        if timestamp:
            url += f"?timestamp={timestamp}"
        return self._procesar_data_historicals(self._get(url))

    def get_historical_1h(self, timestamp=None):
        """
        Obtiene datos históricos de una hora.
        Si no se especifica timestamp, obtiene los más recientes — con la
        misma salvedad que get_historical_5min: sin timestamp la API
        devuelve el bucket de hace DOS horas, no el de hace una.
        """
        url = f"{self.base_url}/1h"
        if timestamp:
            url += f"?timestamp={timestamp}"
        return self._procesar_data_historicals(self._get(url))

    def get_historical_6h(self, timestamp=None):
        """
        Obtiene datos históricos de seis horas.
        Si no se especifica timestamp, obtiene los más recientes (misma
        salvedad que get_historical_5min).
        """
        url = f"{self.base_url}/6h"
        if timestamp:
            url += f"?timestamp={timestamp}"
        return self._procesar_data_historicals(self._get(url))

    def get_item_mapping(self):
        """Obtiene el mapeo de ID a nombre de item"""
        return self._get(f"{self.base_url}/mapping")

    def obtener_precios_item(self, item_id : int, timestep : str):
        """
        Obtiene una serie de tiempo de los precios del item de id dado.

            id - (required) Item id to return a time-series for.
            timestep - (required) Timestep of the time-series. Valid options are "5m", "1h", "6h" and "24h".

        """
        url = f"{self.base_url}/timeseries?timestep={timestep}&id={item_id}"
        return self._procesar_series_id(self._get(url), item_id)

## Procesadores
    def _procesar_data_historicals(self, data_response_json ):
        """
        Filtra los ítems que en ese bucket no tuvieron transacciones de las
        dos puntas (avgHighPrice/avgLowPrice): sin las dos no hay margen de
        flip que calcular ni target de entrenamiento posible.

        Esto deja HUECOS en la serie de un ítem poco líquido — medido contra
        la API real: Elysian spirit shield pierde el 28% de las horas, un
        ítem de 3rd age el 96%. No es un bug de acá (el dato realmente no
        existe), pero sí es la razón por la que preprocesamiento.py reindexa
        cada serie a la grilla regular antes de calcular lags/medias/target:
        sin eso, "lag de 1 período" significa "la fila anterior que exista",
        que puede ser de hace 10 horas, y el target deja de ser el
        movimiento de un período.
        """
        processed_data = []

        for item_id, data in data_response_json['data'].items():
            if data.get('avgHighPrice') and data.get('avgLowPrice'):
                processed_data.append({
                    'item_id': int(item_id),
                    'timestamp': data_response_json['timestamp'],
                    'avg_high_price': data['avgHighPrice'],  ## high es precio de compra mas alto
                    'avg_low_price': data['avgLowPrice'], ## precio de venta mas bajo
                    'high_volume': data.get('highPriceVolume', 0),
                    'low_volume': data.get('lowPriceVolume', 0)
                })

        return pd.DataFrame(processed_data)

    def _procesar_prices(self, data_response_json):
        processed_data = []
        for item_id, data in data_response_json['data'].items():
            if data.get('high') and data.get('low'):
                processed_data.append({
                    'item_id' : int(item_id),
                    'high' : data['high'],
                    'highTime' : data['highTime'],
                    'low': data['low'],
                    'lowTime': data['lowTime']
                })

        return pd.DataFrame(processed_data)

    def _procesar_series_id( self, data_response_json, id ):
        processed_data = []

        for data in data_response_json['data']:
            if data.get('avgHighPrice') and data.get('avgLowPrice'):
                processed_data.append({
                    'item_id' : id,
                    'timestamp': data['timestamp'],
                    'avg_high_price': data['avgHighPrice'],  ## high es precio de compra mas alto
                    'avg_low_price': data['avgLowPrice'], ## precio de venta mas bajo
                    'high_volume': data.get('highPriceVolume', 0),
                    'low_volume': data.get('lowPriceVolume', 0)
                })
        return pd.DataFrame(processed_data)


