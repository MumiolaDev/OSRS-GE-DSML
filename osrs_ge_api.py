import requests
import pandas as pd
from datetime import datetime, timedelta
import os
import time

class OSRSGeAPI:
    def __init__(self):
        self.base_url = "https://prices.runescape.wiki/api/v1/osrs"
        self.headers = {
            'User-Agent': 'DataScience proyect - dorlandopg@gmail.com'
        }
    
    def get_latest_prices(self):
        """Obtiene los precios más recientes de todos los items"""
        response = requests.get(
            f"{self.base_url}/latest", 
            headers=self.headers
        )

        response.raise_for_status()  

        df = self._procesar_prices(response.json())

        return df
    
    def get_historical_5min(self, timestamp=None):
        """
        Obtiene datos históricos de 5 minutos.
        Si no se especifica timestamp, obtiene los más recientes.
        """
        url = f"{self.base_url}/5m"
        if timestamp:
            url += f"?timestamp={timestamp}"
        
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()  
        
        df = self._procesar_data_historicals(response.json())

        return df
    
    def get_historical_1h(self, timestamp=None):
        """
        Obtiene datos históricos de una hora.
        Si no se especifica timestamp, obtiene los más recientes.
        """
        url = f"{self.base_url}/1h"
        if timestamp:
            url += f"?timestamp={timestamp}"
        
        
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()  
        
        df = self._procesar_data_historicals(response.json())
        return df
    
    def get_historical_6h(self, timestamp=None):
        """
        Obtiene datos históricos de seis horas.
        Si no se especifica timestamp, obtiene los más recientes.
        """
        url = f"{self.base_url}/6h"
        if timestamp:
            url += f"?timestamp={timestamp}"

        response = requests.get(url, headers=self.headers)
        response.raise_for_status()

        df = self._procesar_data_historicals(response.json())
        return df

    def get_item_mapping(self):
        """Obtiene el mapeo de ID a nombre de item"""
        response = requests.get(
            f"{self.base_url}/mapping", 
            headers=self.headers
        )
        response.raise_for_status()  
        return response.json()
    
    def obtener_precios_item(self, item_id : int, timestep : str):
        """
        Obtiene una serie de tiempo de los precios del item de id dado.

            id - (required) Item id to return a time-series for.
            timestep - (required) Timestep of the time-series. Valid options are "5m", "1h", "6h" and "24h".

        """

        url = f"{self.base_url}/timeseries"
        url += f"?timestep={timestep}"
        url += f"&id={item_id}"

        response = requests.get(
            url,
            headers=self.headers
        )
        response.raise_for_status()

        df = self._procesar_series_id(response.json(), item_id)
        return df

## Procesadores
    def _procesar_data_historicals(self, data_response_json ):
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


