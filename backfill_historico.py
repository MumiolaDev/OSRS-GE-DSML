"""
backfill_historico.py — corrida manual para rellenar huecos de precios_1h/
5m/6h sin reiniciar el recolector.

Desde que `recolector.py` rellena huecos solo en cada arranque
(`rellenar_huecos_al_inicio`), este script ya no es el camino principal
para eso — alcanza con reiniciar el proceso. Se mantiene para cuando
convenga rellenar huecos SIN reiniciar el recolector en vivo (ej. mientras
sigue recolectando, o para forzar la revisión sin esperar al próximo
arranque). Reusa exactamente la misma función.

Es seguro correrlo con recolector.py corriendo en paralelo: usa la misma DB
con INSERT OR IGNORE, así que no duplica nada; puede haber algún choque
puntual de lock de SQLite bajo mucha concurrencia, pero el timeout por
defecto de sqlite3 lo absorbe.
"""

import logging

from base_de_datos import OSRSBaseDatos
from recolector import rellenar_huecos_al_inicio, job_diario

DB_PATH = 'data/osrs_ge.db'

logging.basicConfig(level=logging.INFO, format='%(asctime)s |--| %(levelname)s |--| %(message)s')


def main():
    db = OSRSBaseDatos(DB_PATH)
    rellenar_huecos_al_inicio(db)
    logging.info("Relleno de huecos completo. Reentrenamiento final con todo el historial...")
    job_diario(db)
    logging.info("Listo.")


if __name__ == '__main__':
    main()
