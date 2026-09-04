"""
estado.py — ¿está sano el pipeline? Una respuesta en una pantalla, sin abrir
la app ni la DB a mano.

Pensado para tres usos:

    python estado.py            resumen legible en la terminal
    python estado.py --json     una línea de JSON para la barra de estado
                                (waybar: return-type "json")
    python estado.py --breve    una sola línea, para un prompt o un script

Es de SOLO LECTURA: no arranca nada, no escribe en la DB, no toca la API. Se
puede correr con el recolector andando sin interferir.

La pregunta que contesta no es "¿el proceso está vivo?" sino "¿los datos
están al día?", que no es lo mismo: un recolector corriendo que hace tres
horas que no inserta una fila (sin red, API devolviendo vacío, el loop de
schedule colgado en una request sin timeout) está tan roto como uno muerto,
pero `systemctl status` lo muestra verde. Por eso el veredicto sale de la
antigüedad del último dato horario, y el estado del proceso es un dato más.
"""
import argparse
import json
import sys
import time

import bloqueo
import configuracion
from base_de_datos import OSRSBaseDatos

# La recolección horaria corre a :01 y el dato tarda unos minutos en estar
# agregado del lado de la API, así que hasta ~2h de antigüedad es operación
# normal. Más allá de 6h ya se perdieron varios buckets seguidos.
ANTIGUEDAD_OK_H = 2
ANTIGUEDAD_GRAVE_H = 6

NOMBRES_TABLAS = {
    'precios_5m': '5 minutos',
    'precios_1h': '1 hora',
    'precios_6h': '6 horas',
    'precios_1h_diario': '1h archivado',
}


def _antiguedad(segundos):
    if segundos is None:
        return "—"
    if segundos < 60:
        return "recién"
    if segundos < 3600:
        return f"hace {int(segundos // 60)} min"
    if segundos < 86400:
        return f"hace {int(segundos // 3600)} h"
    return f"hace {int(segundos // 86400)} d"


def recolectar_estado(db_path=None, ahora=None):
    """
    Todo el estado del pipeline en un dict, para que las tres salidas
    (humana, JSON, breve) se armen del mismo lugar y no puedan discrepar.
    """
    db_path = db_path or configuracion.obtener('db_path') or 'data/osrs_ge.db'
    ahora = ahora or time.time()

    estado = {'db_path': db_path, 'ahora': int(ahora)}
    estado['recolector'] = bloqueo.hay_recolector_corriendo(db_path)

    db = OSRSBaseDatos(db_path)
    resumen = db.obtener_resumen_datos()
    estado['tablas'] = {
        tabla: {
            'filas': datos['filas'],
            'hasta_ts': datos['hasta_ts'],
            'antiguedad_s': (ahora - datos['hasta_ts']) if datos['hasta_ts'] else None,
            'dias_historial': datos['dias_historial'],
        }
        for tabla, datos in resumen.items()
    }

    conn = db.conectar()
    c = conn.cursor()
    c.execute('SELECT COUNT(*), MAX(calculado_en) FROM resumen_actual')
    n_screener, screener_ts = c.fetchone()
    estado['screener'] = {
        'items': n_screener,
        'calculado_en': screener_ts,
        'antiguedad_s': (ahora - screener_ts) if screener_ts else None,
    }
    conn.close()

    estado['modelos'] = [
        {
            'nombre': m['nombre'],
            'tipo': m['tipo'],
            'estado': m['estado'],
            'cadencia': m['cadencia'],
            'ultimo_entrenamiento_ts': m['ultimo_entrenamiento_ts'],
            'antiguedad_s': (
                (ahora - m['ultimo_entrenamiento_ts']) if m['ultimo_entrenamiento_ts'] else None
            ),
        }
        for m in db.listar_modelos_config()
    ]

    estado['telegram'] = bool(
        configuracion.obtener('telegram_bot_token') and configuracion.obtener('telegram_chat_id')
    )
    estado.update(_veredicto(estado))
    return estado


def _veredicto(estado):
    """
    'ok' / 'atencion' / 'error' + una línea explicando por qué. El orden de
    los chequeos importa: el primero que falla es el que se reporta, del más
    grave al menos.
    """
    antiguedad = (estado['tablas'].get('precios_1h') or {}).get('antiguedad_s')
    corriendo = estado['recolector'] is not None

    if antiguedad is None:
        return {'salud': 'error', 'motivo': "No hay ningún dato horario todavía."}
    if antiguedad > ANTIGUEDAD_GRAVE_H * 3600:
        return {'salud': 'error',
                'motivo': f"El último dato horario es de {_antiguedad(antiguedad)}."}
    if not corriendo:
        return {'salud': 'atencion',
                'motivo': "No hay ningún recolector corriendo."}
    if antiguedad > ANTIGUEDAD_OK_H * 3600:
        return {'salud': 'atencion',
                'motivo': f"El recolector corre pero el último dato es de {_antiguedad(antiguedad)}."}
    return {'salud': 'ok', 'motivo': f"Datos al día ({_antiguedad(antiguedad)})."}


def _linea_breve(estado):
    icono = {'ok': '✓', 'atencion': '!', 'error': '✗'}[estado['salud']]
    antiguedad = (estado['tablas'].get('precios_1h') or {}).get('antiguedad_s')
    return f"OSRS {icono} {_antiguedad(antiguedad)}"


def formato_humano(estado):
    lineas = []
    icono = {'ok': '✓', 'atencion': '⚠', 'error': '✗'}[estado['salud']]
    lineas.append(f"{icono}  {estado['motivo']}")
    lineas.append("")

    rec = estado['recolector']
    if rec:
        desde = _antiguedad(estado['ahora'] - rec['inicio']) if rec.get('inicio') else "—"
        lineas.append(f"Recolector    corriendo como {rec.get('origen', '?')} "
                      f"(PID {rec.get('pid', '?')}, arrancó {desde})")
    else:
        lineas.append("Recolector    detenido")
    lineas.append(f"Base de datos {estado['db_path']}")
    lineas.append(f"Telegram      {'configurado' if estado['telegram'] else 'sin configurar'}")
    lineas.append("")

    lineas.append(f"{'Datos':<16}{'Filas':>12}  {'Último':<12}{'Historial':>10}")
    for tabla, nombre in NOMBRES_TABLAS.items():
        d = estado['tablas'].get(tabla, {})
        dias = f"{d['dias_historial']:.1f} d" if d.get('dias_historial') is not None else "—"
        lineas.append(f"{nombre:<16}{d.get('filas', 0):>12,}  "
                      f"{_antiguedad(d.get('antiguedad_s')):<12}{dias:>10}")
    lineas.append("")

    scr = estado['screener']
    lineas.append(f"Screener      {scr['items']:,} ítems, recalculado "
                  f"{_antiguedad(scr['antiguedad_s'])}")

    if not estado['modelos']:
        lineas.append("Modelos       ninguno creado todavía "
                      "(la app de escritorio, pestaña 'Mis modelos')")
    else:
        lineas.append("Modelos")
        for m in estado['modelos']:
            lineas.append(f"  {m['nombre']:<24}{m['tipo']:<12}{m['estado']:<10}"
                          f"entrenado {_antiguedad(m['antiguedad_s'])}")
    return "\n".join(lineas)


def formato_waybar(estado):
    """
    El contrato de un módulo custom de waybar con return-type "json": una
    línea con text/tooltip/class. `class` es lo que engancha el color en el
    CSS (.custom-osrs.error, etc.).
    """
    return json.dumps({
        'text': _linea_breve(estado),
        'tooltip': formato_humano(estado).replace('\n', '\r'),
        'class': estado['salud'],
        'alt': estado['salud'],
    }, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    parser.add_argument('--json', action='store_true',
                        help='una línea de JSON para waybar (return-type "json")')
    parser.add_argument('--breve', action='store_true',
                        help='una sola línea de texto')
    parser.add_argument('--db', default=None, help='ruta de la DB (default: la de config.json)')
    args = parser.parse_args()

    try:
        estado = recolectar_estado(args.db)
    except Exception as e:
        # Una barra de estado no puede quedarse sin salida: si la DB no se
        # puede abrir, eso ES el estado y hay que poder mostrarlo.
        if args.json:
            print(json.dumps({'text': 'OSRS ✗', 'tooltip': str(e), 'class': 'error'},
                             ensure_ascii=False))
        else:
            print(f"✗  No se pudo leer el estado: {e}", file=sys.stderr)
        return 2

    if args.json:
        print(formato_waybar(estado))
    elif args.breve:
        print(_linea_breve(estado))
    else:
        print(formato_humano(estado))
    return 0 if estado['salud'] == 'ok' else 1


if __name__ == '__main__':
    sys.exit(main())
