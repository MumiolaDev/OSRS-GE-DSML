#!/usr/bin/env bash
# Instala el recolector como servicio de usuario de systemd (Linux) y deja
# un lanzador de escritorio para la app. Idempotente: correrlo de nuevo
# regenera las unidades con las rutas actuales y recarga systemd.
#
#   ./deploy/instalar_servicio.sh              instala, habilita y arranca
#   ./deploy/instalar_servicio.sh --sin-iniciar  instala sin arrancar
#   ./deploy/instalar_servicio.sh --desinstalar  saca todo
#
# Ver docs/despliegue_24_7.md para el detalle y para las alertas de Telegram.
set -euo pipefail

PROYECTO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UNIDADES="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
LANZADORES="${XDG_DATA_HOME:-$HOME/.local/share}/applications"
SERVICIO="osrs-recolector.service"

# El intérprete del entorno virtual del proyecto si existe; si no, el del
# sistema. Importante: el servicio no hereda ningún `source .../activate`,
# así que la ruta al python correcto tiene que quedar escrita en el unit.
if [[ -x "$PROYECTO/osrs_env/bin/python" ]]; then
    PYTHON="$PROYECTO/osrs_env/bin/python"
elif [[ -x "$PROYECTO/.venv/bin/python" ]]; then
    PYTHON="$PROYECTO/.venv/bin/python"
else
    PYTHON="$(command -v python3)"
    echo "aviso: no encontré un entorno virtual en el proyecto, uso $PYTHON"
fi

desinstalar() {
    systemctl --user disable --now "$SERVICIO" 2>/dev/null || true
    rm -f "$UNIDADES/$SERVICIO" "$UNIDADES/osrs-recolector-fallo.service"
    rm -f "$LANZADORES/osrs-ge.desktop"
    systemctl --user daemon-reload
    echo "Desinstalado."
    exit 0
}

[[ "${1:-}" == "--desinstalar" ]] && desinstalar

render() {  # render <template> <destino>
    sed -e "s|@PROYECTO@|$PROYECTO|g" -e "s|@PYTHON@|$PYTHON|g" "$1" > "$2"
}

mkdir -p "$UNIDADES" "$LANZADORES"
render "$PROYECTO/deploy/osrs-recolector.service.template"       "$UNIDADES/$SERVICIO"
render "$PROYECTO/deploy/osrs-recolector-fallo.service.template" "$UNIDADES/osrs-recolector-fallo.service"
render "$PROYECTO/deploy/osrs-ge.desktop.template"               "$LANZADORES/osrs-ge.desktop"

systemctl --user daemon-reload

echo "Proyecto:    $PROYECTO"
echo "Intérprete:  $PYTHON"
echo "Unidad:      $UNIDADES/$SERVICIO"
echo "Lanzador:    $LANZADORES/osrs-ge.desktop"

if [[ "${1:-}" == "--sin-iniciar" ]]; then
    echo
    echo "Instalado sin arrancar. Para arrancarlo: systemctl --user enable --now $SERVICIO"
    exit 0
fi

systemctl --user enable --now "$SERVICIO"
echo
systemctl --user --no-pager --lines=0 status "$SERVICIO" || true

# Sin lingering, las unidades de usuario mueren al cerrar sesión y no
# arrancan hasta el próximo login. Para un recolector 24/7 en una máquina
# que se deja prendida, esto es lo que hace la diferencia entre "arranca con
# la PC" y "arranca cuando me logueo".
if [[ "$(loginctl show-user "$USER" -p Linger --value 2>/dev/null)" != "yes" ]]; then
    echo
    echo "FALTA UN PASO (pide autenticación, por eso no lo hace este script):"
    echo "    loginctl enable-linger $USER"
    echo "Sin eso el recolector arranca recién cuando iniciás sesión, y se corta al salir."
fi
