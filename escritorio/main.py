"""
main.py — punto de entrada de la app de escritorio.

Uso: python -m escritorio.main
"""
import sys

from PySide6.QtWidgets import QApplication

from escritorio.ventana_principal import VentanaPrincipal


def main():
    app = QApplication(sys.argv)
    ventana = VentanaPrincipal()
    ventana.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
