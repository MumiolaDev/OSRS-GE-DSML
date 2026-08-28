import os
import sys

# Los módulos del proyecto viven en la raíz del repo, no en un paquete
# instalable — agregar la raíz a sys.path para poder importarlos desde
# tests/ sin tener que instalar nada.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
