"""
escritorio — app de escritorio (PySide6) del proyecto: reemplaza a
recolector.py corriendo como proceso NSSM aparte + dashboard.py como
servidor Streamlit aparte por una sola app que controla el recolector en
segundo plano y muestra las vistas que le importan al usuario para decidir
si comprar o vender (ver el plan de la fase "producto de escritorio v1").

No duplica lógica de negocio: reusa los módulos existentes del repo
(recolector.py, base_de_datos.py, metricas.py, prediccion.py, entrenador.py)
tal cual — este paquete es solo la capa de presentación y orquestación en
segundo plano.
"""
