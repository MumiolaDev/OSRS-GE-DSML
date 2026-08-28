"""
paginas/pagina_configuracion.py — edición del token/chat_id de Telegram y
la ruta de la base de datos, persistidos en config.json (ver
configuracion.py) en vez de pedirle al usuario que setee variables de
entorno del sistema operativo a mano.
"""
from PySide6.QtWidgets import (
    QFormLayout, QLabel, QLineEdit, QPushButton, QVBoxLayout, QWidget,
)

import configuracion


class PaginaConfiguracion(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)

        layout.addWidget(QLabel(
            "El token y el chat id de Telegram se guardan en config.json (nunca se commitean). "
            "Si además hay variables de entorno TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID seteadas en "
            "el sistema (ej. un despliegue como servicio), esas tienen prioridad sobre lo que "
            "esté acá."
        ))

        form = QFormLayout()
        self.campo_token = QLineEdit()
        self.campo_token.setEchoMode(QLineEdit.Password)
        form.addRow("Telegram bot token:", self.campo_token)

        self.campo_chat_id = QLineEdit()
        form.addRow("Telegram chat id:", self.campo_chat_id)

        self.campo_db_path = QLineEdit()
        form.addRow("Ruta de la base de datos:", self.campo_db_path)

        layout.addLayout(form)

        self.label_estado = QLabel("")
        layout.addWidget(self.label_estado)

        boton_guardar = QPushButton("Guardar")
        boton_guardar.clicked.connect(self._guardar)
        layout.addWidget(boton_guardar)
        layout.addStretch()

        self._cargar()

    def _cargar(self):
        config = configuracion.cargar_configuracion()
        self.campo_token.setText(config.get('telegram_bot_token') or '')
        self.campo_chat_id.setText(config.get('telegram_chat_id') or '')
        self.campo_db_path.setText(config.get('db_path') or 'data/osrs_ge.db')

    def _guardar(self):
        config = configuracion.cargar_configuracion()
        config['telegram_bot_token'] = self.campo_token.text().strip()
        config['telegram_chat_id'] = self.campo_chat_id.text().strip()
        config['db_path'] = self.campo_db_path.text().strip() or 'data/osrs_ge.db'
        configuracion.guardar_configuracion(config)
        self.label_estado.setText(
            "Guardado. La ruta de la base de datos aplica recién la próxima vez que se abra la app."
        )
