"""Application entry point."""

import sys

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication

from config import APP_NAME, ORGANIZATION_NAME, SETTINGS_NAME
from gui import MainWindow


def main():
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    app = QApplication(sys.argv)
    app.setApplicationName(SETTINGS_NAME)
    app.setOrganizationName(ORGANIZATION_NAME)
    app.setApplicationDisplayName(APP_NAME)

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
