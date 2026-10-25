import sys
import os
import ctypes
import traceback
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt

from ui.main_window import MainWindow

def exception_hook(exctype, value, tb):
    traceback.print_exception(exctype, value, tb)
    sys.__excepthook__(exctype, value, tb)

sys.excepthook = exception_hook

def enable_windows_dark_titlebar(window):
    """
    Applies Windows 10/11 DWM dark titlebar attribute to match the application aesthetic.
    """
    if sys.platform == "win32":
        try:
            hwnd = int(window.winId())
            DWMWA_USE_IMMERSIVE_DARK_MODE = 20
            set_window_attribute = ctypes.windll.dwmapi.DwmSetWindowAttribute
            value = ctypes.c_int(1)
            # Try Windows 11 / Windows 10 20H1+
            res = set_window_attribute(hwnd, DWMWA_USE_IMMERSIVE_DARK_MODE, ctypes.byref(value), ctypes.sizeof(value))
            if res != 0:
                # Fallback for Windows 10 1809 / 1903
                DWMWA_USE_IMMERSIVE_DARK_MODE_OLD = 19
                set_window_attribute(hwnd, DWMWA_USE_IMMERSIVE_DARK_MODE_OLD, ctypes.byref(value), ctypes.sizeof(value))
        except Exception:
            pass

from PySide6.QtGui import QIcon

def get_asset_path(filename: str) -> str:
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_path, filename)

def main():
    if sys.platform == "win32":
        try:
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("Fantasy.AIStudio.App.1.0")
        except Exception:
            pass

    # Set modern High DPI scaling behavior
    QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = QApplication(sys.argv)
    app.setApplicationName("Fantasy AI Studio")
    app.setOrganizationName("Fantasy")

    # Load application icon
    icon_path = get_asset_path("icono.ico")
    if not os.path.exists(icon_path):
        icon_path = get_asset_path("icono.jpg")
    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))

    window = MainWindow()
    if os.path.exists(icon_path):
        window.setWindowIcon(QIcon(icon_path))
    window.show()
    enable_windows_dark_titlebar(window)

    sys.exit(app.exec())

if __name__ == "__main__":
    main()
