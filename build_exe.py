import os
import sys
import shutil
import subprocess

def build():
    """
    Automates the compilation of Fantasy AI into a standalone Windows executable using PyInstaller.
    """
    print("=" * 60)
    print("Iniciando compilacion de Fantasy AI para Windows (.exe)...")
    print("=" * 60)

    # 1. Ensure pyinstaller is installed
    try:
        import PyInstaller
    except ImportError:
        print("[1/4] Instalando PyInstaller...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])

    # 2. Locate onnxruntime and models directories
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, "models")
    main_script = os.path.join(base_dir, "main.py")
    icon_ico = os.path.join(base_dir, "icono.ico")
    icon_jpg = os.path.join(base_dir, "icono.jpg")

    if not os.path.exists(icon_ico) and os.path.exists(icon_jpg):
        try:
            from PIL import Image
            img = Image.open(icon_jpg)
            img.save(icon_ico, format="ICO", sizes=[(256, 256), (128, 128), (64, 64), (48, 48), (32, 32), (16, 16)])
        except Exception:
            pass

    # 3. Assemble optimized PyInstaller command
    # Exclude unused giant packages (torch, webengine, 3d, qml) to reduce build time from 20min to ~45sec
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--name=FantasyAIStudio",
        "--noconfirm",
        "--onedir",
        "--windowed",
        f"--add-data={models_dir};models",
    ]

    if os.path.exists(icon_ico):
        cmd.append(f"--icon={icon_ico}")
        cmd.append(f"--add-data={icon_ico};.")
    if os.path.exists(icon_jpg):
        cmd.append(f"--add-data={icon_jpg};.")

    cmd.extend([
        "--hidden-import=onnxruntime",
        "--hidden-import=onnxruntime.capi.onnxruntime_pybind11_state",
        "--hidden-import=imageio_ffmpeg",
        "--hidden-import=PySide6.QtCore",
        "--hidden-import=PySide6.QtGui",
        "--hidden-import=PySide6.QtWidgets",
        "--hidden-import=cv2",
        "--collect-submodules=onnxruntime",
        "--collect-data=onnxruntime",
        "--collect-binaries=onnxruntime",
        "--exclude-module=torch",
        "--exclude-module=torchvision",
        "--exclude-module=PySide6.QtWebEngine",
        "--exclude-module=PySide6.QtWebEngineCore",
        "--exclude-module=PySide6.QtWebEngineWidgets",
        "--exclude-module=PySide6.QtQml",
        "--exclude-module=PySide6.QtQuick",
        "--exclude-module=PySide6.Qt3DCore",
        "--exclude-module=PySide6.Qt3DRender",
        "--exclude-module=PySide6.Qt3DInput",
        "--exclude-module=PySide6.Qt3DLogic",
        "--exclude-module=PySide6.QtPdf",
        "--exclude-module=PySide6.QtBluetooth",
        "--exclude-module=scipy",
        "--exclude-module=pandas",
        "--exclude-module=IPython",
        main_script
    ])

    print("[2/4] Ejecutando empaquetador PyInstaller con dependencias DirectML...")
    res = subprocess.run(cmd)

    if res.returncode == 0:
        dist_dir = os.path.join(base_dir, "dist", "FantasyAIStudio")
        exe_path = os.path.join(dist_dir, "FantasyAIStudio.exe")
        print("=" * 60)
        print(f"COMPILACION EXITOSA!")
        print(f"Ejecutable generado en: {exe_path}")
        print("=" * 60)
    else:
        print(f"Error durante la compilacion. Codigo de salida: {res.returncode}")

if __name__ == "__main__":
    build()
