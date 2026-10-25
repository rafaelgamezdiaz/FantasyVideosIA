# Fantasy AI Studio - Restauración y Super-Resolución Local de Imagen y Video

Aplicación de escritorio nativa para Windows 10/11 orientada a la restauración, mejora cromática, eliminación de ruido y super-resolución de alta fidelidad para imágenes y videos mediante Inteligencia Artificial, ejecutada 100% en local sin telemetría ni consumo de APIs de pago.

---

## 1. Requisitos Previos e Instalación de Dependencias

- **Sistema Operativo**: Windows 10 / Windows 11 (64 bits).
- **Intérprete**: Python 3.10 o superior (verificado en Python 3.13 x64).
- **Herramientas**: FFmpeg (incluido o detectable en PATH).

Instalación de paquetes requeridos:
```powershell
pip install -r requirements.txt
```

---

## 2. Ejecución en Modo Desarrollo

Para iniciar la aplicación directamente desde el código fuente:
```powershell
python main.py
```

La aplicación detectará automáticamente el hardware del sistema (procesador, tarjetas gráficas dedicadas o integradas como Intel Iris Xe, proveedores DirectML/CPU) y cargará la interfaz gráfica oscura con soporte para Drag & Drop y visor interactivo de división antes/después.

---

## 3. Compilación del Ejecutable (.exe)

La compilación está automatizada mediante el script `build_exe.py`, el cual utiliza PyInstaller configurado en modalidad `--onedir` para optimizar los tiempos de arranque y el rendimiento de las librerías dinámicas C++ (ONNX Runtime, DirectML, Qt).

### Procedimiento:
Ejecute en la terminal PowerShell en la raíz del proyecto:
```powershell
python build_exe.py
```

### Proceso que ejecuta el compilador:
1. Verifica si PyInstaller está instalado (lo instala si falta).
2. Detecta `icono.jpg` en la raíz y genera automáticamente el archivo multirresolución `icono.ico` (16x16 hasta 256x256).
3. Compila `main.py` integrando el icono, dependencias de DirectML, FFmpeg y metadatos de Windows.
4. Genera la distribución final en la ruta:
   ```
   dist/FantasyAIStudio/
   ├── FantasyAIStudio.exe      # Cargador ejecutable de la aplicación
   ├── icono.ico / icono.jpg    # Iconos asociados
   └── _internal/               # Entorno embebido (Python DLLs, ONNX Runtime, Qt, modelos)
   ```

---

## 4. Almacenamiento y Gestión de Modelos de Inteligencia Artificial

Los modelos no se descargan de golpe obligatoriamente; la aplicación incluye un modelo residual integrado ultra-liviano (`models/builtin/fast_sr_x2.onnx`) para operar de forma inmediata sin conexión.

### Ubicación de almacenamiento:
- **En modo desarrollo (`python main.py`)**:
  Se almacenan en la carpeta `models/` en la raíz del proyecto.
- **En el ejecutable compilado (`FantasyAIStudio.exe`)**:
  1. **Prioridad 1**: Si existe una carpeta `models/` junto al ejecutable (`FantasyAIStudio\models\`), la aplicación leerá y descargará los modelos directamente allí.
  2. **Prioridad 2**: De no existir dicha carpeta contigua, se almacenan en el subdirectorio interno: `FantasyAIStudio\_internal\models\`.

### Catálogo de Modelos Soportados:
| Categoría | Nombre del Modelo | Archivo en Disco | Función Principal |
| :--- | :--- | :--- | :--- |
| **Super-Resolución** | Real-ESRGAN x2plus | `RealESRGAN_x2plus.onnx` | Escalado 2x de alto rendimiento para video e imagen. |
| **Super-Resolución** | Real-ESRGAN x4plus | `RealESRGAN_x4plus.onnx` | Escalado 4x de alta fidelidad fotográfica. |
| **Super-Resolución** | Real-ESRGAN Anime 6B | `RealESRGAN_x4plus_anime_6B.onnx` | Escalado 4x ligero para ilustraciones y animación. |
| **Super-Resolución** | BSRGAN HD Studio | `BSRGAN.pth` | Tratamiento de fotografía antigua degradada con compresión severa. |
| **Super-Resolución** | Fast Residual SR x2 | `fast_sr_x2.onnx` (builtin) | Motor integrado sin necesidad de descargas externas. |
| **Restauración Facial** | GFPGAN v1.4 | `GFPGANv1.4.onnx` / `.pth` | Reconstrucción de rostros desenfocados o comprimidos. |
| **Colorización** | DDColor Neural Colorizer | `ddcolor_tiny.onnx` | Colorización automática para fotos/video blanco y negro o sepia. |
| **Multimodal VLM** | Qwen2-VL (2B Instruct) | `Qwen2-VL-2B-Instruct-Q4_K_M.gguf` | Análisis visual profundo e interpretación de lenguaje natural. |
| **Multimodal VLM** | Moondream2 | `moondream2_decoder_quantized.onnx` | Modelo ligero ONNX para diagnóstico visual e instrucciones. |

*Nota: Los selectores de la interfaz marcan automáticamente como `(Descargado)` aquellos modelos cuyos pesos ya se encuentren presentes en la carpeta correspondiente.*

---

## 5. Portabilidad: Traslado a Otra Computadora

Para transferir la aplicación a un equipo sin Python instalado:

1. **Copiar la carpeta completa de distribución**:
   Copie la carpeta entera `dist/FantasyAIStudio/`. **No traslade únicamente el archivo `.exe`**, ya que este requiere la subcarpeta `_internal/` para cargar las dependencias y librerías dinámicas (`python313.dll`, DirectML, etc.).
2. **Traslado de modelos ya descargados (Opcional, para uso sin internet)**:
   Copie la carpeta `models/` y colóquela dentro de `FantasyAIStudio/` al mismo nivel que `FantasyAIStudio.exe`. De este modo, la aplicación detectará los modelos como `(Descargado)` desde el primer inicio sin requerir conexión a la red.
3. **Descarga en destino**:
   Si traslada únicamente `dist/FantasyAIStudio/` sin modelos previos, al pulsar los botones de descarga dentro de la aplicación, esta descargará los pesos de forma automática dentro del directorio de la aplicación.
