import os
import sys
import time
import cv2
import numpy as np
from typing import Optional, Tuple, List, Dict

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QComboBox, QSlider, QCheckBox, QProgressBar,
    QFileDialog, QMessageBox, QGroupBox, QSplitter, QPlainTextEdit,
    QFrame, QTabWidget
)
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QIcon, QDragEnterEvent, QDropEvent, QCloseEvent

from core.config import AppConfig
from core.model_manager import ModelManager
from core.engine import InferenceEngine
from core.video_processor import VideoProcessor
from core.nlp_agent import NLPAgent, NLPPipelineAction
from ui.split_view import SplitComparisonView
from ui.style import DARK_THEME_QSS

class PromptTextEdit(QPlainTextEdit):
    returnPressed = Signal()

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and not (event.modifiers() & Qt.KeyboardModifier.ShiftModifier):
            self.returnPressed.emit()
            event.accept()
        else:
            super().keyPressEvent(event)


class ImageWorker(QThread):
    progress_signal = Signal(float, str)
    finished_signal = Signal(object)
    error_signal = Signal(str)

    def __init__(
        self,
        engine: InferenceEngine,
        img_bgr: np.ndarray,
        face_restore: bool,
        face_fidelity: float,
        denoise: float,
        enhance_lighting: bool = False,
        lighting_strength: float = 0.5,
        replace_color_bgr: Optional[Tuple[int, int, int]] = None,
        remove_object: bool = False,
        swap_face_img: Optional[np.ndarray] = None,
        profile: str = "balanced",
        colorize: bool = False
    ):
        super().__init__()
        self.engine = engine
        self.img_bgr = img_bgr
        self.face_restore = face_restore
        self.face_fidelity = face_fidelity
        self.denoise = denoise
        self.enhance_lighting = enhance_lighting
        self.lighting_strength = lighting_strength
        self.replace_color_bgr = replace_color_bgr
        self.remove_object = remove_object
        self.swap_face_img = swap_face_img
        self.profile = profile
        self.colorize = colorize
        self._is_cancelled = False

    def cancel(self):
        self._is_cancelled = True

    def run(self):
        try:
            res = self.engine.process_image(
                img_bgr=self.img_bgr,
                enable_face_restore=self.face_restore,
                face_fidelity=self.face_fidelity,
                denoise_strength=self.denoise,
                enhance_lighting=self.enhance_lighting,
                lighting_strength=self.lighting_strength,
                replace_color_bgr=self.replace_color_bgr,
                remove_object=self.remove_object,
                swap_face_img=self.swap_face_img,
                profile=self.profile,
                colorize=self.colorize,
                progress_callback=lambda p, msg: self.progress_signal.emit(p, msg),
                cancel_check=lambda: self._is_cancelled
            )
            self.finished_signal.emit(res)
        except Exception as e:
            self.error_signal.emit(str(e))


class VideoWorker(QThread):
    progress_signal = Signal(int, int, float, float)
    finished_signal = Signal(str)
    error_signal = Signal(str)
    log_signal = Signal(str)

    def __init__(
        self,
        processor: VideoProcessor,
        input_path: str,
        output_path: str,
        encoder: str,
        crf: int,
        face_restore: bool,
        face_fidelity: float,
        denoise: float,
        enhance_lighting: bool = False,
        lighting_strength: float = 0.5,
        replace_color_bgr: Optional[Tuple[int, int, int]] = None,
        remove_object: bool = False,
        swap_face_img: Optional[np.ndarray] = None,
        profile: str = "balanced",
        colorize: bool = False
    ):
        super().__init__()
        self.processor = processor
        self.input_path = input_path
        self.output_path = output_path
        self.encoder = encoder
        self.crf = crf
        self.face_restore = face_restore
        self.face_fidelity = face_fidelity
        self.denoise = denoise
        self.enhance_lighting = enhance_lighting
        self.lighting_strength = lighting_strength
        self.replace_color_bgr = replace_color_bgr
        self.remove_object = remove_object
        self.swap_face_img = swap_face_img
        self.profile = profile
        self.colorize = colorize
        self._is_cancelled = False

    def cancel(self):
        self._is_cancelled = True

    def run(self):
        try:
            success = self.processor.process_video(
                input_path=self.input_path,
                output_path=self.output_path,
                encoder=self.encoder,
                crf=self.crf,
                enable_face_restore=self.face_restore,
                face_fidelity=self.face_fidelity,
                denoise_strength=self.denoise,
                enhance_lighting=self.enhance_lighting,
                lighting_strength=self.lighting_strength,
                replace_color_bgr=self.replace_color_bgr,
                remove_object=self.remove_object,
                swap_face_img=self.swap_face_img,
                profile=self.profile,
                colorize=self.colorize,
                progress_callback=lambda idx, total, fps, el: self.progress_signal.emit(idx, total, fps, el),
                cancel_check=lambda: self._is_cancelled
            )
            if success:
                self.finished_signal.emit(self.output_path)
            else:
                self.error_signal.emit("El procesamiento de video fue cancelado o finalizó inesperadamente.")
        except Exception as e:
            self.error_signal.emit(str(e))


class ModelDownloadWorker(QThread):
    progress_signal = Signal(int, int, float)
    finished_signal = Signal(bool, str)

    def __init__(self, manager: ModelManager, model_name: str):
        super().__init__()
        self.manager = manager
        self.model_name = model_name
        self._is_cancelled = False

    def cancel(self):
        self._is_cancelled = True

    def run(self):
        try:
            ok = self.manager.download_model(
                self.model_name,
                progress_callback=lambda cur, tot, spd: self.progress_signal.emit(cur, tot, spd),
                cancel_check=lambda: self._is_cancelled
            )
            self.finished_signal.emit(ok, self.model_name)
        except Exception as e:
            self.finished_signal.emit(False, str(e))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Fantasy AI - Restauración y Super-Resolución Local de Imagen y Video")
        self.resize(1340, 860)
        self.setAcceptDrops(True)
        self.setStyleSheet(DARK_THEME_QSS)

        # Window Icon
        base_dir = getattr(sys, "_MEIPASS", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        for icon_cand in ["icono.ico", "icono.jpg"]:
            cand_path = os.path.join(base_dir, icon_cand)
            if os.path.exists(cand_path):
                self.setWindowIcon(QIcon(cand_path))
                break

        # Core components
        self.config = AppConfig()
        self.model_manager = ModelManager(self.config.models_dir)
        self.engine = InferenceEngine(
            preferred_provider=self.config.preferred_provider,
            tile_size=self.config.default_tile_size,
            tile_overlap=self.config.default_tile_overlap,
            num_threads=self.config.cpu_info["physical_cores"]
        )
        self.video_processor = VideoProcessor(self.config.ffmpeg_path, self.engine) if self.config.ffmpeg_path else None

        # State
        self.current_file_path: Optional[str] = None
        self.is_video_mode: bool = False
        self.loaded_image_bgr: Optional[np.ndarray] = None
        self.enhanced_image_bgr: Optional[np.ndarray] = None
        self.reference_face_bgr: Optional[np.ndarray] = None
        self.active_worker: Optional[QThread] = None

        self._build_ui()
        self._load_default_model()
        self._log(f"Sistema inicializado: CPU {self.config.cpu_info['name']} ({self.config.cpu_info['physical_cores']} núcleos)")
        self._log(f"Dispositivo GPU detectado: {self.config.gpus[0]['name']}")
        self._log(f"Motor activo: {self.engine.active_provider} | Codificador recomendado: {self.config.get_recommended_encoder()}")

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(10)

        # 1. Top Hardware Status Bar
        main_layout.addWidget(self._create_hardware_panel())

        # 2. Middle Body: Splitter with Left Controls and Right Comparison View
        body_splitter = QSplitter(Qt.Orientation.Horizontal)
        body_splitter.setHandleWidth(3)

        controls_container = self._create_controls_panel()
        body_splitter.addWidget(controls_container)

        self.split_view = SplitComparisonView()
        body_splitter.addWidget(self.split_view)
        body_splitter.setStretchFactor(0, 0)
        body_splitter.setStretchFactor(1, 1)

        main_layout.addWidget(body_splitter, stretch=1)

        # 3. Bottom Progress and Status Section
        main_layout.addWidget(self._create_bottom_panel())

    def _create_hardware_panel(self) -> QWidget:
        panel = QFrame()
        panel.setStyleSheet("background-color: #161B22; border: 1px solid #30363D; border-radius: 6px; padding: 6px;")
        layout = QHBoxLayout(panel)
        layout.setContentsMargins(10, 4, 10, 4)

        gpu_name = self.config.gpus[0]["name"] if self.config.gpus else "Integrated GPU"
        badge_style = "font-weight: 600; font-size: 11px; padding: 3px 8px; border-radius: 4px; background-color: #21262D;"

        lbl_cpu = QLabel(f"💻 CPU: {self.config.cpu_info['name'][:30]}")
        lbl_cpu.setStyleSheet(badge_style + " color: #79C0FF;")
        layout.addWidget(lbl_cpu)

        lbl_gpu = QLabel(f"🎮 GPU: {gpu_name}")
        lbl_gpu.setStyleSheet(badge_style + " color: #7EE787;")
        layout.addWidget(lbl_gpu)

        self.lbl_provider = QLabel(f"⚡ Inferencia: {self.engine.active_provider}")
        self.lbl_provider.setStyleSheet(badge_style + " color: #FFA657;")
        layout.addWidget(self.lbl_provider)

        encoder_name = self.config.get_recommended_encoder()
        lbl_enc = QLabel(f"🎬 Video HW: {encoder_name}")
        lbl_enc.setStyleSheet(badge_style + " color: #D2A8FF;")
        layout.addWidget(lbl_enc)

        layout.addStretch()
        return panel

    def _create_controls_panel(self) -> QWidget:
        container = QWidget()
        container.setFixedWidth(410)
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 8, 0)
        layout.setSpacing(10)

        # 1. Top Compact File Card
        grp_file = QGroupBox("Entrada Multimedia")
        file_layout = QVBoxLayout(grp_file)
        file_layout.setSpacing(6)
        file_layout.setContentsMargins(10, 10, 10, 10)

        btn_browse = QPushButton("📁 Cargar Imagen o Video...")
        btn_browse.clicked.connect(self._browse_file)
        file_layout.addWidget(btn_browse)

        self.lbl_file_info = QLabel("Arrastre un archivo o use el botón superior.")
        self.lbl_file_info.setWordWrap(True)
        self.lbl_file_info.setStyleSheet("color: #8B949E; font-size: 11px;")
        file_layout.addWidget(self.lbl_file_info)
        layout.addWidget(grp_file)

        # 2. Main Tabbed Workspace
        self.tab_widget = QTabWidget()

        # TAB 1: Prompt & IA
        tab_prompt = QWidget()
        t1_layout = QVBoxLayout(tab_prompt)
        t1_layout.setSpacing(8)
        t1_layout.setContentsMargins(8, 12, 8, 8)

        t1_layout.addWidget(QLabel("Instrucciones en Lenguaje Natural (IA):"))
        self.txt_prompt = PromptTextEdit()
        self.txt_prompt.setPlaceholderText("Ej: 'Mejora la iluminación, rostros y aplica modo ultra'...")
        self.txt_prompt.setMaximumHeight(70)
        self.txt_prompt.returnPressed.connect(self._start_processing)
        t1_layout.addWidget(self.txt_prompt)

        btn_run_instruction = QPushButton("🚀 Ejecutar Instrucción (Enter)")
        btn_run_instruction.setObjectName("btnPrimary")
        btn_run_instruction.clicked.connect(self._start_processing)
        t1_layout.addWidget(btn_run_instruction)

        vlm_box = QHBoxLayout()
        vlm_box.addWidget(QLabel("Modelo VLM (Visión/Prompt):"))
        self.cmb_vlm_tab1 = QComboBox()
        self.cmb_vlm_tab1.currentIndexChanged.connect(self._on_vlm_tab1_changed)
        vlm_box.addWidget(self.cmb_vlm_tab1)
        t1_layout.addLayout(vlm_box)

        source_box = QHBoxLayout()
        source_box.addWidget(QLabel("Aplicar sobre:"))
        self.cmb_input_source = QComboBox()
        self.cmb_input_source.addItem("Imagen Original", "original")
        self.cmb_input_source.addItem("Resultado Anterior (Modo Iterativo)", "result")
        source_box.addWidget(self.cmb_input_source)
        t1_layout.addLayout(source_box)

        t1_layout.addWidget(QLabel("Sugerencias Rápidas:"))
        chips_layout1 = QHBoxLayout()
        btn_chip1 = QPushButton("✨ Auto-Mejora")
        btn_chip1.clicked.connect(lambda: self.txt_prompt.setPlainText("Mejora automáticamente la iluminación, suprime el ruido y restaura rostros en modo ultra"))
        chips_layout1.addWidget(btn_chip1)

        btn_chip2 = QPushButton("☀️ Iluminación")
        btn_chip2.clicked.connect(lambda: self.txt_prompt.setPlainText("Mejorar iluminación y contraste de sombras en la escena"))
        chips_layout1.addWidget(btn_chip2)

        btn_chip3 = QPushButton("✂️ Eliminar")
        btn_chip3.clicked.connect(lambda: self.txt_prompt.setPlainText("Eliminar objeto central y rellenar coherentemente"))
        chips_layout1.addWidget(btn_chip3)
        t1_layout.addLayout(chips_layout1)

        chips_layout2 = QHBoxLayout()
        btn_chip4 = QPushButton("🎨 Colorear B&W")
        btn_chip4.clicked.connect(lambda: self.txt_prompt.setPlainText("Colorear foto en blanco y negro a color con IA"))
        chips_layout2.addWidget(btn_chip4)

        btn_chip5 = QPushButton("👤 Face Swap")
        btn_chip5.clicked.connect(lambda: self.txt_prompt.setPlainText("Reemplazar rostro con imagen de referencia"))
        chips_layout2.addWidget(btn_chip5)
        t1_layout.addLayout(chips_layout2)

        t1_layout.addSpacing(6)
        self.btn_ref_face = QPushButton("📷 Cargar Rostro de Referencia (Face Swap)...")
        self.btn_ref_face.clicked.connect(self._select_reference_face)
        t1_layout.addWidget(self.btn_ref_face)

        self.lbl_ref_face_info = QLabel("Sin rostro de referencia seleccionado.")
        self.lbl_ref_face_info.setStyleSheet("color: #8B949E; font-size: 11px;")
        t1_layout.addWidget(self.lbl_ref_face_info)
        t1_layout.addStretch()

        self.tab_widget.addTab(tab_prompt, "Prompt e IA")

        # TAB 2: Modelos & Calidad
        tab_models = QWidget()
        t2_layout = QVBoxLayout(tab_models)
        t2_layout.setSpacing(10)
        t2_layout.setContentsMargins(8, 12, 8, 8)

        t2_layout.addWidget(QLabel("Perfil de Rendimiento y Tiempo:"))
        self.cmb_profile = QComboBox()
        self.cmb_profile.addItem("Equilibrado (Recomendado, Alta Calidad)", "balanced")
        self.cmb_profile.addItem("Rápido (Minutos, Super-Resolución 2x Ligera)", "fast")
        self.cmb_profile.addItem("Ultra (Fidelidad de Estudio, Doble Pasada)", "ultra")
        t2_layout.addWidget(self.cmb_profile)

        t2_layout.addWidget(QLabel("Modelo de Inteligencia Artificial (Super-Resolución):"))
        self.cmb_model = QComboBox()
        self._populate_models_combo()
        self.cmb_model.currentIndexChanged.connect(self._on_model_changed)
        t2_layout.addWidget(self.cmb_model)

        self.btn_download_model = QPushButton("Descargar Modelo Seleccionado")
        self.btn_download_model.setObjectName("btnDownload")
        self.btn_download_model.clicked.connect(self._download_selected_model)
        t2_layout.addWidget(self.btn_download_model)

        self.lbl_model_desc = QLabel()
        self.lbl_model_desc.setWordWrap(True)
        self.lbl_model_desc.setStyleSheet("color: #8B949E; font-size: 11px;")
        t2_layout.addWidget(self.lbl_model_desc)

        t2_layout.addSpacing(6)
        t2_layout.addWidget(QLabel("Modelo de Visión y Lenguaje (Multimodal VLM):"))
        self.cmb_vlm_model = QComboBox()
        self._populate_vlm_combo()
        self.cmb_vlm_model.currentIndexChanged.connect(self._on_vlm_model_changed)
        t2_layout.addWidget(self.cmb_vlm_model)

        self.btn_download_vlm = QPushButton("Descargar Modelo VLM Seleccionado")
        self.btn_download_vlm.setObjectName("btnDownload")
        self.btn_download_vlm.clicked.connect(self._download_selected_vlm_model)
        t2_layout.addWidget(self.btn_download_vlm)

        self.lbl_vlm_desc = QLabel()
        self.lbl_vlm_desc.setWordWrap(True)
        self.lbl_vlm_desc.setStyleSheet("color: #8B949E; font-size: 11px;")
        t2_layout.addWidget(self.lbl_vlm_desc)

        t2_layout.addWidget(QLabel("Tamaño de Mosaico (Adaptive Tiling VRAM):"))
        self.cmb_tile_size = QComboBox()
        self.cmb_tile_size.addItem("256 x 256 (Recomendado Intel Iris Xe)", 256)
        self.cmb_tile_size.addItem("128 x 128 (Ultra-bajo consumo VRAM)", 128)
        self.cmb_tile_size.addItem("384 x 384 (Equilibrado CPU/GPU)", 384)
        self.cmb_tile_size.addItem("512 x 512 (GPU dedicada alta VRAM)", 512)
        self.cmb_tile_size.currentIndexChanged.connect(self._on_tile_size_changed)
        t2_layout.addWidget(self.cmb_tile_size)
        t2_layout.addStretch()

        self.tab_widget.addTab(tab_models, "Modelos y Escala")

        # TAB 3: Ajustes Manuales
        tab_settings = QWidget()
        t3_layout = QVBoxLayout(tab_settings)
        t3_layout.setSpacing(10)
        t3_layout.setContentsMargins(8, 12, 8, 8)

        self.chk_face_restore = QCheckBox("Restauración Facial (GFPGAN / CodeFormer)")
        self.chk_face_restore.setChecked(True)
        t3_layout.addWidget(self.chk_face_restore)

        self.chk_colorize = QCheckBox("Colorizar Blanco y Negro / Sepia (DDColor IA)")
        self.chk_colorize.setChecked(False)
        t3_layout.addWidget(self.chk_colorize)

        face_fidelity_box = QHBoxLayout()
        face_fidelity_box.addWidget(QLabel("Fidelidad Facial (w):"))
        self.lbl_fidelity_val = QLabel("0.80")
        face_fidelity_box.addWidget(self.lbl_fidelity_val)
        t3_layout.addLayout(face_fidelity_box)

        self.sld_face_fidelity = QSlider(Qt.Orientation.Horizontal)
        self.sld_face_fidelity.setRange(50, 100)
        self.sld_face_fidelity.setValue(80)
        self.sld_face_fidelity.valueChanged.connect(lambda v: self.lbl_fidelity_val.setText(f"{v/100:.2f}"))
        t3_layout.addWidget(self.sld_face_fidelity)

        denoise_box = QHBoxLayout()
        denoise_box.addWidget(QLabel("Reducción de Ruido:"))
        self.lbl_denoise_val = QLabel("0.0")
        denoise_box.addWidget(self.lbl_denoise_val)
        t3_layout.addLayout(denoise_box)

        self.sld_denoise = QSlider(Qt.Orientation.Horizontal)
        self.sld_denoise.setRange(0, 10)
        self.sld_denoise.setValue(0)
        self.sld_denoise.valueChanged.connect(lambda v: self.lbl_denoise_val.setText(f"{v/10:.1f}"))
        t3_layout.addWidget(self.sld_denoise)

        self.lbl_video_enc = QLabel("Codificador Video FFmpeg:")
        t3_layout.addWidget(self.lbl_video_enc)
        self.cmb_video_enc = QComboBox()
        for enc in self.config.available_encoders:
            self.cmb_video_enc.addItem(enc)
        t3_layout.addWidget(self.cmb_video_enc)
        t3_layout.addStretch()

        self.tab_widget.addTab(tab_settings, "Ajustes Finos")

        layout.addWidget(self.tab_widget, stretch=1)

        # 3. Fixed Bottom Actions Bar
        actions_box = QHBoxLayout()
        self.btn_process = QPushButton("Iniciar Restauración")
        self.btn_process.setObjectName("btnPrimary")
        self.btn_process.clicked.connect(self._start_processing)
        actions_box.addWidget(self.btn_process)

        self.btn_cancel = QPushButton("Cancelar")
        self.btn_cancel.setObjectName("btnCancel")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self._cancel_processing)
        actions_box.addWidget(self.btn_cancel)
        layout.addLayout(actions_box)

        extra_actions_box = QHBoxLayout()
        self.btn_promote_result = QPushButton("🔄 Usar como Nueva Base")
        self.btn_promote_result.setEnabled(False)
        self.btn_promote_result.setToolTip("Establece el resultado actual como la imagen de entrada para continuar editando de forma iterativa.")
        self.btn_promote_result.clicked.connect(self._promote_result_to_base)
        extra_actions_box.addWidget(self.btn_promote_result)

        self.btn_save = QPushButton("💾 Guardar...")
        self.btn_save.setEnabled(False)
        self.btn_save.clicked.connect(self._save_result)
        extra_actions_box.addWidget(self.btn_save)
        layout.addLayout(extra_actions_box)

        return container

    def _create_bottom_panel(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        # Progress bar and stats
        info_layout = QHBoxLayout()
        self.lbl_status = QLabel("Listo.")
        self.lbl_status.setStyleSheet("font-weight: 600; color: #58A6FF;")
        info_layout.addWidget(self.lbl_status)

        self.lbl_metrics = QLabel("")
        self.lbl_metrics.setStyleSheet("color: #8B949E; font-size: 11px;")
        info_layout.addWidget(self.lbl_metrics, alignment=Qt.AlignmentFlag.AlignRight)
        layout.addLayout(info_layout)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        # Terminal Log
        self.log_console = QPlainTextEdit()
        self.log_console.setObjectName("logConsole")
        self.log_console.setReadOnly(True)
        self.log_console.setMaximumHeight(90)
        layout.addWidget(self.log_console)

        return container

    def _log(self, text: str):
        timestamp = time.strftime("%H:%M:%S")
        self.log_console.appendPlainText(f"[{timestamp}] {text}")
        self.log_console.verticalScrollBar().setValue(self.log_console.verticalScrollBar().maximum())

    def _populate_models_combo(self):
        current_model = self.cmb_model.currentData() if hasattr(self, "cmb_model") else None
        self.cmb_model.blockSignals(True)
        self.cmb_model.clear()

        for m_name, info in self.model_manager.catalog.items():
            if info.category == "Super-Resolution":
                is_dl = self.model_manager.is_model_downloaded(m_name)
                display_name = f"{m_name} (Descargado)" if is_dl else m_name
                self.cmb_model.addItem(display_name, userData=m_name)

        builtin_name = "Fast Residual SR x2 (Built-in Ligero)"
        is_builtin_dl = self.model_manager.is_model_downloaded(builtin_name)
        display_builtin = f"{builtin_name} (Descargado)" if is_builtin_dl else builtin_name
        self.cmb_model.addItem(display_builtin, userData=builtin_name)

        if current_model:
            for idx in range(self.cmb_model.count()):
                if self.cmb_model.itemData(idx) == current_model:
                    self.cmb_model.setCurrentIndex(idx)
                    break

        self.cmb_model.blockSignals(False)

    def _populate_vlm_combo(self):
        current_model = self.cmb_vlm_model.currentData() if hasattr(self, "cmb_vlm_model") else None

        combos = []
        if hasattr(self, "cmb_vlm_model"):
            combos.append(self.cmb_vlm_model)
        if hasattr(self, "cmb_vlm_tab1"):
            combos.append(self.cmb_vlm_tab1)

        for combo in combos:
            combo.blockSignals(True)
            combo.clear()

        for m_name, info in self.model_manager.catalog.items():
            if info.category == "Multimodal-VLM":
                is_dl = self.model_manager.is_model_downloaded(m_name)
                display_name = f"{m_name} (Descargado)" if is_dl else m_name
                for combo in combos:
                    combo.addItem(display_name, userData=m_name)

        if current_model:
            for combo in combos:
                for idx in range(combo.count()):
                    if combo.itemData(idx) == current_model:
                        combo.setCurrentIndex(idx)
                        break

        for combo in combos:
            combo.blockSignals(False)

    def _on_vlm_tab1_changed(self, idx: int):
        if hasattr(self, "cmb_vlm_model") and self.cmb_vlm_model.currentIndex() != idx:
            self.cmb_vlm_model.blockSignals(True)
            self.cmb_vlm_model.setCurrentIndex(idx)
            self.cmb_vlm_model.blockSignals(False)
            self._on_vlm_model_changed(idx)

    def _on_vlm_model_changed(self, idx: int):
        if hasattr(self, "cmb_vlm_tab1") and self.cmb_vlm_tab1.currentIndex() != idx:
            self.cmb_vlm_tab1.blockSignals(True)
            self.cmb_vlm_tab1.setCurrentIndex(idx)
            self.cmb_vlm_tab1.blockSignals(False)

        model_name = self.cmb_vlm_model.currentData() or self.cmb_vlm_model.currentText()
        if model_name and model_name.endswith(" (Descargado)"):
            model_name = model_name[:-len(" (Descargado)")]
        if model_name in self.model_manager.catalog:
            info = self.model_manager.catalog[model_name]
            is_dl = self.model_manager.is_model_downloaded(model_name)
            size_mb = info.size_bytes / (1024 * 1024)
            if is_dl:
                status_text = "Descargado y Disponible en Disco"
                self.btn_download_vlm.setText(f"Re-descargar Modelo VLM ({size_mb:.1f} MB)")
                self.btn_download_vlm.setEnabled(True)
            else:
                status_text = "No descargado"
                self.btn_download_vlm.setText(f"⬇️ Descargar Modelo VLM ({size_mb:.1f} MB)")
                self.btn_download_vlm.setEnabled(True)
            self.lbl_vlm_desc.setText(f"{info.description}\nArchivo: {info.filename} | Estado: {status_text}")

    def _download_selected_vlm_model(self):
        if self.active_worker and self.active_worker.isRunning():
            QMessageBox.warning(self, "Operación en curso", "Hay una tarea o descarga en ejecución. Espere a que finalice o cancélela.")
            return

        model_name = self.cmb_vlm_model.currentData() or self.cmb_vlm_model.currentText()
        if model_name and model_name.endswith(" (Descargado)"):
            model_name = model_name[:-len(" (Descargado)")]
        if model_name not in self.model_manager.catalog:
            return

        if self.model_manager.is_model_downloaded(model_name):
            reply = QMessageBox.question(
                self,
                "Modelo VLM ya descargado",
                f"El modelo '{model_name}' ya se encuentra descargado y listo para usar en disco.\n\n¿Desea volver a descargarlo y reemplazar el archivo existente?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            if reply != QMessageBox.StandardButton.Yes:
                self._log(f"Descarga cancelada: Se conserva el modelo existente '{model_name}'.")
                return

        self._log(f"Iniciando descarga de modelo VLM: {model_name}...")
        self.btn_download_vlm.setEnabled(False)
        self.btn_process.setEnabled(False)

        self.active_worker = ModelDownloadWorker(self.model_manager, model_name)
        self.active_worker.progress_signal.connect(self._on_download_progress)
        self.active_worker.finished_signal.connect(self._on_download_finished)
        self.active_worker.start()

    def _load_default_model(self):
        self._on_model_changed(self.cmb_model.currentIndex())
        if hasattr(self, "cmb_vlm_model"):
            self._on_vlm_model_changed(self.cmb_vlm_model.currentIndex())

    def _on_model_changed(self, idx: int):
        model_name = self.cmb_model.currentData() or self.cmb_model.currentText()
        if model_name and model_name.endswith(" (Descargado)"):
            model_name = model_name[:-len(" (Descargado)")]
        if model_name in self.model_manager.catalog:
            info = self.model_manager.catalog[model_name]
            is_dl = self.model_manager.is_model_downloaded(model_name)
            size_mb = info.size_bytes / (1024 * 1024)
            if is_dl:
                status_text = "Descargado y Operativo"
                self.btn_download_model.setText(f"Re-descargar Modelo ({size_mb:.1f} MB)")
                self.btn_download_model.setEnabled(True)
                model_path = self.model_manager.get_model_path(model_name)
                self.engine.load_model(model_path, scale=info.scale)
                self.lbl_provider.setText(f"⚡ Inferencia: {self.engine.active_provider}")
            else:
                status_text = "No descargado (usando motor integrado temporal)"
                self.btn_download_model.setText(f"⬇️ Descargar Modelo ({size_mb:.1f} MB)")
                self.btn_download_model.setEnabled(True)
                fallback_path = self.model_manager.get_model_path("builtin")
                if fallback_path:
                    self.engine.load_model(fallback_path, scale=2)
            self.lbl_model_desc.setText(f"{info.description}\nEscala: {info.scale}x | Estado: {status_text}")
        else:
            # Builtin fallback
            model_path = self.model_manager.get_model_path("builtin")
            if model_path and os.path.exists(model_path):
                self.engine.load_model(model_path, scale=2)
            self.lbl_model_desc.setText("Modelo nativo embebido sin descargas externas. Listo para uso local inmediato.")
            self.btn_download_model.setText("Modelo Embebido Instalado")
            self.btn_download_model.setEnabled(False)

    def _on_tile_size_changed(self, idx: int):
        size = self.cmb_tile_size.currentData()
        self.engine.tile_size = size
        self._log(f"Tamaño de mosaico VRAM ajustado a: {size}x{size}")

    def _download_selected_model(self):
        if self.active_worker and self.active_worker.isRunning():
            QMessageBox.warning(self, "Operación en curso", "Hay una tarea o descarga en ejecución. Espere a que finalice o cancélela.")
            return

        model_name = self.cmb_model.currentData() or self.cmb_model.currentText()
        if model_name and model_name.endswith(" (Descargado)"):
            model_name = model_name[:-len(" (Descargado)")]
        if model_name not in self.model_manager.catalog:
            return

        if self.model_manager.is_model_downloaded(model_name):
            reply = QMessageBox.question(
                self,
                "Modelo ya descargado",
                f"El modelo '{model_name}' ya se encuentra descargado y listo para usar en disco.\n\n¿Desea volver a descargarlo y reemplazar el archivo existente?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            if reply != QMessageBox.StandardButton.Yes:
                self._log(f"Descarga cancelada: Se conserva el modelo existente '{model_name}'.")
                return

        self._log(f"Iniciando descarga de: {model_name}...")
        self.btn_download_model.setEnabled(False)
        self.btn_process.setEnabled(False)

        self.active_worker = ModelDownloadWorker(self.model_manager, model_name)
        self.active_worker.progress_signal.connect(self._on_download_progress)
        self.active_worker.finished_signal.connect(self._on_download_finished)
        self.active_worker.start()

    def _on_download_progress(self, current: int, total: int, speed_mb: float):
        pct = int((current / max(total, 1)) * 100)
        self.progress_bar.setValue(pct)
        cur_mb = current / (1024 * 1024)
        tot_mb = total / (1024 * 1024)
        self.lbl_status.setText(f"Descargando modelo: {pct}% ({cur_mb:.1f}/{tot_mb:.1f} MB a {speed_mb:.2f} MB/s)")

    def _on_download_finished(self, ok: bool, message: str):
        if self.active_worker:
            self.active_worker.wait(300)
            self.active_worker = None

        self.btn_process.setEnabled(True)
        if ok:
            self._log(f"Modelo descargado y verificado correctamente: {message}")
            self._populate_models_combo()
            self._populate_vlm_combo()
            self._on_model_changed(self.cmb_model.currentIndex())
            if hasattr(self, "cmb_vlm_model"):
                self._on_vlm_model_changed(self.cmb_vlm_model.currentIndex())
        else:
            self._log(f"Error al descargar modelo: {message}")
            self.btn_download_model.setEnabled(True)
            if hasattr(self, "btn_download_vlm"):
                self.btn_download_vlm.setEnabled(True)
        self.lbl_status.setText("Listo.")
        self.progress_bar.setValue(0)

    def _browse_file(self):
        filter_str = "Archivos multimedia (*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff *.mp4 *.mkv *.mov *.avi);;Imágenes (*.png *.jpg *.jpeg *.webp *.bmp *.tiff);;Videos (*.mp4 *.mkv *.mov *.avi)"
        path, _ = QFileDialog.getOpenFileName(self, "Seleccionar Imagen o Video", "", filter_str)
        if path:
            self._load_file(path)

    def _load_file(self, file_path: str):
        if not os.path.exists(file_path):
            return

        self.current_file_path = file_path
        ext = os.path.splitext(file_path)[1].lower()
        video_exts = [".mp4", ".mkv", ".mov", ".avi", ".webm"]

        self.is_video_mode = ext in video_exts

        if self.is_video_mode:
            if not self.video_processor:
                QMessageBox.critical(self, "Error FFmpeg", "No se encontró el ejecutable FFmpeg para procesar video.")
                return
            meta = self.video_processor.probe_video(file_path)
            self.lbl_file_info.setText(
                f"🎬 Video: {os.path.basename(file_path)}\n"
                f"Resolución: {meta.width}x{meta.height} | {meta.fps:.2f} FPS\n"
                f"Duración: {meta.duration:.1f}s (~{meta.total_frames} cuadros)"
            )
            # Extract first frame as preview
            cap = cv2.VideoCapture(file_path)
            ret, frame = cap.read()
            cap.release()
            if ret:
                self.loaded_image_bgr = frame
                self.split_view.set_single_preview(frame)
        else:
            img = cv2.imread(file_path)
            if img is None:
                QMessageBox.critical(self, "Error de Archivo", "No se pudo decodificar el archivo de imagen.")
                return
            self.loaded_image_bgr = img
            h, w = img.shape[:2]
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            self.lbl_file_info.setText(
                f"🖼️ Imagen: {os.path.basename(file_path)}\n"
                f"Resolución: {w}x{h} px | {size_mb:.2f} MB"
            )
            self.split_view.set_single_preview(img)

        self._log(f"Archivo cargado: {os.path.basename(file_path)}")
        self.btn_save.setEnabled(False)

    def _select_reference_face(self):
        filter_str = "Imágenes (*.png *.jpg *.jpeg *.webp *.bmp)"
        path, _ = QFileDialog.getOpenFileName(self, "Seleccionar Rostro de Referencia", "", filter_str)
        if path and os.path.exists(path):
            img = cv2.imread(path)
            if img is not None:
                self.reference_face_bgr = img
                self.lbl_ref_face_info.setText(f"Rostro cargado: {os.path.basename(path)}")
                self._log(f"Rostro de referencia cargado para Face Swap: {os.path.basename(path)}")

    def _promote_result_to_base(self):
        if self.enhanced_image_bgr is None:
            return
        self.loaded_image_bgr = self.enhanced_image_bgr.copy()
        h, w = self.loaded_image_bgr.shape[:2]
        self.split_view.set_single_preview(self.loaded_image_bgr)
        self.lbl_file_info.setText(
            f"🖼️ Base actualizada (Iterativa)\n"
            f"Resolución: {w}x{h} px"
        )
        self._log("Resultado promovido a imagen base para el siguiente procesamiento.")

    def _start_processing(self):
        if not self.current_file_path:
            QMessageBox.warning(self, "Atención", "Por favor seleccione una imagen o video primero.")
            return

        if self.active_worker and self.active_worker.isRunning():
            QMessageBox.warning(self, "Operación en curso", "Hay una tarea de procesamiento o descarga en ejecución.")
            return

        self.btn_process.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.btn_save.setEnabled(False)
        self.btn_promote_result.setEnabled(False)
        self.progress_bar.setValue(0)

        # Baseline controls
        face_restore = self.chk_face_restore.isChecked()
        face_fidelity = self.sld_face_fidelity.value() / 100.0
        denoise = self.sld_denoise.value() / 10.0
        selected_profile = self.cmb_profile.currentData() or "balanced"
        colorize = self.chk_colorize.isChecked()

        enhance_lighting = False
        lighting_strength = 0.5
        replace_color_bgr = None
        remove_object = False

        # Select target image buffer (original vs previous result)
        use_prev = (self.cmb_input_source.currentData() == "result") and (self.enhanced_image_bgr is not None)
        base_img = self.enhanced_image_bgr if use_prev else self.loaded_image_bgr

        # Natural language prompt interpretation
        prompt_text = self.txt_prompt.toPlainText().strip()
        if prompt_text:
            action = NLPAgent.parse_instruction(prompt_text, base_img)
            self._log(f"Plan de Acción IA: {action.summary}")
            selected_profile = action.profile
            face_restore = action.face_restore or face_restore
            face_fidelity = action.face_fidelity if action.face_restore else face_fidelity
            denoise = max(action.denoise, denoise)
            enhance_lighting = action.enhance_lighting
            lighting_strength = action.lighting_strength
            colorize = action.colorize or colorize
            if action.replace_color_query:
                replace_color_bgr = action.replace_color_query[1]
            if action.remove_object_query:
                remove_object = True

        if self.is_video_mode:
            dir_name = os.path.dirname(self.current_file_path)
            base_name, ext = os.path.splitext(os.path.basename(self.current_file_path))
            out_video_path = os.path.join(dir_name, f"{base_name}_restored_fantasy{ext}")

            encoder_clean = self.cmb_video_enc.currentText().split()[0]
            self._log(f"Iniciando procesamiento de video [{selected_profile.upper()}]: {out_video_path}")
            self._log(f"Codificador de salida: {encoder_clean}")

            self.active_worker = VideoWorker(
                processor=self.video_processor,
                input_path=self.current_file_path,
                output_path=out_video_path,
                encoder=encoder_clean,
                crf=17,
                face_restore=face_restore,
                face_fidelity=face_fidelity,
                denoise=denoise,
                enhance_lighting=enhance_lighting,
                lighting_strength=lighting_strength,
                replace_color_bgr=replace_color_bgr,
                remove_object=remove_object,
                swap_face_img=self.reference_face_bgr,
                profile=selected_profile,
                colorize=colorize
            )
            self.active_worker.progress_signal.connect(self._on_video_progress)
            self.active_worker.finished_signal.connect(self._on_video_finished)
            self.active_worker.error_signal.connect(self._on_worker_error)
            self.active_worker.start()
        else:
            source_label = "Resultado Iterativo" if use_prev else "Imagen Original"
            self._log(f"Iniciando procesamiento de imagen [{selected_profile.upper()}] (Base: {source_label}) ({self.engine.active_provider})...")
            self.active_worker = ImageWorker(
                engine=self.engine,
                img_bgr=base_img,
                face_restore=face_restore,
                face_fidelity=face_fidelity,
                denoise=denoise,
                enhance_lighting=enhance_lighting,
                lighting_strength=lighting_strength,
                replace_color_bgr=replace_color_bgr,
                remove_object=remove_object,
                swap_face_img=self.reference_face_bgr,
                profile=selected_profile,
                colorize=colorize
            )
            self.active_worker.progress_signal.connect(self._on_image_progress)
            self.active_worker.finished_signal.connect(self._on_image_finished)
            self.active_worker.error_signal.connect(self._on_worker_error)
            self.active_worker.start()

    def _on_image_progress(self, progress: float, msg: str):
        self.progress_bar.setValue(int(progress * 100))
        self.lbl_status.setText(msg)

    def _on_image_finished(self, enhanced_bgr: np.ndarray):
        if self.active_worker:
            self.active_worker.wait(300)
            self.active_worker = None

        use_prev = (self.cmb_input_source.currentData() == "result") and (self.enhanced_image_bgr is not None)
        before_img = self.enhanced_image_bgr if use_prev else self.loaded_image_bgr
        self.enhanced_image_bgr = enhanced_bgr
        self.split_view.set_images_from_numpy(before_img, enhanced_bgr)
        self.btn_process.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.btn_save.setEnabled(True)
        self.btn_promote_result.setEnabled(True)
        self.lbl_status.setText("¡Imagen restaurada con éxito!")
        self.progress_bar.setValue(100)
        h, w = enhanced_bgr.shape[:2]
        self._log(f"Restauración completa. Nueva resolución: {w}x{h} px")

    def _on_video_progress(self, current_frame: int, total_frames: int, fps: float, elapsed: float):
        pct = int((current_frame / max(total_frames, 1)) * 100)
        self.progress_bar.setValue(pct)
        remaining_sec = max(0, (total_frames - current_frame) / max(fps, 0.001))
        self.lbl_status.setText(f"Procesando video: {current_frame}/{total_frames} cuadros ({pct}%)")
        self.lbl_metrics.setText(f"FPS: {fps:.2f} | Transcurrido: {elapsed:.1f}s | Restante: {remaining_sec:.1f}s")

    def _on_video_finished(self, out_path: str):
        if self.active_worker:
            self.active_worker.wait(300)
            self.active_worker = None

        self.btn_process.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.lbl_status.setText("¡Video procesado y codificado con éxito!")
        self.progress_bar.setValue(100)
        self._log(f"Video guardado exitosamente en: {out_path}")
        QMessageBox.information(self, "Proceso Completado", f"El video restaurado ha sido guardado en:\n{out_path}")

    def _on_worker_error(self, err_msg: str):
        if self.active_worker:
            self.active_worker.wait(300)
            self.active_worker = None

        self.btn_process.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.lbl_status.setText("Error en el procesamiento.")
        self._log(f"ERROR: {err_msg}")
        QMessageBox.critical(self, "Error de Ejecución", err_msg)

    def _cancel_processing(self):
        if self.active_worker and self.active_worker.isRunning():
            self._log("Cancelando operación en curso...")
            self.lbl_status.setText("Cancelando...")
            self.active_worker.cancel()
            self.active_worker.wait(1000)
            self.btn_cancel.setEnabled(False)

    def _save_result(self):
        if self.enhanced_image_bgr is None:
            return
        save_path, _ = QFileDialog.getSaveFileName(self, "Guardar Imagen Restaurada", "restored.png", "PNG (*.png);;JPG (*.jpg);;TIFF (*.tiff)")
        if save_path:
            cv2.imwrite(save_path, self.enhanced_image_bgr)
            self._log(f"Imagen guardada en: {save_path}")
            QMessageBox.information(self, "Guardado", f"Archivo guardado exitosamente:\n{save_path}")

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        urls = event.mimeData().urls()
        if urls:
            file_path = urls[0].toLocalFile()
            self._load_file(file_path)

    def closeEvent(self, event: QCloseEvent):
        if self.active_worker and self.active_worker.isRunning():
            self._log("Cierre de aplicación: esperando finalización de hilo activo...")
            self.active_worker.cancel()
            self.active_worker.wait(2000)
        event.accept()
