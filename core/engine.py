import os
import cv2
import math
import numpy as np
from typing import Optional, Tuple, Callable, List

import onnxruntime as ort
from core.image_ops import ImageEnhancementOps

class InferenceEngine:
    """
    High-fidelity AI inference engine supporting DirectML (Intel Iris Xe / AMD / Nvidia),
    CUDA, and multi-core CPU with adaptive tiling and seamless boundary blending.
    """
    def __init__(
        self,
        preferred_provider: str = "DmlExecutionProvider",
        tile_size: int = 256,
        tile_overlap: int = 32,
        num_threads: int = 4
    ):
        self.preferred_provider = preferred_provider
        self.tile_size = tile_size
        self.tile_overlap = tile_overlap
        self.num_threads = num_threads
        self.session: Optional[ort.InferenceSession] = None
        self.colorizer_session: Optional[ort.InferenceSession] = None
        self.active_provider: str = "CPUExecutionProvider"
        self.current_model_path: Optional[str] = None
        self.current_scale: int = 2

        # Initialize face detector
        self.face_cascade = None
        if hasattr(cv2, "CascadeClassifier") and hasattr(cv2, "data") and hasattr(cv2.data, "haarcascades"):
            cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
            if os.path.exists(cascade_path):
                try:
                    self.face_cascade = cv2.CascadeClassifier(cascade_path)
                except Exception:
                    self.face_cascade = None

    def get_colorizer_session(self, models_dir: str = "models") -> Optional[ort.InferenceSession]:
        """
        Loads DDColor neural colorizer session on-demand using CPUExecutionProvider
        to avoid DirectML graph fusion errors with attention/MatMul layers.
        """
        if self.colorizer_session is not None:
            return self.colorizer_session

        candidates = [
            os.path.join(models_dir, "ddcolor_tiny.onnx"),
            os.path.join(models_dir, "ddcolor.onnx"),
            "models/ddcolor_tiny.onnx"
        ]
        for p in candidates:
            if os.path.exists(p):
                try:
                    sess_options = ort.SessionOptions()
                    sess_options.intra_op_num_threads = self.num_threads
                    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                    self.colorizer_session = ort.InferenceSession(
                        p,
                        sess_options=sess_options,
                        providers=["CPUExecutionProvider"]
                    )
                    return self.colorizer_session
                except Exception:
                    pass
        return None

    def load_model(self, model_path: str, scale: int = 2) -> bool:
        """
        Initializes the ONNX runtime inference session with DirectML or CPU fallback.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = self.num_threads
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        available = ort.get_available_providers()

        # 1. Try DirectML in isolation
        if self.preferred_provider in available and self.preferred_provider != "CPUExecutionProvider":
            try:
                self.session = ort.InferenceSession(
                    model_path,
                    sess_options=sess_options,
                    providers=[self.preferred_provider]
                )
                self.active_provider = self.session.get_providers()[0]
                self.current_model_path = model_path
                self.current_scale = scale
                return True
            except (Exception, BaseException):
                self.session = None

        # 2. Stable CPU Fallback
        try:
            self.session = ort.InferenceSession(
                model_path,
                sess_options=sess_options,
                providers=["CPUExecutionProvider"]
            )
            self.active_provider = "CPUExecutionProvider"
            self.current_model_path = model_path
            self.current_scale = scale
            return True
        except (Exception, BaseException):
            return False

    def process_image(
        self,
        img_bgr: np.ndarray,
        enable_face_restore: bool = False,
        face_fidelity: float = 0.8,
        denoise_strength: float = 0.0,
        enhance_lighting: bool = False,
        lighting_strength: float = 0.5,
        replace_color_bgr: Optional[Tuple[int, int, int]] = None,
        remove_object: bool = False,
        swap_face_img: Optional[np.ndarray] = None,
        profile: str = "balanced",
        colorize: bool = False,
        progress_callback: Optional[Callable[[float, str], None]] = None,
        cancel_check: Optional[Callable[[], bool]] = None
    ) -> np.ndarray:
        """
        Executes end-to-end image restoration and multi-modal processing pipeline:
        1. Generative/Structural edits (Inpainting, Face Swap, Color Shift)
        2. Colorization of monochrome/B&W images (DDColor AI)
        3. Dynamic Range & Lighting enhancement (Retinex / CLAHE)
        4. Denoising pre-pass
        5. Seamless adaptive tiling super-resolution
        6. Face restoration and seamless blending
        """
        if img_bgr is None or img_bgr.size == 0:
            raise ValueError("Input image is empty or invalid.")

        h, w, c = img_bgr.shape
        processed = img_bgr.copy()

        if cancel_check and cancel_check():
            return processed

        # 0A. Face Swap (if reference face provided)
        if swap_face_img is not None and swap_face_img.size > 0:
            if progress_callback:
                progress_callback(0.05, "Ejecutando trasplante e integración facial (Face Swap)...")
            processed = ImageEnhancementOps.swap_faces(processed, swap_face_img)

        # 0B. Object removal / Inpainting
        if remove_object:
            if progress_callback:
                progress_callback(0.10, "Eliminando anomalías y aplicando inpainting coherente...")
            processed = ImageEnhancementOps.remove_salient_object(processed)

        # 0C. Color replacement
        if replace_color_bgr is not None:
            if progress_callback:
                progress_callback(0.15, "Aplicando remapeo cromático selectivo...")
            processed = ImageEnhancementOps.replace_color(processed, replace_color_bgr)

        # 0D. Colorization (B&W to full realistic color via DDColor)
        if colorize:
            if progress_callback:
                progress_callback(0.18, "Restaurando y generando colores realistas con IA (DDColor)...")
            color_sess = self.get_colorizer_session()
            processed = ImageEnhancementOps.colorize_image(processed, colorizer_session=color_sess)

        # 0E. Lighting enhancement
        if enhance_lighting:
            if progress_callback:
                progress_callback(0.22, "Mejorando iluminación y rango dinámico (Retinex/CLAHE)...")
            processed = ImageEnhancementOps.enhance_lighting_retinex(processed, strength=lighting_strength)

        # 1. Pre-pass Denoising
        if denoise_strength > 0:
            if progress_callback:
                progress_callback(0.25, "Aplicando reducción de ruido adaptativo...")
            d = int(denoise_strength * 5) * 2 + 3
            sigma = denoise_strength * 25.0
            processed = cv2.bilateralFilter(processed, d=d, sigmaColor=sigma, sigmaSpace=sigma)

        # 2. Super-Resolution via Adaptive Tiling
        if progress_callback:
            progress_callback(0.30, f"Iniciando super-resolución [{profile.upper()}] ({self.active_provider})...")

        if self.session is not None:
            upscaled = self._tiled_inference(processed, progress_callback, cancel_check)
        else:
            upscaled = self._algorithmic_upscale(processed, self.current_scale)

        # In ultra mode, apply extra fine-detail unsharp pass
        if profile == "ultra":
            gaussian = cv2.GaussianBlur(upscaled, (0, 0), 1.5)
            upscaled = np.clip(cv2.addWeighted(upscaled, 1.25, gaussian, -0.25, 0), 0, 255).astype(np.uint8)

        if cancel_check and cancel_check():
            return upscaled

        # 3. Face Restoration step
        if enable_face_restore and self.face_cascade is not None:
            if progress_callback:
                progress_callback(0.85, "Detectando y reconstruyendo microdetalles faciales...")
            upscaled = self._restore_faces(upscaled, face_fidelity, cancel_check)

        if progress_callback:
            progress_callback(1.0, "Procesamiento completado con éxito.")

        return upscaled

    def _tiled_inference(
        self,
        img_bgr: np.ndarray,
        progress_callback: Optional[Callable[[float, str], None]] = None,
        cancel_check: Optional[Callable[[], bool]] = None
    ) -> np.ndarray:
        """
        Divides the image into overlapping tiles, executes model inference on each,
        and reconstructs the output with linear cosine feathering to eliminate seams.
        """
        h, w, c = img_bgr.shape
        scale = self.current_scale
        tile = self.tile_size
        overlap = self.tile_overlap

        out_h, out_w = h * scale, w * scale
        output = np.zeros((out_h, out_w, c), dtype=np.float32)
        weight_map = np.zeros((out_h, out_w, 1), dtype=np.float32)

        # Calculate grid steps
        step = tile - overlap
        x_steps = int(math.ceil(max(w - overlap, 1) / step))
        y_steps = int(math.ceil(max(h - overlap, 1) / step))
        total_tiles = x_steps * y_steps

        # Create 2D feathering weight matrix for seamless blending
        tile_weight = self._create_tile_weight(tile * scale, overlap * scale)

        tile_idx = 0
        input_name = self.session.get_inputs()[0].name

        for y_idx in range(y_steps):
            y1 = min(y_idx * step, max(0, h - tile))
            y2 = min(y1 + tile, h)
            actual_tile_h = y2 - y1

            for x_idx in range(x_steps):
                if cancel_check and cancel_check():
                    return cv2.resize(img_bgr, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4)

                x1 = min(x_idx * step, max(0, w - tile))
                x2 = min(x1 + tile, w)
                actual_tile_w = x2 - x1

                tile_crop = img_bgr[y1:y2, x1:x2]

                # Run inference on tile
                tile_out = self._run_single_tile(tile_crop, input_name)

                # Output coordinate bounds
                out_y1, out_y2 = y1 * scale, y2 * scale
                out_x1, out_x2 = x1 * scale, x2 * scale

                # Trim weight to match actual crop if at border
                tw = tile_weight[: (y2 - y1) * scale, : (x2 - x1) * scale]

                output[out_y1:out_y2, out_x1:out_x2] += tile_out * tw
                weight_map[out_y1:out_y2, out_x1:out_x2] += tw

                tile_idx += 1
                if progress_callback and total_tiles > 0:
                    pct = 0.2 + 0.65 * (tile_idx / total_tiles)
                    progress_callback(pct, f"Procesando mosaico {tile_idx}/{total_tiles}...")

        # Normalize by accumulated weights
        np.maximum(weight_map, 1e-4, out=weight_map)
        output /= weight_map
        return np.clip(output, 0, 255).astype(np.uint8)

    def _run_single_tile(self, tile_bgr: np.ndarray, input_name: str) -> np.ndarray:
        """
        Executes normalized tensor inference on a single tile.
        """
        th, tw, _ = tile_bgr.shape
        # Convert BGR uint8 to RGB float32 [0.0, 1.0], shape [1, 3, H, W]
        rgb = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor_in = np.transpose(rgb, (2, 0, 1))[np.newaxis, ...]

        try:
            outputs = self.session.run(None, {input_name: tensor_in})
            out_tensor = outputs[0][0]  # Shape [3, H*scale, W*scale]
            out_rgb = np.transpose(out_tensor, (1, 2, 0))
            out_rgb = np.clip(out_rgb * 255.0, 0, 255).astype(np.uint8)
            out_bgr = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)
            return out_bgr.astype(np.float32)
        except Exception:
            # Algorithmic fallback if tile execution encounters an unexpected dimension error
            scaled = cv2.resize(tile_bgr, (tw * self.current_scale, th * self.current_scale), interpolation=cv2.INTER_LANCZOS4)
            return scaled.astype(np.float32)

    def _create_tile_weight(self, size: int, overlap: int) -> np.ndarray:
        """
        Generates linear-cosine feathering mask for border blending.
        """
        weight_1d = np.ones(size, dtype=np.float32)
        if overlap > 0 and 2 * overlap <= size:
            ramp = np.linspace(0.01, 1.0, overlap, dtype=np.float32)
            weight_1d[:overlap] = ramp
            weight_1d[-overlap:] = ramp[::-1]
        mask_2d = np.outer(weight_1d, weight_1d)[:, :, np.newaxis]
        return mask_2d

    def _algorithmic_upscale(self, img_bgr: np.ndarray, scale: int) -> np.ndarray:
        """
        High-fidelity edge-preserving upscaling with unsharp masking for immediate feedback.
        """
        h, w = img_bgr.shape[:2]
        upscaled = cv2.resize(img_bgr, (w * scale, h * scale), interpolation=cv2.INTER_LANCZOS4)
        gaussian = cv2.GaussianBlur(upscaled, (0, 0), 2.0)
        sharpened = cv2.addWeighted(upscaled, 1.4, gaussian, -0.4, 0)
        return np.clip(sharpened, 0, 255).astype(np.uint8)

    def _restore_faces(self, img_bgr: np.ndarray, fidelity: float, cancel_check: Optional[Callable[[], bool]] = None) -> np.ndarray:
        """
        Detects facial regions, applies texture refinement and reconstructive unsharp masking,
        and blends them back using soft elliptical feathering.
        """
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=4, minSize=(60, 60))
        if len(faces) == 0:
            return img_bgr

        result = img_bgr.copy()
        for (x, y, w, h) in faces:
            if cancel_check and cancel_check():
                break

            pad = int(w * 0.15)
            y1, y2 = max(0, y - pad), min(result.shape[0], y + h + pad)
            x1, x2 = max(0, x - pad), min(result.shape[1], x + w + pad)
            face_roi = result[y1:y2, x1:x2]

            # High-fidelity bilateral and detail enhancement
            smoothed = cv2.bilateralFilter(face_roi, d=9, sigmaColor=75, sigmaSpace=75)
            details = cv2.detailEnhance(face_roi, sigma_s=10, sigma_r=0.15)
            restored_face = cv2.addWeighted(details, fidelity, smoothed, (1.0 - fidelity) * 0.5, 0)

            # Create elliptical feather mask
            mask = np.zeros((y2 - y1, x2 - x1), dtype=np.float32)
            center = ((x2 - x1) // 2, (y2 - y1) // 2)
            axes = ((x2 - x1) // 2 - 4, (y2 - y1) // 2 - 4)
            cv2.ellipse(mask, center, axes, 0, 0, 360, 1.0, -1)
            mask = cv2.GaussianBlur(mask, (21, 21), 11)[:, :, np.newaxis]

            # Seamless paste-back
            blended = restored_face * mask + face_roi * (1.0 - mask)
            result[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)

        return result
