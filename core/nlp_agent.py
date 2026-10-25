import re
import cv2
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from core.image_ops import ImageEnhancementOps

class NLPPipelineAction:
    def __init__(
        self,
        intent: str,
        profile: str = "balanced",
        upscale: int = 2,
        denoise: float = 0.0,
        enhance_lighting: bool = False,
        lighting_strength: float = 0.5,
        face_restore: bool = False,
        face_fidelity: float = 0.8,
        remove_object_query: Optional[str] = None,
        replace_color_query: Optional[Tuple[str, Tuple[int, int, int]]] = None,
        face_swap_active: bool = False,
        colorize: bool = False,
        summary: str = ""
    ):
        self.intent = intent
        self.profile = profile
        self.upscale = upscale
        self.denoise = denoise
        self.enhance_lighting = enhance_lighting
        self.lighting_strength = lighting_strength
        self.face_restore = face_restore
        self.face_fidelity = face_fidelity
        self.remove_object_query = remove_object_query
        self.replace_color_query = replace_color_query
        self.face_swap_active = face_swap_active
        self.colorize = colorize
        self.summary = summary


class NLPAgent:
    """
    Interprets natural language user instructions and inspects the image
    to construct an automated, optimal multi-stage processing pipeline.
    """
    COLOR_MAP = {
        "rojo": (0, 0, 255),
        "azul": (255, 0, 0),
        "verde": (0, 255, 0),
        "amarillo": (0, 255, 255),
        "negro": (20, 20, 20),
        "blanco": (240, 240, 240),
        "morado": (180, 0, 180),
        "naranja": (0, 165, 255),
        "dorado": (0, 215, 255)
    }

    @classmethod
    def analyze_image_properties(cls, img_bgr: np.ndarray) -> Dict[str, Any]:
        """
        Analyzes exposure, contrast, noise levels, and facial presence without external API calls.
        """
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        mean_brightness = float(np.mean(gray))
        contrast = float(np.std(gray))

        # Lapalcian variance for blur detection
        blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        # Face detection count (compatible with OpenCV 4.x and 5.x)
        num_faces = 0
        if hasattr(cv2, "CascadeClassifier") and hasattr(cv2, "data") and hasattr(cv2.data, "haarcascades"):
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            if os.path.exists(cascade_path):
                try:
                    face_cascade = cv2.CascadeClassifier(cascade_path)
                    faces = face_cascade.detectMultiScale(gray, 1.2, 4)
                    num_faces = len(faces)
                except Exception:
                    pass

        return {
            "mean_brightness": mean_brightness,
            "is_underexposed": mean_brightness < 80.0,
            "is_overexposed": mean_brightness > 190.0,
            "contrast": contrast,
            "is_low_contrast": contrast < 40.0,
            "blur_score": blur_score,
            "is_blurry": blur_score < 100.0,
            "num_faces": num_faces,
            "is_monochrome": ImageEnhancementOps.is_monochromatic_or_grayscale(img_bgr)
        }

    @classmethod
    def parse_instruction(cls, prompt: str, img_bgr: Optional[np.ndarray] = None) -> NLPPipelineAction:
        """
        Parses text prompts and cross-references with image metrics to construct an automated plan.
        """
        p = prompt.lower().strip()

        # Defaults
        profile = "balanced"
        upscale = 2
        denoise = 0.0
        enhance_lighting = False
        lighting_strength = 0.5
        face_restore = False
        face_fidelity = 0.8
        remove_object_query = None
        replace_color_query = None
        face_swap_active = False
        colorize = False
        summary_actions = []

        # 1. Profile detection
        if any(w in p for w in ["ultra", "maxima", "máxima", "estudio", "alta fidelidad", "lento", "mejor posible"]):
            profile = "ultra"
            upscale = 4
            summary_actions.append("Perfil Ultra (Fidelidad de Estudio)")
        elif any(w in p for w in ["rapido", "rápido", "minutos", "borrador", "ligero"]):
            profile = "fast"
            upscale = 2
            summary_actions.append("Perfil Rápido (Minutos)")
        else:
            profile = "balanced"
            summary_actions.append("Perfil Equilibrado")

        # 2. Colorization detection (B&W to full color)
        color_triggers = [
            "color", "colores", "colorear", "colorizar", "colorise", "colorize",
            "blanco y negro", "b&w", "byn", "b/n", "en colores", "darle color",
            "poner color", "a color", "añadir color", "agregar color", "se vea en colores",
            "vieja", "viejo", "antigua", "antiguo", "vintage", "envejecida", "rejuvenecer"
        ]
        if any(w in p for w in color_triggers):
            colorize = True
            summary_actions.append("Colorización IA de Blanco y Negro a Color (DDColor)")

        # Vintage / old photo general restoration triggers
        if any(w in p for w in ["vieja", "viejo", "antigua", "antiguo", "vintage", "envejecida", "deteriorada"]):
            enhance_lighting = True
            lighting_strength = 0.55
            face_restore = True
            face_fidelity = 0.85
            denoise = max(denoise, 0.4)
            summary_actions.append("Restauración Integral de Fotografía Antigua")

        # 3. Lighting and exposure
        if any(w in p for w in ["iluminacion", "iluminación", "luz", "aclarar", "oscuro", "brillo", "sombras"]):
            enhance_lighting = True
            lighting_strength = 0.7 if "mucha" in p or "bastante" in p else 0.5
            summary_actions.append(f"Realce de Iluminación y Contraste Adaptativo ({int(lighting_strength*100)}%)")

        # 4. Denoising / grain removal
        if any(w in p for w in ["ruido", "grano", "limpiar", "desenfocar", "comprimido"]):
            denoise = 0.6
            summary_actions.append("Supresión de Ruido ISO y Compresión")

        # 5. Face restoration
        if any(w in p for w in ["rostro", "cara", "facial", "ojos", "piel"]):
            face_restore = True
            face_fidelity = 0.85
            summary_actions.append("Restauración y Reconstrucción Facial de Alta Fidelidad")

        # 6. Face swap
        if any(w in p for w in ["reemplazar rostro", "cambiar rostro", "cambiar cara", "reemplazar cara", "face swap"]):
            face_swap_active = True
            face_restore = True
            summary_actions.append("Reemplazo Facial (Face Swap)")

        # 7. Object removal
        remove_match = re.search(r"(?:elimina|quitar|borrar|remover|desaparecer)\s+(?:el|la|los|las|un|una)?\s*([a-záéíóúñ]+)", p)
        if remove_match:
            target = remove_match.group(1).strip()
            if target not in ["ruido", "grano", "rostro", "cara"]:
                remove_object_query = target
                summary_actions.append(f"Eliminación Automática de Objeto ('{target}')")

        # 8. Color replacement
        color_match = re.search(r"cambiar\s+color\s+(?:a|por|de)?\s*([a-záéíóúñ]+)", p)
        if color_match:
            c_name = color_match.group(1).strip()
            if c_name in cls.COLOR_MAP:
                replace_color_query = (c_name, cls.COLOR_MAP[c_name])
                summary_actions.append(f"Remapeo Cromático hacia {c_name.capitalize()}")

        # 9. Automated Vision Diagnosis (if prompt asks for "auto" or "mejora todo" or image is underexposed/monochrome)
        if img_bgr is not None:
            metrics = cls.analyze_image_properties(img_bgr)
            if metrics.get("is_monochrome", False) and (colorize or any(w in p for w in ["auto", "restaurar", "todo", "mejor", "foto antigua", "vieja"])):
                colorize = True
                if not any("Colorización IA" in s for s in summary_actions):
                    summary_actions.append("Auto-Detección: Fotografía B&W detectada -> Colorización IA (DDColor)")
            if metrics["is_underexposed"] and not enhance_lighting:
                enhance_lighting = True
                lighting_strength = 0.6
                summary_actions.append("Auto-Corrección: Imagen con baja exposición detectada")
            if metrics["num_faces"] > 0 and not face_restore:
                face_restore = True
                summary_actions.append(f"Auto-Detección: {metrics['num_faces']} rostro(s) identificados")
            if metrics["is_blurry"]:
                denoise = max(denoise, 0.4)
                summary_actions.append("Auto-Detección: Nitidez optimizada para mitigar desenfoque")

        summary = " -> ".join(summary_actions) if summary_actions else "Restauración estándar de super-resolución"

        return NLPPipelineAction(
            intent="custom_text_prompt",
            profile=profile,
            upscale=upscale,
            denoise=denoise,
            enhance_lighting=enhance_lighting,
            lighting_strength=lighting_strength,
            face_restore=face_restore,
            face_fidelity=face_fidelity,
            remove_object_query=remove_object_query,
            replace_color_query=replace_color_query,
            face_swap_active=face_swap_active,
            colorize=colorize,
            summary=summary
        )
