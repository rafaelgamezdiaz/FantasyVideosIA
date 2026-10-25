import cv2
import numpy as np
from typing import Optional, Tuple

class ImageEnhancementOps:
    """
    Precision image operations including lighting enhancement, color remapping,
    object inpainting, and facial identity swapping.
    """

    @staticmethod
    def enhance_lighting_retinex(img_bgr: np.ndarray, strength: float = 0.5) -> np.ndarray:
        """
        Enhances dynamic range and lifts shadows using Adaptive CLAHE and Multi-Scale Gaussian Retinex.
        """
        # Convert to LAB color space
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # Apply CLAHE to L channel
        clip_limit = 2.0 + strength * 2.5
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        l_clahe = clahe.apply(l)

        # Multi-scale Retinex simulation
        l_float = l.astype(np.float32) + 1.0
        sigma_list = [15, 80, 250]
        retinex = np.zeros_like(l_float)
        for s in sigma_list:
            blur = cv2.GaussianBlur(l_float, (0, 0), s) + 1.0
            retinex += np.log10(l_float) - np.log10(blur)
        retinex /= len(sigma_list)

        # Normalize retinex
        retinex = (retinex - np.min(retinex)) / (np.max(retinex) - np.min(retinex) + 1e-5) * 255.0
        l_blended = cv2.addWeighted(l_clahe, 1.0 - strength * 0.5, retinex.astype(np.uint8), strength * 0.5, 0)

        merged_lab = cv2.merge((l_blended, a, b))
        return cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)

    @staticmethod
    def replace_color(img_bgr: np.ndarray, target_bgr: Tuple[int, int, int]) -> np.ndarray:
        """
        Shifts dominant saturation color toward the target color while strictly preserving luminance and texture.
        """
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        # Target HSV
        target_mat = np.uint8([[list(target_bgr)]])
        target_hsv = cv2.cvtColor(target_mat, cv2.COLOR_BGR2HSV)[0][0]
        target_hue = target_hsv[0]

        # Shift hue only where saturation is meaningful (above threshold)
        mask = s > 40
        h[mask] = target_hue
        s[mask] = np.clip(s[mask] * 1.1, 0, 255).astype(np.uint8)

        new_hsv = cv2.merge((h, s, v))
        return cv2.cvtColor(new_hsv, cv2.COLOR_HSV2BGR)

    @staticmethod
    def remove_salient_object(img_bgr: np.ndarray, radius_pct: float = 0.25) -> np.ndarray:
        """
        Removes foreground anomalies or salient objects using edge-aware morphology and inpainting.
        """
        h, w = img_bgr.shape[:2]
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # Create central saliency / contrast gradient mask
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        grad = np.sqrt(sobelx**2 + sobely**2)
        grad = np.uint8(np.clip(grad / np.max(grad) * 255, 0, 255))

        _, thresh = cv2.threshold(grad, 120, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        mask = cv2.dilate(thresh, kernel, iterations=3)

        # Perform seamless Navier-Stokes inpainting
        inpainted = cv2.inpaint(img_bgr, mask, inpaintRadius=5, flags=cv2.INPAINT_NS)
        return inpainted

    @staticmethod
    def swap_faces(dest_img_bgr: np.ndarray, src_face_bgr: np.ndarray) -> np.ndarray:
        """
        Locates faces in dest_img, aligns and transplants identity features from src_face_bgr,
        blending borders with Poisson seamless cloning.
        """
        if not (hasattr(cv2, "CascadeClassifier") and hasattr(cv2, "data") and hasattr(cv2.data, "haarcascades")):
            return dest_img_bgr

        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        try:
            face_cascade = cv2.CascadeClassifier(cascade_path)
            gray_dest = cv2.cvtColor(dest_img_bgr, cv2.COLOR_BGR2GRAY)
            dest_faces = face_cascade.detectMultiScale(gray_dest, 1.15, 4)
            if len(dest_faces) == 0:
                return dest_img_bgr

            gray_src = cv2.cvtColor(src_face_bgr, cv2.COLOR_BGR2GRAY)
            src_faces = face_cascade.detectMultiScale(gray_src, 1.15, 4)
            if len(src_faces) == 0:
                return dest_img_bgr
        except Exception:
            return dest_img_bgr
            return dest_img_bgr

        # Extract primary source face
        sx, sy, sw, sh = src_faces[0]
        src_crop = src_face_bgr[sy:sy+sh, sx:sx+sw]

        result = dest_img_bgr.copy()
        for (dx, dy, dw, dh) in dest_faces:
            # Resize source face to target face dimensions
            resized_src = cv2.resize(src_crop, (dw, dh), interpolation=cv2.INTER_LANCZOS4)

            # Match color histogram of target face
            src_lab = cv2.cvtColor(resized_src, cv2.COLOR_BGR2LAB)
            target_roi = result[dy:dy+dh, dx:dx+dw]
            target_lab = cv2.cvtColor(target_roi, cv2.COLOR_BGR2LAB)

            for i in range(3):
                src_mean, src_std = np.mean(src_lab[:, :, i]), max(np.std(src_lab[:, :, i]), 1e-4)
                tar_mean, tar_std = np.mean(target_lab[:, :, i]), max(np.std(target_lab[:, :, i]), 1e-4)
                src_lab[:, :, i] = np.clip((src_lab[:, :, i] - src_mean) * (tar_std / src_std) + tar_mean, 0, 255).astype(np.uint8)

            color_matched_src = cv2.cvtColor(src_lab, cv2.COLOR_LAB2BGR)

            # Create elliptical blending mask
            mask = np.zeros((dh, dw), dtype=np.uint8)
            center = (dw // 2, dh // 2)
            axes = (int(dw * 0.42), int(dh * 0.48))
            cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
            mask = cv2.GaussianBlur(mask, (15, 15), 7)

            # Seamless clone
            center_global = (dx + dw // 2, dy + dh // 2)
            try:
                result = cv2.seamlessClone(color_matched_src, result, mask, center_global, cv2.NORMAL_CLONE)
            except Exception:
                # Feathered blend fallback
                mask_norm = mask.astype(np.float32) / 255.0
                mask_3d = mask_norm[:, :, np.newaxis]
                result[dy:dy+dh, dx:dx+dw] = np.clip(color_matched_src * mask_3d + target_roi * (1.0 - mask_3d), 0, 255).astype(np.uint8)

        return result

    @staticmethod
    def is_monochromatic_or_grayscale(img_bgr: np.ndarray, threshold: float = 12.0) -> bool:
        """
        Detects whether an image is grayscale, sepia, or monochromatic.
        """
        if img_bgr is None or img_bgr.size == 0:
            return False
        if len(img_bgr.shape) < 3 or img_bgr.shape[2] < 3:
            return True
        b, g, r = cv2.split(img_bgr)
        diff_bg = float(np.mean(np.abs(b.astype(np.float32) - g.astype(np.float32))))
        diff_gr = float(np.mean(np.abs(g.astype(np.float32) - r.astype(np.float32))))
        return (diff_bg + diff_gr) < threshold

    @classmethod
    def colorize_image(
        cls,
        img_bgr: np.ndarray,
        colorizer_session: Optional[any] = None
    ) -> np.ndarray:
        """
        Colorizes monochrome or black-and-white photos using DDColor ONNX model if available,
        or automatic semantic LAB chrominance estimation.
        """
        if img_bgr is None or img_bgr.size == 0:
            return img_bgr

        h, w = img_bgr.shape[:2]
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        if colorizer_session is not None:
            try:
                l_resized = cv2.resize(gray, (512, 512))
                l_norm = l_resized.astype(np.float32) / 255.0
                net_input = np.stack([l_norm, l_norm, l_norm], axis=0)[np.newaxis, ...]
                input_name = colorizer_session.get_inputs()[0].name
                out = colorizer_session.run(None, {input_name: net_input})[0]
                ab = out[0].transpose(1, 2, 0)  # (512, 512, 2)
                ab_resized = cv2.resize(ab, (w, h), interpolation=cv2.INTER_CUBIC)
                a = np.clip(ab_resized[:, :, 0] + 128.0, 0, 255).astype(np.uint8)
                b = np.clip(ab_resized[:, :, 1] + 128.0, 0, 255).astype(np.uint8)
                lab = cv2.merge([gray, a, b])
                return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            except Exception:
                pass

        # Heuristic chrominance synthesis fallback
        # Generates subtle warm natural tones (sepia to realistic skin/environment)
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        # Apply skin-tone warm bias to mid-tones
        midtones = np.exp(-((l.astype(np.float32) - 140.0) ** 2) / (2.0 * (50.0 ** 2)))
        a_synth = np.clip(128.0 + midtones * 16.0, 0, 255).astype(np.uint8)
        b_synth = np.clip(128.0 + midtones * 24.0, 0, 255).astype(np.uint8)
        synth_lab = cv2.merge([l, a_synth, b_synth])
        return cv2.cvtColor(synth_lab, cv2.COLOR_LAB2BGR)

