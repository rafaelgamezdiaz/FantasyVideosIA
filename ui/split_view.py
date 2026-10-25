import numpy as np
import cv2
from PySide6.QtWidgets import QWidget
from PySide6.QtCore import Qt, QRect, QPoint
from PySide6.QtGui import QPainter, QImage, QPixmap, QColor, QPen, QBrush, QFont

class SplitComparisonView(QWidget):
    """
    Interactive Before / After dual-view widget with draggable split slider.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.slider_pos = 0.5  # 0.0 to 1.0
        self.is_dragging = False

        self.pixmap_before: QPixmap = QPixmap()
        self.pixmap_after: QPixmap = QPixmap()
        self.setMinimumSize(400, 300)

    def set_images_from_numpy(self, before_bgr: np.ndarray, after_bgr: np.ndarray):
        """
        Accepts OpenCV BGR arrays and prepares synchronized QPixmaps.
        """
        if before_bgr is not None and before_bgr.size > 0:
            h_b, w_b, c_b = before_bgr.shape
            rgb_b = cv2.cvtColor(before_bgr, cv2.COLOR_BGR2RGB)
            q_img_b = QImage(rgb_b.data, w_b, h_b, c_b * w_b, QImage.Format.Format_RGB888)
            self.pixmap_before = QPixmap.fromImage(q_img_b)

        if after_bgr is not None and after_bgr.size > 0:
            h_a, w_a, c_a = after_bgr.shape
            rgb_a = cv2.cvtColor(after_bgr, cv2.COLOR_BGR2RGB)
            q_img_a = QImage(rgb_a.data, w_a, h_a, c_a * w_a, QImage.Format.Format_RGB888)
            self.pixmap_after = QPixmap.fromImage(q_img_a)

        self.update()

    def set_single_preview(self, img_bgr: np.ndarray):
        """
        Sets a single image for preview before processing starts.
        """
        if img_bgr is not None and img_bgr.size > 0:
            h, w, c = img_bgr.shape
            rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            q_img = QImage(rgb.data, w, h, c * w, QImage.Format.Format_RGB888)
            self.pixmap_before = QPixmap.fromImage(q_img)
            self.pixmap_after = QPixmap()  # Clear after
        self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            slider_pixel_x = int(self.width() * self.slider_pos)
            if abs(event.position().x() - slider_pixel_x) < 25:
                self.is_dragging = True
            else:
                self.slider_pos = max(0.0, min(1.0, event.position().x() / self.width()))
                self.update()

    def mouseMoveEvent(self, event):
        if self.is_dragging:
            self.slider_pos = max(0.0, min(1.0, event.position().x() / self.width()))
            self.update()
        else:
            slider_pixel_x = int(self.width() * self.slider_pos)
            if abs(event.position().x() - slider_pixel_x) < 25:
                self.setCursor(Qt.CursorShape.SplitHCursor)
            else:
                self.setCursor(Qt.CursorShape.ArrowCursor)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.is_dragging = False

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#090D12"))

        w = self.width()
        h = self.height()

        if self.pixmap_before.isNull() and self.pixmap_after.isNull():
            # Placeholder text
            painter.setPen(QColor("#484F58"))
            font = QFont("Segoe UI", 12, QFont.Weight.DemiBold)
            painter.setFont(font)
            painter.drawText(
                self.rect(),
                Qt.AlignmentFlag.AlignCenter,
                "Arrastre y suelte una imagen o video aquí\nFormatos soportados: PNG, JPG, WEBP, TIFF, MP4, MKV, AVI"
            )
            return

        # Single image mode
        if not self.pixmap_before.isNull() and self.pixmap_after.isNull():
            scaled = self.pixmap_before.scaled(w, h, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            x_offset = (w - scaled.width()) // 2
            y_offset = (h - scaled.height()) // 2
            painter.drawPixmap(x_offset, y_offset, scaled)
            self._draw_badge(painter, 16, 16, "ORIGINAL", QColor(33, 38, 45, 200), QColor("#58A6FF"))
            return

        # Dual split mode
        split_x = int(w * self.slider_pos)

        # Scale both images to fit widget area identically
        scaled_before = self.pixmap_before.scaled(w, h, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        scaled_after = self.pixmap_after.scaled(w, h, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)

        target_w = max(scaled_before.width(), scaled_after.width())
        target_h = max(scaled_before.height(), scaled_after.height())
        x_offset = (w - target_w) // 2
        y_offset = (h - target_h) // 2

        # 1. Draw Before (Left side clip)
        painter.save()
        painter.setClipRect(0, 0, split_x, h)
        painter.drawPixmap(x_offset, y_offset, target_w, target_h, scaled_before)
        painter.restore()

        # 2. Draw After (Right side clip)
        painter.save()
        painter.setClipRect(split_x, 0, w - split_x, h)
        painter.drawPixmap(x_offset, y_offset, target_w, target_h, scaled_after)
        painter.restore()

        # 3. Draw Slider Separator Line
        pen = QPen(QColor("#58A6FF"), 2)
        painter.setPen(pen)
        painter.drawLine(split_x, 0, split_x, h)

        # 4. Draw Center Circular Handle
        handle_center = QPoint(split_x, h // 2)
        painter.setBrush(QBrush(QColor("#1F6FEB")))
        painter.setPen(QPen(QColor("#FFFFFF"), 2))
        painter.drawEllipse(handle_center, 14, 14)

        # Handle inner arrows
        painter.setPen(QPen(QColor("#FFFFFF"), 1))
        painter.drawLine(split_x - 5, h // 2, split_x - 2, h // 2 - 3)
        painter.drawLine(split_x - 5, h // 2, split_x - 2, h // 2 + 3)
        painter.drawLine(split_x + 5, h // 2, split_x + 2, h // 2 - 3)
        painter.drawLine(split_x + 5, h // 2, split_x + 2, h // 2 + 3)

        # 5. Draw Badges
        self._draw_badge(painter, 16, 16, "ANTES", QColor(33, 38, 45, 200), QColor("#8B949E"))
        self._draw_badge(painter, w - 100, 16, "DESPUÉS", QColor(35, 134, 54, 200), QColor("#7EE787"))

    def _draw_badge(self, painter: QPainter, x: int, y: int, text: str, bg_color: QColor, text_color: QColor):
        painter.save()
        font = QFont("Segoe UI", 9, QFont.Weight.Bold)
        painter.setFont(font)
        rect = QRect(x, y, 84, 24)
        painter.setBrush(QBrush(bg_color))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(rect, 4, 4)
        painter.setPen(QPen(text_color))
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, text)
        painter.restore()
