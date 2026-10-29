"""Face annotations drawn with QPainter (fixed infrastructure, do not edit).

All sizes are derived from the image size, so boxes, landmarks and text stay
readable for a 640x480 webcam as well as for a 4K photo. Text is drawn with Qt,
so names in any script (e.g. Persian) render correctly.
"""

from dataclasses import dataclass

from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QColor, QFont, QFontMetricsF, QPainter, QPen, QPolygonF

from config import (
    COLOR_LANDMARK,
    COLOR_MATCH,
    COLOR_PENDING,
    COLOR_UNKNOWN,
    UNKNOWN_LABEL,
)

STATUS_NO_MODEL = "no_model"
STATUS_PENDING = "pending"
STATUS_READY = "ready"


@dataclass
class FaceOverlay:
    """Everything needed to draw one face (image coordinates)."""
    bbox: tuple                 # (x1, y1, x2, y2)
    landmarks: QPolygonF        # 478 points
    status: str                 # STATUS_*
    name: str = ""              # raw SVM prediction
    score: float = 0.0          # raw cosine similarity


def landmarks_to_polygon(points):
    polygon = QPolygonF()
    for x, y in points:
        polygon.append(QPointF(float(x), float(y)))
    return polygon


def describe(overlay, threshold):
    """(label text, colour) of an overlay for the current threshold."""
    if overlay.status == STATUS_NO_MODEL:
        return "No model", COLOR_PENDING
    if overlay.status == STATUS_PENDING:
        return "Analyzing...", COLOR_PENDING
    if overlay.score >= threshold and overlay.name:
        return f"{overlay.name}  {overlay.score:.2f}", COLOR_MATCH
    return f"{UNKNOWN_LABEL}  {overlay.score:.2f}", COLOR_UNKNOWN


def paint_overlays(painter, overlays, image_w, image_h, threshold, show_landmarks):
    """Draw all overlays. `painter` must already map image coordinates to the widget."""
    if not overlays:
        return
    side = float(max(image_w, image_h))
    line_w = max(2.0, side * 0.003)
    dot_w = max(1.5, side * 0.0030)
    font_px = max(14, int(side * 0.028))
    pad = font_px * 0.3

    font = QFont()
    font.setPixelSize(font_px)
    font.setBold(True)
    painter.setFont(font)
    metrics = QFontMetricsF(font)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

    for overlay in overlays:
        text, color = describe(overlay, threshold)
        qcolor = QColor(*color)
        x1, y1, x2, y2 = overlay.bbox

        if show_landmarks and overlay.landmarks is not None:
            pen = QPen(QColor(*COLOR_LANDMARK))
            pen.setWidthF(dot_w)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(pen)
            painter.drawPoints(overlay.landmarks)

        pen = QPen(qcolor)
        pen.setWidthF(line_w)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(QRectF(x1, y1, x2 - x1, y2 - y1))

        # Label bar above the box (below the top edge if there is no room).
        text_w = metrics.horizontalAdvance(text)
        bar_w = text_w + 2 * pad
        bar_h = metrics.height() + pad
        bar_x = min(max(0.0, x1), max(0.0, image_w - bar_w))
        bar_y = y1 - bar_h - line_w * 0.5
        if bar_y < 0:
            bar_y = min(y1 + line_w, image_h - bar_h)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(qcolor)
        painter.drawRect(QRectF(bar_x, bar_y, bar_w, bar_h))
        painter.setPen(QColor(15, 20, 24))
        painter.drawText(QPointF(bar_x + pad, bar_y + pad * 0.5 + metrics.ascent()), text)
