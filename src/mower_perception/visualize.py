"""Draw detection boxes and segmentation masks onto an image."""
import cv2
import numpy as np

from .config import CLASS_COLORS, DEFAULT_MASK_COLOR
from .detector import PredictionResult


def draw_boxes(image: np.ndarray, result: PredictionResult) -> np.ndarray:
    annotated = image.copy()
    for box in result.boxes:
        x1, y1, x2, y2 = (int(v) for v in box.xyxy)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (60, 220, 60), 2)
        label = f"{box.class_name} {box.confidence:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(annotated, (x1, max(y1 - th - 6, 0)), (x1 + tw + 4, y1), (60, 220, 60), -1)
        cv2.putText(
            annotated, label, (x1 + 2, max(y1 - 4, th)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA,
        )
    return annotated


def overlay_masks(image: np.ndarray, result: PredictionResult, thickness: int = 3) -> np.ndarray:
    """Draw a clean colored outline around each segmented region (no solid fill)."""
    annotated = image.copy()
    h, w = annotated.shape[:2]
    for seg in result.masks:
        color = CLASS_COLORS.get(seg.class_name, DEFAULT_MASK_COLOR)
        mask = cv2.resize(seg.mask.astype("uint8"), (w, h), interpolation=cv2.INTER_NEAREST)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(annotated, contours, -1, color, thickness, lineType=cv2.LINE_AA)
    return annotated


def visualize(image: np.ndarray, result: PredictionResult) -> np.ndarray:
    """Draw segmentation mask outlines underneath and detection boxes on top."""
    annotated = overlay_masks(image, result) if result.masks else image.copy()
    if result.boxes:
        annotated = draw_boxes(annotated, result)
    return annotated


def legend_markdown(include_detection: bool = False) -> str:
    """A short color legend for the segmentation overlay, for display above/below the demo image."""
    swatches = {"lawn": "🟩 green", "boundary": "🟥 red", "barriers": "🟦 blue"}
    lines = [f"- **{name}** — {swatch} outline" for name, swatch in swatches.items()]
    if include_detection:
        lines.append("- boxed labels (e.g. \"Tree 0.72\") — obstacle/object detections, with confidence score")
    return "\n".join(lines)


def format_detections_table(result: PredictionResult) -> str:
    """Human-readable listing of detections/segments for a Gradio textbox."""
    lines = [f"{b.class_name} {b.confidence:.2f}" for b in result.boxes]
    lines += [f"{s.class_name} (segmentation) {s.confidence:.2f}" for s in result.masks]
    return "\n".join(lines) if lines else "No detections above threshold."
