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


def overlay_masks(image: np.ndarray, result: PredictionResult) -> np.ndarray:
    """Draw a colored bounding box + label around each segmented region.

    Labels are stacked to avoid overlapping when two boxes share a corner
    (common for adjacent classes like lawn/boundary).
    """
    annotated = image.copy()
    h, w = annotated.shape[:2]
    label_rects: list[tuple[int, int, int, int]] = []
    for seg in result.masks:
        color = CLASS_COLORS.get(seg.class_name, DEFAULT_MASK_COLOR)
        mask = cv2.resize(seg.mask.astype("uint8"), (w, h), interpolation=cv2.INTER_NEAREST)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        x, y, bw, bh = cv2.boundingRect(np.concatenate(contours))
        cv2.rectangle(annotated, (x, y), (x + bw, y + bh), color, 2)

        label = f"{seg.class_name} {seg.confidence:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        lw, lh = tw + 4, th + 6
        ly = y - lh
        while any(
            ly < ry + rh and ly + lh > ry and x < rx + rw and x + lw > rx
            for rx, ry, rw, rh in label_rects
        ):
            ly += lh
        ly = max(ly, 0)
        cv2.rectangle(annotated, (x, ly), (x + lw, ly + lh), color, -1)
        cv2.putText(
            annotated, label, (x + 2, ly + lh - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA,
        )
        label_rects.append((x, ly, lw, lh))
    return annotated


def visualize(image: np.ndarray, result: PredictionResult) -> np.ndarray:
    """Draw segmentation boxes underneath and detection boxes on top."""
    annotated = overlay_masks(image, result) if result.masks else image.copy()
    if result.boxes:
        annotated = draw_boxes(annotated, result)
    return annotated


def legend_markdown(include_detection: bool = False) -> str:
    """A short color legend for the segmentation overlay, for display above/below the demo image."""
    swatches = {"lawn": "🟩 green", "boundary": "🟥 red", "barriers": "🟦 blue"}
    lines = [f"- **{name}** — {swatch} box" for name, swatch in swatches.items()]
    if include_detection:
        lines.append("- boxed labels (e.g. \"Tree 0.72\") — obstacle/object detections, with confidence score")
    return "\n".join(lines)


def format_detections_table(result: PredictionResult) -> str:
    """Human-readable listing of detections/segments for a Gradio textbox."""
    lines = [f"{b.class_name} {b.confidence:.2f}" for b in result.boxes]
    lines += [f"{s.class_name} (segmentation) {s.confidence:.2f}" for s in result.masks]
    return "\n".join(lines) if lines else "No detections above threshold."
