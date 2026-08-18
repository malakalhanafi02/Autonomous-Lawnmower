"""Paths and defaults for the perception package."""
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent.parent.parent

DETECTION_WEIGHTS = PACKAGE_ROOT / "models" / "detection_best.pt"
SEGMENTATION_WEIGHTS = PACKAGE_ROOT / "models" / "segmentation_best.pt"

DEFAULT_CONF = 0.4

# BGR, cv2 convention — used to tint segmentation masks per class id.
MASK_COLORS = [
    (60, 200, 60),
    (60, 60, 220),
    (220, 160, 60),
    (200, 60, 200),
    (60, 200, 220),
]
