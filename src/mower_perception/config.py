"""Paths and defaults for the perception package."""
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent.parent.parent

DETECTION_WEIGHTS = PACKAGE_ROOT / "models" / "detection_best.pt"
SEGMENTATION_WEIGHTS = PACKAGE_ROOT / "models" / "segmentation_best.pt"

DEFAULT_CONF = 0.4

# BGR, cv2 convention — used to box segmentation regions by class name so
# the same class always gets the same color across photos (not assigned by
# detection order, which made "lawn" show up a different color each time).
CLASS_COLORS = {
    "lawn": (80, 175, 76),       # Material green — safe to cut
    "boundary": (53, 57, 229),   # Material red — no-cut edge
    "barriers": (243, 150, 33),  # Material blue — obstacle/non-grass
}
DEFAULT_MASK_COLOR = (54, 191, 255)  # Material amber — fallback for any unlisted class
