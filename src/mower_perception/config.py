"""Paths and defaults for the perception package."""
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent.parent.parent

DETECTION_WEIGHTS = PACKAGE_ROOT / "models" / "detection_best.pt"
SEGMENTATION_WEIGHTS = PACKAGE_ROOT / "models" / "segmentation_best.pt"

DEFAULT_CONF = 0.4

# BGR, cv2 convention — used to tint segmentation masks by class name so the
# same class always gets the same color across photos (not assigned by
# detection order, which made "lawn" show up a different color each time).
CLASS_COLORS = {
    "lawn": (60, 200, 60),       # green — safe to cut
    "boundary": (50, 50, 220),   # red — no-cut edge
    "barriers": (200, 80, 60),   # blue — obstacle/non-grass
}
DEFAULT_MASK_COLOR = (60, 200, 220)  # yellow — fallback for any unlisted class
