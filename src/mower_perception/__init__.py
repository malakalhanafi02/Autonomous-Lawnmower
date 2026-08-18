"""Lawn Mower Perception — YOLOv8 obstacle detection and lawn segmentation.

AI-perception module of the MSE 4499 autonomous lawn mower capstone.
"""
from .detector import BoxDetection, Detector, PredictionResult, SegmentationMask
from .visualize import draw_boxes, format_detections_table, legend_markdown, overlay_masks, visualize

__version__ = "0.1.0"

__all__ = [
    "BoxDetection",
    "Detector",
    "PredictionResult",
    "SegmentationMask",
    "draw_boxes",
    "format_detections_table",
    "legend_markdown",
    "overlay_masks",
    "visualize",
]
