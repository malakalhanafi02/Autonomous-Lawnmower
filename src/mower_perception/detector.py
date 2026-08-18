"""Unified YOLOv8 detector/segmenter for the mower perception stack."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

from .config import DEFAULT_CONF, DETECTION_WEIGHTS, SEGMENTATION_WEIGHTS

try:
    import numpy as np
except ImportError:  # pragma: no cover - numpy is a hard dependency at runtime
    np = None


@dataclass
class BoxDetection:
    class_name: str
    confidence: float
    xyxy: tuple[float, float, float, float]


@dataclass
class SegmentationMask:
    class_name: str
    confidence: float
    mask: "np.ndarray"  # HxW boolean array


@dataclass
class PredictionResult:
    boxes: list[BoxDetection] = field(default_factory=list)
    masks: list[SegmentationMask] = field(default_factory=list)


class Detector:
    """Loads YOLOv8 detection + segmentation weights once and runs inference on demand."""

    def __init__(
        self,
        detection_weights: Union[str, Path] = DETECTION_WEIGHTS,
        segmentation_weights: Union[str, Path] = SEGMENTATION_WEIGHTS,
        conf: float = DEFAULT_CONF,
    ) -> None:
        from ultralytics import YOLO

        self.conf = conf
        self._detect_model = YOLO(str(detection_weights))
        self._segment_model = YOLO(str(segmentation_weights))

    def predict(self, image, mode: str = "both", conf: Optional[float] = None) -> PredictionResult:
        """Run detection and/or segmentation on `image` (path, ndarray, or PIL.Image)."""
        conf = self.conf if conf is None else conf
        result = PredictionResult()
        if mode in ("detection", "both"):
            result.boxes = self._run_detection(image, conf)
        if mode in ("segmentation", "both"):
            result.masks = self._run_segmentation(image, conf)
        return result

    def _run_detection(self, image, conf: float) -> list[BoxDetection]:
        outputs = self._detect_model.predict(source=image, conf=conf, verbose=False)
        if not outputs or outputs[0].boxes is None:
            return []
        res = outputs[0]
        boxes = res.boxes
        detections = []
        for xyxy, cls_id, confidence in zip(
            boxes.xyxy.cpu().numpy(),
            boxes.cls.cpu().numpy().astype(int),
            boxes.conf.cpu().numpy(),
        ):
            detections.append(
                BoxDetection(
                    class_name=res.names[int(cls_id)],
                    confidence=float(confidence),
                    xyxy=tuple(float(v) for v in xyxy),
                )
            )
        return detections

    def _run_segmentation(self, image, conf: float) -> list[SegmentationMask]:
        outputs = self._segment_model.predict(source=image, conf=conf, verbose=False)
        if not outputs or outputs[0].masks is None:
            return []
        res = outputs[0]
        masks_data = res.masks.data.cpu().numpy()  # (N, H, W), values 0/1
        cls_ids = res.boxes.cls.cpu().numpy().astype(int)
        confs = res.boxes.conf.cpu().numpy()
        segments = []
        for mask, cls_id, confidence in zip(masks_data, cls_ids, confs):
            segments.append(
                SegmentationMask(
                    class_name=res.names[int(cls_id)],
                    confidence=float(confidence),
                    mask=mask.astype(bool),
                )
            )
        return segments
