from pathlib import Path

import pytest

from mower_perception.detector import BoxDetection, Detector, PredictionResult, SegmentationMask
from mower_perception.config import DETECTION_WEIGHTS, SEGMENTATION_WEIGHTS

EXAMPLE_IMAGE = Path(__file__).parent.parent / "examples" / "inputs" / "yard_1.jpg"

weights_available = DETECTION_WEIGHTS.exists() and SEGMENTATION_WEIGHTS.exists()


def test_box_detection_dataclass():
    d = BoxDetection(class_name="person", confidence=0.91, xyxy=(1.0, 2.0, 3.0, 4.0))
    assert d.class_name == "person"
    assert d.confidence > 0.5


def test_segmentation_mask_dataclass():
    import numpy as np

    m = SegmentationMask(class_name="grass", confidence=0.8, mask=np.ones((4, 4), dtype=bool))
    assert m.mask.shape == (4, 4)


def test_prediction_result_defaults_empty():
    result = PredictionResult()
    assert result.boxes == []
    assert result.masks == []


@pytest.mark.skipif(not weights_available, reason="model weights not present")
def test_detector_predict_smoke():
    detector = Detector(conf=0.25)
    result = detector.predict(str(EXAMPLE_IMAGE), mode="both")
    assert isinstance(result, PredictionResult)
    assert isinstance(result.boxes, list)
    assert isinstance(result.masks, list)
