import numpy as np

from mower_perception.detector import BoxDetection, PredictionResult, SegmentationMask
from mower_perception.visualize import draw_boxes, format_detections_table, overlay_masks, visualize


def _blank_image(h=100, w=100):
    return np.zeros((h, w, 3), dtype=np.uint8)


def test_draw_boxes_returns_same_shape():
    image = _blank_image()
    result = PredictionResult(boxes=[BoxDetection("person", 0.91, (10, 10, 50, 50))])
    annotated = draw_boxes(image, result)
    assert annotated.shape == image.shape
    assert not np.array_equal(annotated, image)  # something was drawn


def test_overlay_masks_draws_outline():
    image = _blank_image()
    mask = np.zeros((100, 100), dtype=bool)
    mask[20:80, 20:80] = True
    result = PredictionResult(masks=[SegmentationMask("grass", 0.8, mask)])
    annotated = overlay_masks(image, result)
    assert annotated.shape == image.shape
    assert annotated[20, 50].sum() > 0  # mask border got outlined
    assert annotated[50, 50].sum() == 0  # mask interior untouched (outline only)
    assert annotated[5, 5].sum() == 0  # outside mask untouched


def test_visualize_combines_boxes_and_masks():
    image = _blank_image()
    mask = np.ones((100, 100), dtype=bool)
    result = PredictionResult(
        boxes=[BoxDetection("dog", 0.7, (5, 5, 40, 40))],
        masks=[SegmentationMask("grass", 0.9, mask)],
    )
    annotated = visualize(image, result)
    assert annotated.shape == image.shape


def test_format_detections_table_empty():
    assert format_detections_table(PredictionResult()) == "No detections above threshold."


def test_format_detections_table_lists_entries():
    result = PredictionResult(boxes=[BoxDetection("cat", 0.55, (0, 0, 1, 1))])
    table = format_detections_table(result)
    assert "cat 0.55" in table
