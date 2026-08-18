# Lawn Mower Perception 🌱

Real-time **obstacle detection** and **lawn segmentation** for an autonomous lawn mower,
built with **YOLOv8**. This is the AI-perception module of my Mechatronics capstone
(MSE 4499) — extracted here as a clean, standalone, tested Python package with an
interactive demo.

> The full capstone was a boundary-free autonomous mower (perception + SLAM + planning +
> safety, on ROS / Jetson Nano). **This repo focuses on the perception side** — the computer
> vision that decides where it's safe to mow and what to avoid.

## What it does
- **Detection** — locates safety-relevant obstacles (people, pets, objects) in the mower's
  camera view so the system can avoid them.
- **Segmentation** — classifies *cuttable lawn* vs *boundaries / non-grass*, used for
  "no-cut" reasoning so the blade only runs on valid grass.

## Results
| Task | Model | Metric | Score |
|------|-------|--------|-------|
| Obstacle detection | YOLOv8 | box mAP@0.5 | **0.764** |
| Lawn segmentation | YOLOv8-seg | mask mAP@0.5 | **0.745** |

*Trained on a custom, hand-labeled lawn dataset; tuned architecture and hyperparameters.*

![Sample prediction](examples/val_batch0_pred.jpg)

## Demo
An interactive **Gradio** app — upload an image and see detections and lawn segmentation
overlaid with confidence scores.

<!-- Add a screenshot/GIF here after your first run: ![Demo](examples/demo.png) -->

```bash
python app.py
```

Try it with the sample yard photos in `examples/inputs/`, or upload your own.

## Simulation demo

The full capstone (perception + SLAM + planning + safety) was validated end-to-end
in Gazebo/RViz.

> 🎥 [Simulation walkthrough video](#) — link coming soon

## Installation
```bash
git clone https://github.com/malakalhanafi02/lawn-mower-perception.git
cd lawn-mower-perception
pip install -r requirements.txt
```

## Usage (as a library)
```python
from mower_perception.detector import Detector

det = Detector("models/detection_best.pt", conf=0.4)
results = det.predict("examples/yard.jpg")   # returns structured detections
```

## How it works
YOLOv8 ("You Only Look Once") is a single-pass CNN that predicts all boxes/masks in one
forward pass, making it fast enough for real-time use on edge hardware (the capstone ran it
on a Jetson Nano). Detections are used **conservatively** — treated as cautious
"avoid / no-cut" hints rather than exact boundaries — because in a safety-critical system
it's better to be slightly over-cautious than to trust an imperfect mask.

## Project structure
```
src/mower_perception/   # detector.py, visualize.py, config.py
tests/                  # pytest unit tests
app.py                  # Gradio demo
models/                 # trained weights (detection_best.pt, segmentation_best.pt)
examples/               # sample images + result figures
training/               # training notebooks (Colab, reference)
docs/ARCHITECTURE.md    # how this fits into the full robot (SLAM, planning, safety)
```

## Full system

This repo is the standalone perception module. It was built as part of a larger
boundary-free autonomous mower — SLAM-based mapping, sensor-fused localization,
coverage path planning, and a layered safety system — running on ROS 1 Noetic on a
Jetson Nano. See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the full pipeline.

## Context & credits
AI-perception component of a 5-person MSE 4499 Mechatronics capstone, which earned an
Honourable Mention at Western Engineering Design Day. Detection and segmentation models
trained and tuned by me.

## License
MIT
