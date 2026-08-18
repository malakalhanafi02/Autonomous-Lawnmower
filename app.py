"""Gradio demo for the lawn mower perception models.

Upload a yard photo and see YOLOv8 obstacle detection and/or lawn
segmentation overlaid on it, with a confidence-threshold slider.
Run: python app.py  (set share=True below for a temporary public link)
"""
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image

from mower_perception import Detector, format_detections_table, legend_markdown, visualize

EXAMPLES_DIR = Path(__file__).parent / "examples" / "inputs"

_detector = Detector()  # loads both models once at startup


def run(image: Image.Image, mode: str, conf: float):
    if image is None:
        return None, "Upload an image to get started."
    image_bgr = np.array(image.convert("RGB"))[:, :, ::-1]
    mode_key = mode.lower()
    result = _detector.predict(image_bgr, mode=mode_key, conf=conf)
    annotated_bgr = visualize(image_bgr, result)
    annotated_rgb = annotated_bgr[:, :, ::-1]
    return annotated_rgb, format_detections_table(result)


with gr.Blocks(title="Lawn Mower Perception") as demo:
    gr.Markdown(
        "# Lawn Mower Perception 🌱\n"
        "YOLOv8 obstacle detection + lawn segmentation from an autonomous "
        "mower capstone. Upload a yard photo, pick a mode, and adjust the "
        "confidence threshold. Trained on a small custom-labeled dataset — a "
        "random backyard photo may need a lower confidence threshold than a "
        "photo similar to the training set."
    )
    with gr.Accordion("What am I looking at?", open=False):
        gr.Markdown(legend_markdown())
    with gr.Row():
        with gr.Column():
            image_in = gr.Image(type="pil", label="Yard photo")
            mode_in = gr.Radio(
                ["Detection", "Segmentation", "Both"], value="Both", label="Mode"
            )
            conf_in = gr.Slider(0.0, 1.0, value=0.4, step=0.05, label="Confidence threshold")
            run_btn = gr.Button("Run", variant="primary")
            if EXAMPLES_DIR.exists():
                gr.Examples(
                    examples=sorted(str(p) for p in EXAMPLES_DIR.glob("*.jpg")),
                    inputs=image_in,
                )
        with gr.Column():
            image_out = gr.Image(label="Annotated result")
            table_out = gr.Textbox(label="Detections", lines=6)

    run_btn.click(run, inputs=[image_in, mode_in, conf_in], outputs=[image_out, table_out])
    image_in.change(run, inputs=[image_in, mode_in, conf_in], outputs=[image_out, table_out])

if __name__ == "__main__":
    demo.launch(share=False)
