"""Gradio demo for the lawn mower perception models.

Upload a yard photo and see YOLOv8 lawn segmentation (cuttable lawn vs.
boundary vs. barriers) overlaid on it, with a confidence-threshold slider.
Run: python app.py  (set share=True below for a temporary public link)
"""
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image

from mower_perception import Detector, format_detections_table, legend_markdown, visualize

EXAMPLES_DIR = Path(__file__).parent / "examples" / "inputs"

_detector = Detector(detection_weights=None)  # segmentation only


def run(image: Image.Image, conf: float):
    if image is None:
        return None, "Upload an image to get started."
    image_bgr = np.array(image.convert("RGB"))[:, :, ::-1]
    result = _detector.predict(image_bgr, mode="segmentation", conf=conf)
    annotated_bgr = visualize(image_bgr, result)
    annotated_rgb = annotated_bgr[:, :, ::-1]
    return annotated_rgb, format_detections_table(result)


theme = gr.themes.Soft(primary_hue="green", secondary_hue="emerald", neutral_hue="slate")

with gr.Blocks(title="Lawn Mower Perception") as demo:
    gr.Markdown(
        "# 🌱 Lawn Mower Perception\n"
        "YOLOv8 lawn segmentation from an autonomous mower capstone — classifies "
        "each pixel as cuttable lawn, boundary, or barrier. Upload a yard photo "
        "and adjust the confidence threshold."
    )
    with gr.Accordion("What am I looking at?", open=False):
        gr.Markdown(legend_markdown())
    with gr.Row():
        with gr.Column():
            image_in = gr.Image(type="pil", label="Yard photo")
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

    run_btn.click(run, inputs=[image_in, conf_in], outputs=[image_out, table_out])
    image_in.change(run, inputs=[image_in, conf_in], outputs=[image_out, table_out])

if __name__ == "__main__":
    demo.launch(share=False, theme=theme)
