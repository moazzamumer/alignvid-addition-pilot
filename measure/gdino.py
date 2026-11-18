"""Grounding DINO (HF transformers) wrapper shared by dev_thresholds.py and detect.py."""
import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

import os
MODEL_ID = os.environ.get("GDINO_MODEL", "IDEA-Research/grounding-dino-tiny")  # HF id or local folder


class GDino:
    def __init__(self, model_id=MODEL_ID, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(self.device).eval()
        self.model_id = model_id

    @torch.no_grad()
    def detect(self, image: Image.Image, query: str, box_threshold: float, text_threshold: float):
        """Return list of {box:[x0,y0,x1,y1] in pixels, score, label} for one text query.
        Query must be lower-case and end with a period, per the model card."""
        text = query.strip().lower().rstrip(".") + "."
        inputs = self.processor(images=image, text=text, return_tensors="pt").to(self.device)
        out = self.model(**inputs)
        res = self.processor.post_process_grounded_object_detection(
            out, inputs.input_ids, threshold=box_threshold, text_threshold=text_threshold,
            target_sizes=[image.size[::-1]])[0]
        labels = res.get("text_labels", res.get("labels"))
        return [{"box": [round(float(v), 1) for v in b], "score": round(float(s), 3), "label": str(l)}
                for b, s, l in zip(res["boxes"], res["scores"], labels)]
