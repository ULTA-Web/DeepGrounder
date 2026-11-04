"""Client for the external semantic-reward scorer used by GRPO."""
import base64
from io import BytesIO
import math
import os
import re

from PIL import Image
import requests
from torch import nn


class CLIPFunction(nn.Module):
    def __init__(self, url=None):
        super().__init__()
        url = url or os.environ.get("DEEPGROUNDER_SEMANTIC_URL")
        if not url:
            raise ValueError(
                "Set DEEPGROUNDER_SEMANTIC_URL to your Git-RSCLIP service base URL, "
                "or use --reward_funcs accuracy format for the two-reward ablation."
            )
        self.url = f"{url.rstrip('/')}/score"
        self.answer_tag_pattern = r"<answer>(.*?)</answer>"
        self.bbox_pattern = r"\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)]"

    def _preprocess(self, completion, image_path, caption):
        content = completion[0]["content"]
        answer = re.search(self.answer_tag_pattern, content, re.DOTALL)
        box = re.search(self.bbox_pattern, answer.group(1)) if answer else None
        if box is None:
            return None
        bbox = list(map(int, box.groups()))
        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            return None
        # Missing/corrupt source images are configuration errors, not model misses.
        with Image.open(image_path) as source:
            crop = source.convert("RGB").crop(bbox).resize((256, 256))
        buffered = BytesIO()
        crop.save(buffered, format="JPEG")
        return {"image_base64": base64.b64encode(buffered.getvalue()).decode("ascii"),
                "normal_caption": caption}

    def __call__(self, completions, image_paths, normal_captions):
        if len(completions) != len(image_paths) or len(completions) != len(normal_captions):
            raise ValueError("Semantic reward inputs must have equal lengths")
        inputs, indices = [], []
        rewards = [0.0] * len(completions)
        for index, (completion, image, caption) in enumerate(zip(completions, image_paths, normal_captions)):
            payload = self._preprocess(completion, image, caption)
            if payload is not None:
                indices.append(index)
                inputs.append(payload)
        if not inputs:
            return rewards

        # A broken service must not silently turn full training into an ablation.
        response = requests.post(self.url, json={"inputs": inputs}, timeout=30)
        response.raise_for_status()
        scores = response.json()["rewards"]
        if len(scores) != len(inputs):
            raise ValueError("Semantic scorer returned the wrong number of rewards")
        for index, value in zip(indices, scores):
            score = float(value)
            if not math.isfinite(score):
                raise ValueError("Semantic scorer returned a non-finite reward")
            rewards[index] = score
        return rewards
