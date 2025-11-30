# backend/pipeline.py

import os
import torch
from io import BytesIO
from typing import Optional
from PIL import Image

from huggingface_hub import InferenceClient
from transformers import CLIPProcessor, CLIPModel


# ---------------------------------------------------------------
# 1. Load HF TOKEN
# ---------------------------------------------------------------
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise RuntimeError("Please set HF_TOKEN environment variable.")


# ---------------------------------------------------------------
# 2. HuggingFace Chat Completion Client
# ---------------------------------------------------------------
client = InferenceClient(api_key=HF_TOKEN)

MODEL_NAME = "HuggingFaceH4/zephyr-7b-beta"


# ---------------------------------------------------------------
# 3. Text Query Function
# ---------------------------------------------------------------
def answer_text_query(question: str) -> str:
    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are an AI assistant. Answer clearly."},
            {"role": "user", "content": question},
        ],
        max_tokens=200,
        temperature=0.5,
    )
    return completion.choices[0].message["content"]


# ---------------------------------------------------------------
# 4. CLIP Image Encoder
# ---------------------------------------------------------------
CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
clip_model = CLIPModel.from_pretrained(CLIP_MODEL_ID)
clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)


# ---------------------------------------------------------------
# 5. Image Query Function
# ---------------------------------------------------------------
def answer_image_query(image_bytes: bytes, question: Optional[str] = None) -> str:
    image = Image.open(BytesIO(image_bytes)).convert("RGB")

    candidate_captions = [
        "a person",
        "a group of people",
        "an indoor scene",
        "an outdoor scene",
        "a landscape",
        "a city street",
        "an animal",
        "a vehicle",
        "a document screenshot",
        "a chart or diagram",
        "an object"
    ]

    inputs = clip_processor(
        text=candidate_captions,
        images=image,
        return_tensors="pt",
        padding=True,
    )

    with torch.no_grad():
        outputs = clip_model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1).squeeze()

    best_caption = candidate_captions[int(probs.argmax().item())]

    if not question or question.strip() == "":
        question = "Describe the image."

    prompt = (
        f"The image appears to show: {best_caption}. "
        f"User question: {question}"
    )

    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are an AI assistant."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=300,
        temperature=0.5,
    )

    return completion.choices[0].message["content"]
