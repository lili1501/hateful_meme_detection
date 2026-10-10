"""Hugging Face helpers shared by training and scoring. Imports torch/transformers lazily. UNTESTED here (no GPU stack)."""
from __future__ import annotations

from PIL import Image

from memes.prompt import NO, YES, build_messages

DEFAULT_MODEL = "HuggingFaceTB/SmolVLM-Instruct"     # Idefics3 architecture, Apache-2.0, Hugging Face (US/France)


def load_processor(model_id: str, image_size: int = 384, image_splitting: bool = False):
    """Smaller images / no tiling keep memory and latency low. The SAME settings must be used at train and inference
    time (predict_vllm.py passes them through `mm_processor_kwargs`)."""
    from transformers import AutoProcessor
    return AutoProcessor.from_pretrained(model_id, do_image_splitting=image_splitting,
                                         size={"longest_edge": image_size})


def open_image(path: str) -> Image.Image:
    with Image.open(path) as im:
        return im.convert("RGB").copy()


def prompt_text(processor, text: str) -> str:
    """Chat-templated prompt ending where the model should answer."""
    return processor.apply_chat_template(build_messages(text), add_generation_prompt=True)


def yes_no_token_ids(tokenizer) -> tuple[list[int], list[int]]:
    """All single-token spellings of Yes / No that this tokenizer has."""
    def ids(word):
        out = set()
        for v in (word, " " + word, word.lower(), " " + word.lower()):
            t = tokenizer.encode(v, add_special_tokens=False)
            if len(t) == 1:
                out.add(t[0])
        return sorted(out)
    y, n = ids(YES), ids(NO)
    if not y or not n:
        raise ValueError("tokenizer has no single-token Yes/No")
    return y, n
