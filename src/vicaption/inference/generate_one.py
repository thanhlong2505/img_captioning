from __future__ import annotations

from pathlib import Path

from PIL import Image


def _move_mapping_to_device(mapping: dict, device):
    moved = {}
    for key, value in mapping.items():
        moved[key] = value.to(device) if hasattr(value, "to") else value
    return moved


def generate_caption(
    image_path: str,
    model,
    processor,
    tokenizer,
    prompt: str,
    generation_config: dict,
    device,
) -> str:
    """Generate one Vietnamese caption string for an image."""
    image = Image.open(Path(image_path)).convert("RGB")
    pixel_values = processor(images=[image], return_tensors="pt")["pixel_values"].to(device)
    prompt_batch = tokenizer(
        [prompt],
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    prompt_batch = _move_mapping_to_device(prompt_batch, device)

    sequences = model.generate(
        pixel_values=pixel_values,
        prompt_input_ids=prompt_batch["input_ids"],
        prompt_attention_mask=prompt_batch["attention_mask"],
        generation_kwargs=generation_config,
    )

    first = sequences[0] if hasattr(sequences, "__getitem__") else sequences
    text = tokenizer.decode(first, skip_special_tokens=True)
    return text.strip()

