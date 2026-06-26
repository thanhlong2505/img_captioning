from __future__ import annotations

from PIL import Image


class CaptionCollator:
    """Load images and tokenize prompt/captions for captioner training."""

    def __init__(
        self,
        siglip_processor,
        qwen_tokenizer,
        prompt: str,
        max_prompt_length: int,
        max_caption_length: int,
    ):
        self.siglip_processor = siglip_processor
        self.qwen_tokenizer = qwen_tokenizer
        self.prompt = prompt
        self.max_prompt_length = max_prompt_length
        self.max_caption_length = max_caption_length

    def __call__(self, batch: list[dict[str, str]]) -> dict:
        images = [Image.open(item["image_path"]).convert("RGB") for item in batch]
        image_ids = [item["image_id"] for item in batch]
        captions = [item["caption"] for item in batch]

        image_batch = self.siglip_processor(images=images, return_tensors="pt")
        prompt_batch = self.qwen_tokenizer(
            [self.prompt] * len(batch),
            padding=True,
            truncation=True,
            max_length=self.max_prompt_length,
            return_tensors="pt",
        )
        caption_batch = self.qwen_tokenizer(
            captions,
            padding=True,
            truncation=True,
            max_length=self.max_caption_length,
            return_tensors="pt",
        )

        return {
            "pixel_values": image_batch["pixel_values"],
            "prompt_input_ids": prompt_batch["input_ids"],
            "prompt_attention_mask": prompt_batch["attention_mask"],
            "caption_input_ids": caption_batch["input_ids"],
            "caption_attention_mask": caption_batch["attention_mask"],
            "image_ids": image_ids,
            "captions": captions,
        }

