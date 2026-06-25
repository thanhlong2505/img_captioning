#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


def load_model(model_name):
    print(f"Loading model: {model_name}")

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )

    processor = AutoProcessor.from_pretrained(model_name)

    return model, processor


def generate_caption(
    image_path,
    model,
    processor,
    prompt
):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": str(image_path)
                },
                {
                    "type": "text",
                    "text": prompt
                }
            ]
        }
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    )

    inputs = inputs.to(model.device)

    generated_ids = model.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=False
    )

    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    )[0]

    return output_text.strip()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--images",
        required=True,
        help="Folder chứa ảnh"
    )

    parser.add_argument(
        "--out",
        default="captions.jsonl"
    )

    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-VL-3B-Instruct"
    )

    parser.add_argument(
        "--max-images",
        type=int,
        default=None
    )

    parser.add_argument(
        "--prompt",
        default=(
            "Describe this image in Vietnamese. "
            "Write one concise caption suitable for image captioning dataset."
        )
    )

    args = parser.parse_args()

    model, processor = load_model(args.model)

    image_dir = Path(args.images)

    image_files = []

    for ext in ["*.jpg", "*.jpeg", "*.png", "*.webp"]:
        image_files.extend(image_dir.rglob(ext))

    image_files = sorted(image_files)

    if args.max_images:
        image_files = image_files[:args.max_images]

    print(f"Found {len(image_files)} images")

    with open(args.out, "w", encoding="utf-8") as f:
        for idx, image_path in enumerate(image_files, start=1):

            try:
                caption = generate_caption(
                    image_path,
                    model,
                    processor,
                    args.prompt
                )

                record = {
                    "image": str(image_path),
                    "caption": caption
                }

                f.write(
                    json.dumps(
                        record,
                        ensure_ascii=False
                    ) + "\n"
                )

                print(
                    f"[{idx}/{len(image_files)}] "
                    f"{image_path.name}"
                )

                print("Caption:", caption)
                print()

            except Exception as e:
                print(
                    f"ERROR {image_path}: {e}"
                )

    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()