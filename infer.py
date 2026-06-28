"""
Inference script chạy local để test checkpoint connector.

Chạy từ root project:
    python local_inference.py

Yêu cầu:
    - checkpoint tại checkpoints/connector_last.pt (hoặc connector_best.pt)
    - data/raw/images/ chứa ảnh
    - data/processed/test.json
"""

import sys
import json
import torch
from pathlib import Path
from PIL import Image

sys.path.insert(0, "src")

# ── Cấu hình ──────────────────────────────────────────────────────────────────
CHECKPOINT_PATH   = "checkpoints/connector_best_epoch6.pt"
IMAGE_DIR         = "data/raw/images"
TEST_JSON         = "data/processed/test.json"
NUM_SAMPLES       = 10

VISION_ENCODER    = "google/siglip2-large-patch16-256"
DECODER           = "Qwen/Qwen3-1.7B"
PROMPT            = "Viết đúng một câu chú thích tiếng Việt tự nhiên, giàu thông tin, mô tả chính xác người, vật, hành động và bối cảnh chính trong ảnh. Không lặp lại, không bịa chi tiết không chắc chắn:"

GENERATION_CONFIG = {
    "max_new_tokens": 64,
    "num_beams": 3,
    "do_sample": False,
    "repetition_penalty": 1.2,
}

# ── Device ────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# ── Load Vision Encoder ───────────────────────────────────────────────────────
print("\n[1/4] Loading Vision Encoder (SigLIP2)...")
from vicaption.models.vision_encoder import SigLIPVisionEncoder
vision_encoder = SigLIPVisionEncoder(
    model_name=VISION_ENCODER,
    torch_dtype=torch.float16,
)
vision_encoder = vision_encoder.to(device)
vision_encoder.eval()
print("     OK")

# ── Load Decoder (4-bit vì GPU local 6GB) ─────────────────────────────────────
print("[2/4] Loading Decoder (Qwen3-1.7B, 4-bit)...")
from vicaption.models.decoder import QwenDecoder
decoder = QwenDecoder(
    model_name=DECODER,
    torch_dtype=torch.float16,
    load_in_4bit=True,
    bnb_4bit_compute_dtype="fp16",
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    device_map="auto",
)
print("     OK")

# ── Load Connector + checkpoint ───────────────────────────────────────────────
print("[3/4] Loading Connector + checkpoint...")
from vicaption.models.connector import QwenStyleConnector
connector = QwenStyleConnector(
    vision_dim=1024,
    llm_dim=2048,
    spatial_merge_size=2,
)

ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu")
# hỗ trợ cả 2 format: raw state_dict hoặc wrapped dict
state_dict = ckpt.get("connector_state_dict", ckpt.get('connector', ckpt))
connector.load_state_dict(state_dict)
connector = connector.to(device).half()
connector.eval()
print(f"     OK — loaded from {CHECKPOINT_PATH}")

# ── Build Captioner ───────────────────────────────────────────────────────────
print("[4/4] Building Captioner...")
from vicaption.models.captioner import VietnameseCaptioner
model = VietnameseCaptioner(vision_encoder, connector, decoder)
model.eval()
print("     OK\n")

# ── Helpers ───────────────────────────────────────────────────────────────────
processor = vision_encoder.processor
tokenizer = decoder.tokenizer


def generate_caption(image_path: str) -> str:
    image = Image.open(image_path).convert("RGB")
    pixel_values = processor(images=[image], return_tensors="pt")["pixel_values"].to(device)

    prompt_batch = tokenizer(
        [PROMPT],
        padding=True,
        truncation=True,
        max_length=32,
        return_tensors="pt",
    )
    prompt_input_ids      = prompt_batch["input_ids"].to(device)
    prompt_attention_mask = prompt_batch["attention_mask"].to(device)

    with torch.no_grad():
        sequences = model.generate(
            pixel_values=pixel_values,
            prompt_input_ids=prompt_input_ids,
            prompt_attention_mask=prompt_attention_mask,
            generation_kwargs=GENERATION_CONFIG,
        )

    first = sequences[0] if hasattr(sequences, "__getitem__") else sequences
    text  = tokenizer.decode(first, skip_special_tokens=True)
    return text.strip()


# ── Inference ─────────────────────────────────────────────────────────────────
with open(TEST_JSON, encoding="utf-8") as f:
    test_data = json.load(f)

# lấy NUM_SAMPLES ảnh, không trùng image_id
seen = set()
samples = []
for item in test_data:
    if item["image_id"] not in seen:
        seen.add(item["image_id"])
        samples.append(item)
    if len(samples) >= NUM_SAMPLES:
        break

print(f"{'='*60}")
print(f"Inference {NUM_SAMPLES} ảnh từ test set")
print(f"{'='*60}\n")

for i, item in enumerate(samples):
    image_path = Path(IMAGE_DIR) / item["image_id"]
    if not image_path.exists():
        print(f"[{i+1}] ⚠  Không tìm thấy: {item['image_id']}\n")
        continue

    try:
        pred = generate_caption(str(image_path))
    except Exception as e:
        print(f"[{i+1}] ✗  Lỗi: {e}\n")
        continue

    print(f"[{i+1}] {item['image_id']}")
    print(f"     Ref : {item['caption']}")
    print(f"     Pred: {pred}")
    print()