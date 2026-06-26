from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from vicaption.data.collator import CaptionCollator
from vicaption.data.dataset import Flickr30kViDataset
from vicaption.models.captioner import VietnameseCaptioner
from vicaption.models.connector import QwenStyleConnector
from vicaption.models.decoder import QwenDecoder
from vicaption.models.vision_encoder import SigLIPVisionEncoder
from vicaption.train.trainer import Trainer
from vicaption.utils.config import load_config
from vicaption.utils.device import get_device
from vicaption.utils.seed import set_seed


def run_training(config_path: str) -> dict[str, float]:
    config = load_config(config_path)
    set_seed(int(config["project"].get("seed", 42)))
    device = get_device(config["project"].get("device", "cuda"))

    model_cfg = config["model"]
    train_cfg = config["train"]
    data_cfg = config["data"]
    dtype = torch.float16 if device.type == "cuda" and train_cfg.get("mixed_precision") == "fp16" else torch.float32

    vision_encoder = SigLIPVisionEncoder(
        model_name=model_cfg["vision_encoder"],
        image_size=int(model_cfg["image_size"]),
        patch_size=int(model_cfg["patch_size"]),
        torch_dtype=dtype,
    )
    decoder = QwenDecoder(model_name=model_cfg["decoder"], torch_dtype=dtype)
    connector = QwenStyleConnector(
        vision_dim=int(model_cfg["vision_dim"]),
        llm_dim=int(model_cfg["llm_dim"]),
        spatial_merge_size=int(model_cfg["spatial_merge_size"]),
    )
    model = VietnameseCaptioner(vision_encoder, connector, decoder).to(device)

    total_trainable, connector_params = model.trainable_parameter_counts()
    assert total_trainable == connector_params, "Only Connector parameters should be trainable."

    processed_dir = Path(data_cfg["processed_dir"])
    train_dataset = Flickr30kViDataset(str(processed_dir / "train.json"), data_cfg["image_dir"])
    val_dataset = Flickr30kViDataset(str(processed_dir / "val.json"), data_cfg["image_dir"])
    if len(train_dataset) == 0:
        raise ValueError("Training split is empty. Run preprocessing after adding Flickr30k Vietnamese data.")

    collator = CaptionCollator(
        siglip_processor=vision_encoder.processor,
        qwen_tokenizer=decoder.tokenizer,
        prompt=config["prompt"]["text"],
        max_prompt_length=int(train_cfg["max_prompt_length"]),
        max_caption_length=int(train_cfg["max_caption_length"]),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=True,
        num_workers=int(train_cfg["num_workers"]),
        collate_fn=collator,
        pin_memory=device.type == "cuda",
    )
    val_loader = None
    if len(val_dataset) > 0:
        val_loader = DataLoader(
            val_dataset,
            batch_size=int(train_cfg["batch_size"]),
            shuffle=False,
            num_workers=int(train_cfg["num_workers"]),
            collate_fn=collator,
            pin_memory=device.type == "cuda",
        )

    optimizer = torch.optim.AdamW(
        model.connector.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    trainer = Trainer(model, train_loader, val_loader, optimizer, config, device)
    return trainer.fit()

