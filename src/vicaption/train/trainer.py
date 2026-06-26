from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from vicaption.utils.checkpoint import save_checkpoint
from vicaption.utils.logger import get_logger


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if hasattr(value, "to") else value
    return moved


class Trainer:
    def __init__(self, model, train_loader, val_loader, optimizer, config, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.config = config
        self.device = device
        self.logger = get_logger()

        train_cfg = config["train"]
        self.grad_accum_steps = int(train_cfg.get("grad_accum_steps", 1))
        self.log_every = int(train_cfg.get("log_every", 20))
        self.save_dir = Path(train_cfg.get("save_dir", "checkpoints"))
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.use_amp = (
            device.type == "cuda"
            and str(train_cfg.get("mixed_precision", "")).lower() == "fp16"
        )
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.best_val_loss = float("inf")

    def _autocast(self):
        if self.use_amp:
            return torch.cuda.amp.autocast()
        return nullcontext()

    def _set_train_mode(self) -> None:
        self.model.connector.train()
        self.model.vision_encoder.eval()
        self.model.decoder.eval()

    def train_one_epoch(self, epoch: int) -> float:
        self._set_train_mode()
        self.optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        num_steps = 0

        progress = tqdm(self.train_loader, desc=f"train epoch {epoch}", leave=False)
        for step, batch in enumerate(progress):
            batch = move_batch_to_device(batch, self.device)
            with self._autocast():
                outputs = self.model(batch)
                raw_loss = outputs["loss"]
                loss = raw_loss / self.grad_accum_steps

            self.scaler.scale(loss).backward()
            should_step = (step + 1) % self.grad_accum_steps == 0 or step + 1 == len(self.train_loader)
            if should_step:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)

            loss_value = float(raw_loss.detach().cpu())
            total_loss += loss_value
            num_steps += 1
            progress.set_postfix(loss=f"{loss_value:.4f}")
            if self.log_every and (step + 1) % self.log_every == 0:
                self.logger.info("epoch=%s step=%s loss=%.4f", epoch, step + 1, loss_value)

        return total_loss / max(num_steps, 1)

    @torch.no_grad()
    def validate(self, epoch: int) -> float:
        if self.val_loader is None:
            return float("inf")

        self.model.eval()
        total_loss = 0.0
        num_steps = 0
        progress = tqdm(self.val_loader, desc=f"val epoch {epoch}", leave=False)
        for batch in progress:
            batch = move_batch_to_device(batch, self.device)
            with self._autocast():
                outputs = self.model(batch)
            loss_value = float(outputs["loss"].detach().cpu())
            total_loss += loss_value
            num_steps += 1
            progress.set_postfix(loss=f"{loss_value:.4f}")

        return total_loss / max(num_steps, 1)

    def fit(self) -> dict[str, float]:
        epochs = int(self.config["train"].get("epochs", 1))
        validate_every_epoch = bool(self.config["train"].get("validate_every_epoch", True))
        history = {"train_loss": float("inf"), "val_loss": float("inf")}

        for epoch in range(1, epochs + 1):
            train_loss = self.train_one_epoch(epoch)
            val_loss = self.validate(epoch) if validate_every_epoch else float("inf")
            history = {"train_loss": train_loss, "val_loss": val_loss}
            self.logger.info(
                "epoch=%s train_loss=%.4f val_loss=%s",
                epoch,
                train_loss,
                f"{val_loss:.4f}" if val_loss != float("inf") else "inf",
            )

            last_path = self.save_dir / "connector_last.pt"
            save_checkpoint(last_path, self.model.connector, self.optimizer, epoch, self.best_val_loss, self.config)

            criterion = val_loss if val_loss != float("inf") else train_loss
            if criterion < self.best_val_loss:
                self.best_val_loss = criterion
                best_path = self.save_dir / "connector_best.pt"
                save_checkpoint(best_path, self.model.connector, self.optimizer, epoch, self.best_val_loss, self.config)

        return history

