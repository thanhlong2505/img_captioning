from __future__ import annotations

import json
from pathlib import Path
from typing import Any

try:
    from torch.utils.data import Dataset
except ImportError:  # pragma: no cover - keeps lightweight data tests importable
    class Dataset:  # type: ignore[no-redef]
        pass


class Flickr30kViDataset(Dataset):
    """Flat Flickr30k Vietnamese caption dataset."""

    def __init__(self, json_path: str, image_dir: str):
        self.json_path = Path(json_path)
        self.image_dir = Path(image_dir)

        with self.json_path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        if not isinstance(data, list):
            raise ValueError("Dataset JSON must be a list.")
        self.items: list[dict[str, Any]] = data

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, str]:
        item = self.items[index]
        image_id = item.get("image_id")
        caption = item.get("caption")
        if not isinstance(image_id, str) or not isinstance(caption, str):
            raise ValueError("Each dataset item must contain image_id and caption strings.")

        return {
            "image_id": image_id,
            "image_path": str(self.image_dir / image_id),
            "caption": caption,
        }

