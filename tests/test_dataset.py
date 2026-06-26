from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vicaption.data.dataset import Flickr30kViDataset


def test_dataset_returns_expected_fields(tmp_path):
    json_path = tmp_path / "train.json"
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    json_path.write_text(
        json.dumps([{"image_id": "sample.jpg", "caption": "Một ảnh thử."}], ensure_ascii=False),
        encoding="utf-8",
    )

    dataset = Flickr30kViDataset(str(json_path), str(image_dir))
    item = dataset[0]

    assert len(dataset) == 1
    assert item["image_id"] == "sample.jpg"
    assert item["caption"] == "Một ảnh thử."
    assert item["image_path"] == str(image_dir / "sample.jpg")

