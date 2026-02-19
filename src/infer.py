"""Inference entrypoint for single-image prediction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from src.models import create_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference on one image.")
    parser.add_argument("--weights", required=True, help="Path to model checkpoint.")
    parser.add_argument("--image", required=True, help="Path to image file.")
    parser.add_argument("--topk", type=int, default=5, help="Number of top predictions.")
    parser.add_argument("--device", default="cuda", help="Device to use (cuda/cpu).")
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    requested = requested.lower().strip()
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA requested but unavailable, falling back to CPU.")
        return torch.device("cpu")
    return torch.device(requested)


def build_transform(config: dict) -> transforms.Compose:
    data_cfg = config.get("data", {})
    mean = data_cfg.get("normalize_mean", [0.4914, 0.4822, 0.4465])
    std = data_cfg.get("normalize_std", [0.2470, 0.2435, 0.2616])
    return transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.weights, map_location="cpu")
    config = checkpoint.get("config", {})
    model_name = checkpoint.get("model_name", "baseline")
    class_names = checkpoint.get("class_names", [str(i) for i in range(10)])
    num_classes = int(checkpoint.get("num_classes", len(class_names)))

    model_cfg = config.get("model", {})
    model_kwargs = {key: value for key, value in model_cfg.items() if key != "name"}
    model = create_model(model_name, num_classes=num_classes, **model_kwargs)
    model.load_state_dict(checkpoint["model_state_dict"])

    device = resolve_device(args.device)
    model = model.to(device).eval()

    transform = build_transform(config)
    image = Image.open(Path(args.image)).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)
        topk = min(args.topk, probabilities.shape[1])
        values, indices = torch.topk(probabilities, k=topk, dim=1)

    predictions = []
    for score, index in zip(values[0].tolist(), indices[0].tolist()):
        label = class_names[index] if index < len(class_names) else str(index)
        predictions.append({"class_id": index, "label": label, "confidence": round(score, 6)})

    payload = {
        "image": str(Path(args.image)),
        "weights": str(Path(args.weights)),
        "model_name": model_name,
        "device": str(device),
        "topk": topk,
        "predictions": predictions,
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

