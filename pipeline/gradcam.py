"""Grad-CAM heatmap for the YOLO classifier: shows *where* in the photo drove
the predicted class, since the classifier itself only outputs a whole-image
label with no localization (see README).

This is an approximation, not a validated bounding box: it visualizes which
pixels the network's gradients say mattered most for the predicted class,
computed from the last spatial feature map before the classification head's
global-average-pool (the Classify module's own conv, 1280 channels).

Note the heatmap is drawn on the square, center-cropped view the model
actually sees (Resize(shortest-edge)+CenterCrop, same as training/inference)
-- for a tall photo this crops off top/bottom, so a defect near the top or
bottom edge of the original photo may fall outside the cropped view shown.

Usage:
    python -m pipeline.gradcam path/to/image.jpg
    python -m pipeline.gradcam path/to/image.jpg --weights runs/classify/defects/weights/best.pt --out heatmap.jpg
    python -m pipeline.gradcam path/to/image.jpg --class hniloba   # force the target class instead of top-1
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from ultralytics import YOLO
from ultralytics.data.augment import classify_transforms

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WEIGHTS = REPO_ROOT / "runs" / "classify" / "defects" / "weights" / "best.pt"


def compute_gradcam(
    weights: str, image_path: str, imgsz: int = 640, target_class: str | None = None
) -> tuple[Image.Image, np.ndarray, str, float]:
    """Returns (cropped_input_image, cam_heatmap[0..1] at imgsz x imgsz, predicted_class_name, confidence)."""
    yolo = YOLO(weights)
    net = yolo.model
    net.eval()
    names = net.names

    classify_head = net.model[-1]  # the final Classify module
    target_layer = classify_head.conv  # last spatial conv before global-average-pool

    activation = {}

    def save_activation(_module, _inp, out):
        out.retain_grad()
        activation["value"] = out

    handle = target_layer.register_forward_hook(save_activation)

    pil_img = Image.open(image_path).convert("RGB")
    transform = classify_transforms(imgsz)
    input_tensor = transform(pil_img).unsqueeze(0)
    # Saved checkpoints have requires_grad=False on every parameter (Ultralytics
    # strips the optimizer/grad state for inference-only weights), so the input
    # itself has to require grad or none of the intermediate activations will.
    input_tensor.requires_grad_(True)

    try:
        with torch.enable_grad():
            probs, logits = net(input_tensor)  # eval-mode Classify head returns (softmax, raw_logits)
            probs = probs[0]
            idx = (
                list(names.values()).index(target_class)
                if target_class is not None
                else int(torch.argmax(probs).item())
            )
            score = logits[0, idx]
            net.zero_grad(set_to_none=True)
            score.backward()

            act = activation["value"]
            grad = act.grad
            weights_ = grad.mean(dim=(2, 3), keepdim=True)
            cam = F.relu((weights_ * act).sum(dim=1, keepdim=True))
            cam = F.interpolate(cam, size=(imgsz, imgsz), mode="bilinear", align_corners=False)
            cam = cam[0, 0].detach().numpy()
            cam -= cam.min()
            if cam.max() > 1e-8:
                cam /= cam.max()
    finally:
        handle.remove()

    # Reconstruct the exact square crop the model saw (undo Normalize(mean=0,std=1), which is a no-op here).
    cropped = input_tensor[0].detach().permute(1, 2, 0).clamp(0, 1).numpy()
    cropped_img = Image.fromarray((cropped * 255).astype(np.uint8))

    return cropped_img, cam, names[idx], float(probs[idx].detach())


def overlay_heatmap(base: Image.Image, cam: np.ndarray, alpha: float = 0.45) -> Image.Image:
    base_arr = np.asarray(base).astype(np.float32)

    # Simple blue->red colormap without extra dependencies.
    r = np.clip(1.5 - abs(4 * cam - 3), 0, 1)
    g = np.clip(1.5 - abs(4 * cam - 2), 0, 1)
    b = np.clip(1.5 - abs(4 * cam - 1), 0, 1)
    heat = np.stack([r, g, b], axis=-1) * 255

    blended = (1 - alpha) * base_arr + alpha * heat
    return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("image", help="image file")
    p.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--class", dest="target_class", default=None, help="force this class instead of top-1")
    p.add_argument(
        "--out",
        default=None,
        help="output path (default: gradcam_out/<image-stem>_gradcam.jpg, kept out of data/)",
    )
    p.add_argument("--alpha", type=float, default=0.45, help="heatmap opacity, 0-1")
    args = p.parse_args()

    weights = Path(args.weights)
    if not weights.exists():
        raise SystemExit(f"no weights at {weights} -- train first with `python -m pipeline.train`")

    cropped_img, cam, cls_name, conf = compute_gradcam(
        str(weights), args.image, imgsz=args.imgsz, target_class=args.target_class
    )
    overlay = overlay_heatmap(cropped_img, cam, alpha=args.alpha)

    if args.out:
        out_path = Path(args.out)
    else:
        out_dir = REPO_ROOT / "gradcam_out"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / (Path(args.image).stem + "_gradcam.jpg")
    overlay.save(out_path)
    print(f"{args.image}: predicted '{cls_name}' ({conf:.3f}) -> heatmap saved to {out_path}")


if __name__ == "__main__":
    main()
