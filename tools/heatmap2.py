# -*- coding: utf-8 -*-
import os
import cv2
import argparse
import numpy as np
import torch
from ultralytics import YOLO
from pytorch_grad_cam import EigenCAM


def parse_args():
    p = argparse.ArgumentParser(
        description="Compare Layer 34 heatmaps between Baseline (Siamese) and DHAF-Net"
    )
    p.add_argument(
        "--weights_baseline",
        type=str,
        default="./runs/detect/M3FD-Siamese/weights/best.pt",
        help="Path to baseline (Siamese, no DHAF) weights",
    )
    p.add_argument(
        "--weights_dhaf",
        type=str,
        default="./runs/detect/M3FD-DHAFNet/weights/best.pt",
        help="Path to DHAF-Net weights",
    )
    p.add_argument(
        "--rgb",
        type=str,
        default="./datasets/M3FD/imageRGB/test/04019.png",
        help="Path to RGB image",
    )
    p.add_argument(
        "--ir",
        type=str,
        default="./datasets/M3FD/imageIR/test/04019.png",
        help="Path to IR image",
    )
    p.add_argument("--save_dir", type=str, default="./result", help="Directory to save outputs")
    p.add_argument("--device", type=str, default="cuda:1", help="Device like cuda:0 or cpu")
    p.add_argument("--imgsz", type=int, default=640, help="Resize square image size")
    p.add_argument("--layer", type=int, default=34, help="Layer index to visualize (default 34)")
    p.add_argument("--verbose", action="store_true", help="Print layer list")
    return p.parse_args()


def select_device(dev_str: str):
    if dev_str.startswith("cuda") and torch.cuda.is_available():
        return torch.device(dev_str)
    return torch.device("cpu")


def read_img(path, size):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    if len(img.shape) == 2:
        img = img[:, :, None]
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    return img


def make_input(rgb_img, ir_img):
    # ensure IR has 3 channels for 6-channel stacking
    if ir_img.shape[2] == 1:
        ir_img = np.repeat(ir_img, 3, axis=2)
    six = np.dstack([rgb_img, ir_img])  # (H,W,6)
    tensor = (
        torch.from_numpy(six).permute(2, 0, 1).unsqueeze(0).contiguous()
    )  # (1,6,H,W)
    return tensor


def get_module_by_index(det_model, idx):
    mdl = getattr(det_model, "model", None)
    if mdl is not None and hasattr(mdl, "__getitem__"):
        return mdl[idx]
    # fallback: flatten children
    children = list(det_model.children())
    if idx < 0 or idx >= len(children):
        raise IndexError(f"Layer index {idx} out of range (len={len(children)})")
    return children[idx]


def list_layers(det_model):
    mdl = getattr(det_model, "model", None)
    if mdl is not None and hasattr(mdl, "__getitem__"):
        return [f"{i}: {type(m).__name__}" for i, m in enumerate(mdl)]
    return [f"{i}: {type(m).__name__}" for i, m in enumerate(det_model.children())]


def compute_eigencam(det_model, target_layer, input_tensor, device, eigen_smooth=True):
    use_cuda = device.type == "cuda"
    cam = EigenCAM(det_model, target_layers=[target_layer], use_cuda=use_cuda)
    # Important: provide empty targets to avoid classification argmax on detector tuple outputs
    grayscale_cam = cam(
        input_tensor=input_tensor.to(device), targets=[], eigen_smooth=eigen_smooth
    )
    return grayscale_cam[0]


def to_visual_base(img):
    if img.shape[2] == 1:
        base = np.repeat(img, 3, axis=2)
    else:
        base = img
    return base


def overlay_cam(base_img01, cam_map):
    heat = (cam_map * 255.0).astype(np.uint8)
    heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    base_uint8 = (np.clip(base_img01, 0, 1) * 255.0).astype(np.uint8)
    overlay = cv2.addWeighted(base_uint8, 0.5, heat, 0.5, 0)
    return overlay


def put_title(img, text):
    out = img.copy()
    cv2.putText(out, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    device = select_device(args.device)

    # Load models
    yo_base = YOLO(args.weights_baseline)
    base_model = yo_base.model
    base_model.to(device).eval()

    yo_dhaf = YOLO(args.weights_dhaf)
    dhaf_model = yo_dhaf.model
    dhaf_model.to(device).eval()

    # Read images and build 6-channel input
    rgb = read_img(args.rgb, args.imgsz)
    ir = read_img(args.ir, args.imgsz)
    inp = make_input(rgb, ir)

    # Select target layer
    try:
        base_layer = get_module_by_index(base_model, args.layer)
    except Exception as e:
        print(f"[Error] Baseline model layer {args.layer} not found: {e}")
        if args.verbose:
            print("Baseline available layers:")
            for s in list_layers(base_model):
                print(s)
        return

    try:
        dhaf_layer = get_module_by_index(dhaf_model, args.layer)
    except Exception as e:
        print(f"[Error] DHAF model layer {args.layer} not found: {e}")
        if args.verbose:
            print("DHAF available layers:")
            for s in list_layers(dhaf_model):
                print(s)
        return

    # Compute CAMs
    base_cam = compute_eigencam(base_model, base_layer, inp, device, eigen_smooth=True)
    dhaf_cam = compute_eigencam(dhaf_model, dhaf_layer, inp, device, eigen_smooth=True)

    # Prepare overlays and 1-row comparison
    rgb_base = to_visual_base(rgb)
    ir_base = to_visual_base(ir if ir.shape[2] == 3 else np.repeat(ir, 3, axis=2))

    tiles = [
        put_title((rgb_base * 255).astype(np.uint8), "Original RGB"),
        put_title((ir_base * 255).astype(np.uint8), "Original IR"),
        put_title(overlay_cam(rgb_base, base_cam), "Baseline (Layer 34)"),
        put_title(overlay_cam(rgb_base, dhaf_cam), "DHAF Module (Layer 34)"),
    ]

    row = np.hstack(tiles)

    stem = os.path.splitext(os.path.basename(args.rgb))[0]
    save_path = os.path.join(args.save_dir, f"baseline_vs_dhaf_layer{args.layer}_{stem}.png")
    cv2.imwrite(save_path, cv2.cvtColor(row, cv2.COLOR_RGB2BGR))

    # Save individual overlays
    cv2.imwrite(
        os.path.join(args.save_dir, f"baseline_layer{args.layer}_overlay_{stem}.png"),
        cv2.cvtColor(tiles[2], cv2.COLOR_RGB2BGR),
    )
    cv2.imwrite(
        os.path.join(args.save_dir, f"dhaf_layer{args.layer}_overlay_{stem}.png"),
        cv2.cvtColor(tiles[3], cv2.COLOR_RGB2BGR),
    )

    print(f"Saved: {save_path}")


if __name__ == "__main__":
    main()