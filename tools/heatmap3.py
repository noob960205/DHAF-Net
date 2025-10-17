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
        description="DHAF-Net EigenCAM heatmaps for specified layers"
    )
    p.add_argument(
        "--weights",
        type=str,
        default="./runs/detect/M3FD-DHAFNet/weights/best.pt",
        help="Path to trained weights .pt",
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
    p.add_argument(
        "--save_dir", type=str, default="./result", help="Directory to save outputs"
    )
    p.add_argument(
        "--device", type=str, default="cuda:1", help="Device like cuda:0 or cpu"
    )
    p.add_argument("--imgsz", type=int, default=640, help="Resize square image size")
    p.add_argument(
        "--layers", type=str, default="7,17,27,34", help="Comma-separated layer indices"
    )
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
    if ir_img.shape[2] == 1:
        # replicate single-channel IR to 3 channels to form 6-channel input
        ir_img = np.repeat(ir_img, 3, axis=2)
    elif ir_img.shape[2] == 3:
        # ensure IR is in RGB order like rgb_img (already handled in read_img)
        pass
    # stack to 6 channels
    six = np.dstack([rgb_img, ir_img])  # (H,W,6)
    tensor = (
        torch.from_numpy(six).permute(2, 0, 1).unsqueeze(0).contiguous()
    )  # (1,6,H,W)
    return tensor


def get_module_by_index(det_model, idx):
    # Ultralytics DetectionModel keeps blocks in det_model.model (ModuleList)
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
    # grayscale_cam: (N,H,W) in [0,1]
    return grayscale_cam[0]


def to_visual_base(img):
    # img in [0,1], shape (H,W,C)
    if img.shape[2] == 1:
        base = np.repeat(img, 3, axis=2)
    else:
        base = img
    return base


def overlay_cam(base_img01, cam_map):
    # base_img01: (H,W,3) float in [0,1]
    heat = (cam_map * 255.0).astype(np.uint8)
    heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    base_uint8 = (np.clip(base_img01, 0, 1) * 255.0).astype(np.uint8)
    overlay = cv2.addWeighted(base_uint8, 0.5, heat, 0.5, 0)
    return overlay


def put_title(img, text):
    out = img.copy()
    cv2.putText(
        out,
        text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return out


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    device = select_device(args.device)

    # Load YOLO model and get torch nn.Module
    yo = YOLO(args.weights)
    det_model = yo.model
    det_model.to(device).eval()

    # Read images and build 6-channel input
    rgb = read_img(args.rgb, args.imgsz)
    ir = read_img(args.ir, args.imgsz)
    # ensure IR has 3 channels for 6-channel stacking
    if ir.shape[2] == 1:
        ir = np.repeat(ir, 3, axis=2)
    inp = make_input(rgb, ir)

    # Prepare layers
    indices = [int(x.strip()) for x in args.layers.split(",") if x.strip()]
    if args.verbose:
        print("Available layers:")
        for s in list_layers(det_model):
            print(s)
    target_layers = []
    for idx in indices:
        try:
            target_layers.append(get_module_by_index(det_model, idx))
        except Exception as e:
            print(f"[Warn] Cannot select layer {idx}: {e}")
    if len(target_layers) != len(indices):
        print(
            "[Error] Some target layers not found. Please use --verbose to list layers."
        )
        return

    # Compute EigenCAM for each target layer
    cams = []
    for tl in target_layers:
        cam_map = compute_eigencam(det_model, tl, inp, device, eigen_smooth=True)
        cams.append(cam_map)

    # Prepare overlays and a comparison strip
    rgb_base = to_visual_base(rgb)
    ir_base = to_visual_base(ir if ir.shape[2] == 3 else np.repeat(ir, 3, axis=2))

    overlays = [
        put_title((rgb_base * 255).astype(np.uint8), "Original RGB"),
        put_title((ir_base * 255).astype(np.uint8), "Original IR"),
        put_title(overlay_cam(rgb_base, cams[0]), f"RGB Branch (Layer {indices[0]})"),
        put_title(overlay_cam(ir_base, cams[1]), f"IR Branch (Layer {indices[1]})"),
        put_title(overlay_cam(rgb_base, cams[2]), f"Commonality Branch (Layer {indices[2]})"),
        put_title(overlay_cam(rgb_base, cams[3]), f"DHAF Module (Layer {indices[3]})"),
    ]

    row = np.hstack(overlays)

    stem = os.path.splitext(os.path.basename(args.rgb))[0]
    save_path = os.path.join(args.save_dir, f"dhaf_heatmaps_{stem}.png")
    cv2.imwrite(save_path, cv2.cvtColor(row, cv2.COLOR_RGB2BGR))

    # save single overlays too
    names = [
        f"layer{indices[0]}_rgb_overlay_{stem}.png",
        f"layer{indices[1]}_ir_overlay_{stem}.png",
        f"layer{indices[2]}_common_overlay_{stem}.png",
        f"layer{indices[3]}_dhaf_overlay_{stem}.png",
    ]
    singles = [overlays[2], overlays[3], overlays[4], overlays[5]]
    for n, im in zip(names, singles):
        cv2.imwrite(os.path.join(args.save_dir, n), cv2.cvtColor(im, cv2.COLOR_RGB2BGR))

    print(f"Saved: {save_path}")


if __name__ == "__main__":
    main()
