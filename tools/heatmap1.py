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
        description="对比多种结构在指定层的EigenCAM热力图（用于论证架构必要性）"
    )
    # 各模型权重路径
    p.add_argument(
        "--weights_rgb", type=str, default="./runs/detect/M3FD-RGB/weights/best.pt", help="RGB-Only权重路径"
    )
    p.add_argument(
        "--weights_ir", type=str, default="./runs/detect/M3FD-IR/weights/best.pt", help="IR-Only权重路径"
    )
    p.add_argument(
        "--weights_early", type=str, default="./runs/detect/M3FD-EarlyFusion/weights/best.pt", help="Early Fusion权重路径"
    )
    p.add_argument(
        "--weights_twostream", type=str, default="./runs/detect/M3FD-TwoStream/weights/best.pt", help="Two Stream权重路径"
    )
    p.add_argument(
        "--weights_baseline", type=str, default="./runs/detect/M3FD-Siamese/weights/best.pt", help="Baseline(孪生)权重路径"
    )

    # 图像与设备
    p.add_argument("--rgb", type=str, default="./datasets/M3FD/imageRGB/test/04019.png", help="RGB图像路径")
    p.add_argument("--ir", type=str, default="./datasets/M3FD/imageIR/test/04019.png", help="IR图像路径")
    p.add_argument("--save_dir", type=str, default="./result", help="保存目录")
    p.add_argument("--device", type=str, default="cuda:1", help="cuda:0/cuda:1 或 cpu")
    p.add_argument("--imgsz", type=int, default=640, help="输入尺寸(方形缩放)")

    # 各模型目标层索引（默认按需求：RGB-Only=6, IR-Only=6, Early=6, TwoStream=24, Baseline=34）
    p.add_argument("--layer_rgb", type=int, default=6, help="RGB-Only目标层索引")
    p.add_argument("--layer_ir", type=int, default=6, help="IR-Only目标层索引")
    p.add_argument("--layer_early", type=int, default=6, help="Early Fusion目标层索引")
    p.add_argument("--layer_twostream", type=int, default=24, help="Two Stream目标层索引")
    p.add_argument("--layer_baseline", type=int, default=34, help="Baseline(孪生)目标层索引")

    p.add_argument("--verbose", action="store_true", help="打印各模型层索引列表")
    return p.parse_args()


def select_device(dev_str: str):
    if dev_str.startswith("cuda") and torch.cuda.is_available():
        return torch.device(dev_str)
    return torch.device("cpu")


def read_img(path, size):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    if len(img.shape) == 2:  # 单通道 -> (H,W,1)
        img = img[:, :, None]
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    return img


def make_input_rgb(rgb_img):
    # (H,W,3)[0,1] -> (1,3,H,W)
    tensor = torch.from_numpy(rgb_img).permute(2, 0, 1).unsqueeze(0).contiguous()
    return tensor


def make_input_ir(ir_img):
    # IR 确保3通道
    if ir_img.shape[2] == 1:
        ir_img = np.repeat(ir_img, 3, axis=2)
    tensor = torch.from_numpy(ir_img).permute(2, 0, 1).unsqueeze(0).contiguous()
    return tensor


def make_input_six(rgb_img, ir_img):
    # 将IR扩展为3通道后与RGB拼接 -> 6通道
    if ir_img.shape[2] == 1:
        ir_img = np.repeat(ir_img, 3, axis=2)
    six = np.dstack([rgb_img, ir_img])
    tensor = torch.from_numpy(six).permute(2, 0, 1).unsqueeze(0).contiguous()
    return tensor


def get_module_by_index(det_model, idx):
    mdl = getattr(det_model, "model", None)
    if mdl is not None and hasattr(mdl, "__getitem__"):
        return mdl[idx]
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
    # 重要：targets=[] 避免在检测模型tuple输出上做argmax
    grayscale_cam = cam(input_tensor=input_tensor.to(device), targets=[], eigen_smooth=eigen_smooth)
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
    # OpenCV默认字体不支持中文，标题使用英文，顺序与表头一致
    cv2.putText(out, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    device = select_device(args.device)

    # 读取图像
    rgb = read_img(args.rgb, args.imgsz)
    ir = read_img(args.ir, args.imgsz)

    # 加载模型（只取内部的torch.nn.Module用于CAM）
    y_rgb = YOLO(args.weights_rgb);     m_rgb = y_rgb.model.to(device).eval()
    y_ir = YOLO(args.weights_ir);       m_ir = y_ir.model.to(device).eval()
    y_early = YOLO(args.weights_early); m_early = y_early.model.to(device).eval()
    y_two = YOLO(args.weights_twostream); m_two = y_two.model.to(device).eval()
    y_base = YOLO(args.weights_baseline); m_base = y_base.model.to(device).eval()

    if args.verbose:
        print("RGB-Only Layers:")
        for s in list_layers(m_rgb): print(s)
        print("\nIR-Only Layers:")
        for s in list_layers(m_ir): print(s)
        print("\nEarly Fusion Layers:")
        for s in list_layers(m_early): print(s)
        print("\nTwo Stream Layers:")
        for s in list_layers(m_two): print(s)
        print("\nBaseline(Siamese) Layers:")
        for s in list_layers(m_base): print(s)

    # 选择目标层
    try:
        l_rgb = get_module_by_index(m_rgb, args.layer_rgb)
    except Exception as e:
        print(f"[Error] RGB-Only layer {args.layer_rgb} not found: {e}")
        return
    try:
        l_ir = get_module_by_index(m_ir, args.layer_ir)
    except Exception as e:
        print(f"[Error] IR-Only layer {args.layer_ir} not found: {e}")
        return
    try:
        l_early = get_module_by_index(m_early, args.layer_early)
    except Exception as e:
        print(f"[Error] Early Fusion layer {args.layer_early} not found: {e}")
        return
    try:
        l_two = get_module_by_index(m_two, args.layer_twostream)
    except Exception as e:
        print(f"[Error] Two Stream layer {args.layer_twostream} not found: {e}")
        return
    try:
        l_base = get_module_by_index(m_base, args.layer_baseline)
    except Exception as e:
        print(f"[Error] Baseline layer {args.layer_baseline} not found: {e}")
        return

    # 准备不同输入
    inp_rgb = make_input_rgb(rgb)
    inp_ir = make_input_ir(ir)
    inp_six = make_input_six(rgb, ir)

    # 分别计算CAM
    cam_rgb = compute_eigencam(m_rgb, l_rgb, inp_rgb, device, eigen_smooth=True)
    cam_ir = compute_eigencam(m_ir, l_ir, inp_ir, device, eigen_smooth=True)
    cam_early = compute_eigencam(m_early, l_early, inp_six, device, eigen_smooth=True)
    cam_two = compute_eigencam(m_two, l_two, inp_six, device, eigen_smooth=True)
    cam_base = compute_eigencam(m_base, l_base, inp_six, device, eigen_smooth=True)

    # 叠加到可视化底图
    rgb_base = to_visual_base(rgb)
    ir_base = to_visual_base(ir if ir.shape[2] == 3 else np.repeat(ir, 3, axis=2))

    tiles = [
        put_title((rgb_base * 255).astype(np.uint8), "Original RGB"),
        put_title((ir_base * 255).astype(np.uint8), "Original IR"),
        put_title(overlay_cam(rgb_base, cam_rgb), f"RGB-Only (Layer {args.layer_rgb})"),
        put_title(overlay_cam(ir_base, cam_ir), f"IR-Only (Layer {args.layer_ir})"),
        put_title(overlay_cam(rgb_base, cam_early), f"Early Fusion (Layer {args.layer_early})"),
        put_title(overlay_cam(rgb_base, cam_two), f"Two Stream (Layer {args.layer_twostream})"),
        put_title(overlay_cam(rgb_base, cam_base), f"Baseline (Layer {args.layer_baseline})"),
    ]

    row = np.hstack(tiles)

    stem = os.path.splitext(os.path.basename(args.rgb))[0]
    out_name = (
        f"compare_RGB_IR_Early_Two_Baseline_"
        f"r{args.layer_rgb}_i{args.layer_ir}_e{args.layer_early}_t{args.layer_twostream}_b{args.layer_baseline}_"
        f"{stem}.png"
    )
    save_path = os.path.join(args.save_dir, out_name)
    cv2.imwrite(save_path, cv2.cvtColor(row, cv2.COLOR_RGB2BGR))

    # 另存单张热力图叠加
    single_names = {
        "rgb_only": tiles[2],
        "ir_only": tiles[3],
        "early_fusion": tiles[4],
        "two_stream": tiles[5],
        "baseline": tiles[6],
    }
    for k, im in single_names.items():
        cv2.imwrite(os.path.join(args.save_dir, f"{k}_overlay_{stem}.png"), cv2.cvtColor(im, cv2.COLOR_RGB2BGR))

    print(f"Saved: {save_path}")


if __name__ == "__main__":
    main()