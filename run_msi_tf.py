#!/usr/bin/env python3
"""
Run MSI-Net (TF SavedModel) on an image.
Outputs:
  - <out_prefix>.npy    : raw float32 saliency (H x W)
  - <out_prefix>_u8.png : uint8 [0..255] grayscale map (for exact image compares)
  - <out_prefix>_viz.png: pretty overlay for quick viewing (not for exact compares)
"""

import argparse, os, numpy as np, cv2, tensorflow as tf
from huggingface_hub import snapshot_download


def choose_target_shape(w, h):
    ar = h / float(w)

    def d(a, b):
        return abs(a - b)

    square = d(ar, 1.0)
    landscape = d(ar, 240 / 320)
    portrait = d(ar, 320 / 240)
    if square <= landscape and square <= portrait:
        return (320, 320)  # (W, H)
    if landscape <= portrait:
        return (320, 240)
    return (240, 320)


def preprocess_bgr(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)  # 0..255
    h, w = img_rgb.shape[:2]
    tw, th = choose_target_shape(w, h)
    scale = min(tw / w, th / h)
    sw, sh = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(img_rgb, (sw, sh), interpolation=cv2.INTER_LINEAR)

    vert = th - sh
    horz = tw - sw
    top, bottom = vert // 2, vert - vert // 2
    left, right = horz // 2, horz - horz // 2

    padded = cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        borderType=cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )
    return padded, (top, bottom, left, right), (w, h)


def postprocess_to_original(saliency_map, pads, orig_wh):
    top, bottom, left, right = pads
    H, W = saliency_map.shape[:2]
    roi = saliency_map[
        top : H - bottom if bottom > 0 else H, left : W - right if right > 0 else W
    ]
    ow, oh = orig_wh
    restored = cv2.resize(roi, (ow, oh), interpolation=cv2.INTER_LINEAR)
    return restored


def load_model_cached(repo_id="alexanderkroner/MSI-Net"):
    local_dir = snapshot_download(repo_id=repo_id)
    model = tf.saved_model.load(local_dir)
    # Use the serving signature
    infer = model.signatures.get("serving_default", None)
    if infer is None:
        raise RuntimeError("serving_default signature not found")
    return infer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="Path to input image")
    ap.add_argument("--out_prefix", default="msi_tf_out", help="Prefix for outputs")
    args = ap.parse_args()

    img = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(args.image)

    inp, pads, orig_wh = preprocess_bgr(img)  # HxWx3 float32 (0..255)
    inp_nhwc = inp[None, ...]  # NHWC

    infer = load_model_cached()
    out_dict = infer(tf.constant(inp_nhwc))
    if len(out_dict) != 1:
        raise RuntimeError(f"Expected single output, got keys: {list(out_dict.keys())}")
    out_tensor = next(iter(out_dict.values()))
    out = out_tensor.numpy()  # [1, H, W, 1] float32
    out = np.squeeze(out, axis=(0, 3))  # HxW

    sal = postprocess_to_original(out, pads, orig_wh=orig_wh)  # HxW float32

    # Save raw float output (best for numeric comparisons)
    np.save(f"{args.out_prefix}.npy", sal)

    # Deterministic uint8 grayscale (0..255) for exact image compares
    # Note: min/max normalization makes exact equality sensitive to small changes.
    # Use only to compare two results produced by the SAME pipeline.
    mn, mx = float(sal.min()), float(sal.max())
    if mx > mn:
        sal_u8 = np.clip((sal - mn) * (255.0 / (mx - mn)) + 0.5, 0, 255).astype(
            np.uint8
        )
    else:
        sal_u8 = np.zeros_like(sal, dtype=np.uint8)
    cv2.imwrite(f"{args.out_prefix}_u8.png", sal_u8)

    # Quick pretty viz (not for exact equality)
    color = cv2.applyColorMap(sal_u8, cv2.COLORMAP_INFERNO)
    viz = cv2.addWeighted(color, 0.65, img, 0.35, 0.0)
    cv2.imwrite(f"{args.out_prefix}_viz.png", viz)

    print(
        f"Saved: {args.out_prefix}.npy, {args.out_prefix}_u8.png, {args.out_prefix}_viz.png"
    )


if __name__ == "__main__":
    # keep TF quiet
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    main()
