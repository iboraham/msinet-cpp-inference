#!/usr/bin/env python3
"""
Compare two result files.
Supports:
  - .npy (float32 arrays)
  - image files (png/jpg/…): compared as uint8 arrays

Outputs:
  - 'IDENTICAL' if 100% equal
  - Otherwise prints shape/type differences and max abs diff
"""

import argparse, os, sys, numpy as np, cv2


def load_any(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        arr = np.load(path)
        return arr, "npy"
    # load as BGR then convert to single-channel if needed?
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    return img, "img"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument(
        "--rtol", type=float, default=1e-5, help="Relative tolerance (numeric compare)"
    )
    ap.add_argument(
        "--atol", type=float, default=1e-6, help="Absolute tolerance (numeric compare)"
    )
    args = ap.parse_args()

    A, ta = load_any(args.a)
    B, tb = load_any(args.b)

    # Exact 100% equality first (shape, dtype, and bytes/pixels)
    if A.shape == B.shape and A.dtype == B.dtype and np.array_equal(A, B):
        print("IDENTICAL")
        sys.exit(0)

    print("NOT IDENTICAL (byte/pixel exact compare failed)")
    print(f"  shapes: {A.shape} vs {B.shape}")
    print(f"  dtypes: {A.dtype} vs {B.dtype}")

    # If shapes differ but are broadcastable nonsense, bail out early
    if A.shape != B.shape:
        # Try resizing images for a rough check if both are images
        if ta == "img" and tb == "img":
            B2 = cv2.resize(
                B, (A.shape[1], A.shape[0]), interpolation=cv2.INTER_NEAREST
            )
            diff = A.astype(np.int64) - B2.astype(np.int64)
            mad = np.abs(diff).max()
            print(f"  (resized) max abs diff: {mad}")
        else:
            print("  Skipping numeric diff (mismatched shapes)")
        sys.exit(1)

    # Numeric difference (works for npy or same-shape images)
    A_f = A.astype(np.float64)
    B_f = B.astype(np.float64)
    absdiff = np.abs(A_f - B_f)
    mad = absdiff.max()
    meanad = absdiff.mean()

    # Relative diff where applicable
    denom = np.maximum(1.0, np.abs(B_f))
    rdiff = np.abs(A_f - B_f) / denom
    mrd = rdiff.max()

    print(f"  max abs diff : {mad}")
    print(f"  mean abs diff: {meanad}")
    print(f"  max rel diff : {mrd}")
    # Tolerance check (useful for TF vs ONNX small float divergences)
    if np.allclose(A_f, B_f, rtol=args.rtol, atol=args.atol):
        print(f"CLOSE within rtol={args.rtol}, atol={args.atol}")
        sys.exit(0)
    else:
        print(f"NOT CLOSE within rtol={args.rtol}, atol={args.atol}")
        sys.exit(1)


if __name__ == "__main__":
    main()
