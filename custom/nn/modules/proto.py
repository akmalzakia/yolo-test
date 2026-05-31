"""
visualise_prototypes.py — Inspect K-means Prototypes Before Training
=====================================================================
Run before training to verify that the extracted prototype kernels look
geometrically sensible for your dataset.

Usage
-----
    python visualise_prototypes.py --data cctsdb.yaml --K 4 --kernel-size 3

Output
------
    prototypes.png  — K prototype images (kernel_size x kernel_size each),
                      upscaled 64x for visibility, saved to current directory.
    prototypes.npy  — K x kernel_size x kernel_size float32 array, saved for
                      offline inspection or loading into a notebook.

What to look for
----------------
  Good outcome  : Distinct blobs — e.g. one circular, one triangular wedge,
                  one rectangular band, one octagonal shape. This means k-means
                  found semantically different groups in the sign crops.

  Bad outcome   : All K images look nearly identical (uniform grey). This
                  happens when the crops are too small, too noisy, or the
                  kernel_size is too small to capture shape structure.
                  Fix: increase --max-crops, or use a larger --kernel-size.

  Partial outcome: Some distinct, some similar. Acceptable — the ortho
                   regulariser (Strategy 4) will push them apart during training.
"""
import glob
import argparse
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",        default="cctsdb.yaml")
    parser.add_argument("--K",           type=int, default=4)   
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument("--max-crops",   type=int, default=2000)
    parser.add_argument("--output-dir",  default=".")
    return parser.parse_args()

def load_cctsdb_crops(
    data_yaml: str,
    split: str = "train",
    max_per_class: int = 1000,
) -> list[np.ndarray]:
    """
    Load bounding-box crops from a YOLO-format dataset.
 
    Reads `data_yaml` to find the image/label directories, then iterates
    label .txt files (YOLO format: class cx cy w h normalised) to extract
    crops from the corresponding images.
 
    Args:
        data_yaml     : path to the dataset YAML (e.g. cctsdb.yaml).
        split         : 'train' or 'val'.
        max_per_class : max crops per class to keep (balanced sampling).
 
    Returns:
        List of grayscale numpy arrays (H, W), variable sizes.
        Returns empty list if loading fails (graceful fallback to Kaiming init).
    """
    import cv2
    import yaml
 
    try:
        with open(data_yaml) as f:
            cfg = yaml.safe_load(f)
    except Exception as e:
        print(f"[proto] Could not read {data_yaml}: {e}. Falling back to Kaiming init.")
        return []
 
    data_root = Path(cfg.get("path", "."))
    img_dir   = data_root / cfg.get(split, f"images/{split}")
    lbl_dir   = Path(str(img_dir).replace("images", "labels"))
 
    if not lbl_dir.exists():
        print(f"[proto] Label dir not found: {lbl_dir}. Falling back to Kaiming init.")
        return []
 
    crops: list[np.ndarray] = []
    counts: dict = {}
 
    label_files = sorted(glob.glob(str(lbl_dir / "*.txt")))
    for lbl_path in label_files:
        # Find corresponding image (try common extensions)
        stem = Path(lbl_path).stem
        img_path = None
        for ext in (".jpg", ".jpeg", ".png", ".bmp"):
            candidate = img_dir / (stem + ext)
            if candidate.exists():
                img_path = candidate
                break
        if img_path is None:
            continue
 
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        H, W = img.shape[:2]
 
        with open(lbl_path) as f:
            lines = f.read().strip().splitlines()
 
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls = int(parts[0])
            cx, cy, bw, bh = map(float, parts[1:5])
 
            if counts.get(cls, 0) >= max_per_class:
                continue
 
            # Convert normalised → pixel coords
            x1 = int((cx - bw / 2) * W)
            y1 = int((cy - bh / 2) * H)
            x2 = int((cx + bw / 2) * W)
            y2 = int((cy + bh / 2) * H)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W, x2), min(H, y2)
 
            if x2 <= x1 or y2 <= y1:
                continue
 
            crop = img[y1:y2, x1:x2]
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            crops.append(gray)
            counts[cls] = counts.get(cls, 0) + 1
 
    print(f"[proto] Loaded {len(crops)} crops from {data_yaml} ({split} split).")
    return crops


def main():
    args = parse_args()

    # ---- Load crops --------------------------------------------------------
    crops = load_cctsdb_crops(args.data, split="train")

    if len(crops) < args.K:
        print(f"ERROR: Only {len(crops)} crops loaded. Need >= {args.K}.")
        return

    # ---- Extract prototypes ------------------------------------------------
    from dyconv import extract_prototypes
    protos = extract_prototypes(
        crop_arrays=crops,
        k_clusters=args.K,
        kernel_size=args.kernel_size,
        max_crops=args.max_crops,
    )

    # ---- Save .npy ---------------------------------------------------------
    out_dir = Path(args.output_dir)
    npy_path = out_dir / "prototypes.npy"
    np.save(npy_path, np.stack(protos))
    print(f"Saved prototypes array → {npy_path}")

    # ---- Render PNG --------------------------------------------------------
    try:
        import cv2

        scale = 64  # upscale factor for visibility
        k = args.kernel_size
        pad = 4
        row_h = k * scale + 2 * pad
        row_w = args.K * (k * scale + 2 * pad)
        canvas = np.ones((row_h, row_w), dtype=np.uint8) * 200  # light grey bg

        for i, proto in enumerate(protos):
            # Normalise to [0, 255]
            p = proto.copy().astype(np.float32)
            p_min, p_max = p.min(), p.max()
            if p_max > p_min:
                p = 255.0 * (p - p_min) / (p_max - p_min)
            else:
                p = np.full_like(p, 128.0)
            p_uint8 = p.astype(np.uint8)

            # Upscale
            big = cv2.resize(p_uint8, (k * scale, k * scale),
                             interpolation=cv2.INTER_NEAREST)

            x_start = i * (k * scale + 2 * pad) + pad
            canvas[pad: pad + k * scale, x_start: x_start + k * scale] = big

        # Add labels
        for i in range(args.K):
            x_start = i * (k * scale + 2 * pad) + pad
            cv2.putText(
                canvas,
                f"K{i}",
                (x_start + 2, row_h - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, 0, 1, cv2.LINE_AA,
            )

        png_path = out_dir / "prototypes.png"
        cv2.imwrite(str(png_path), canvas)
        print(f"Saved prototype visualisation → {png_path}")
        print("\nInterpretation:")
        for i in range(args.K):
            print(f"  K{i}: inspect {png_path} to see if shape is geometrically "
                  f"distinct from the others.")

    except ImportError:
        print("OpenCV not available — skipping PNG output. "
              "Install with: pip install opencv-python-headless")


if __name__ == "__main__":
    main()