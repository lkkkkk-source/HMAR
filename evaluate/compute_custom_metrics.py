import argparse
import os
from typing import List

import numpy as np
import torch
from PIL import Image
from skimage.metrics import structural_similarity


def _list_images_from_dirs(dirs: List[str]):
    paths = []
    for split_dir in dirs:
        for root, _, files in os.walk(split_dir):
            for name in sorted(files):
                if name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp")):
                    paths.append(os.path.join(root, name))
    return paths


def _load_rgb(path: str):
    with Image.open(path) as img:
        return np.asarray(img.convert("RGB"), dtype=np.uint8)


def _compute_ssim(sample_paths: List[str], ref_paths: List[str]):
    if len(sample_paths) != len(ref_paths):
        raise ValueError(f"SSIM expects equal counts, got {len(sample_paths)} samples vs {len(ref_paths)} refs")
    vals = []
    for sample_path, ref_path in zip(sample_paths, ref_paths):
        sample = _load_rgb(sample_path)
        ref = _load_rgb(ref_path)
        vals.append(
            structural_similarity(
                sample,
                ref,
                channel_axis=2,
                data_range=255,
            )
        )
    return float(np.mean(vals))


def _compute_lpips(sample_paths: List[str], ref_paths: List[str], device: str):
    if len(sample_paths) != len(ref_paths):
        raise ValueError(f"LPIPS expects equal counts, got {len(sample_paths)} samples vs {len(ref_paths)} refs")
    import lpips

    loss_fn = lpips.LPIPS(net="alex").to(device)
    vals = []
    for sample_path, ref_path in zip(sample_paths, ref_paths):
        sample = torch.from_numpy(_load_rgb(sample_path)).permute(2, 0, 1).float() / 127.5 - 1.0
        ref = torch.from_numpy(_load_rgb(ref_path)).permute(2, 0, 1).float() / 127.5 - 1.0
        sample = sample.unsqueeze(0).to(device)
        ref = ref.unsqueeze(0).to(device)
        with torch.inference_mode():
            vals.append(loss_fn(sample, ref).item())
    return float(np.mean(vals))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_dir", required=True)
    parser.add_argument("--ref_dirs", nargs="+", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    try:
        from cleanfid import fid
    except ImportError as e:
        raise ImportError("Please install clean-fid: pip install clean-fid") from e

    sample_paths = _list_images_from_dirs([args.sample_dir])
    ref_paths = _list_images_from_dirs(args.ref_dirs)

    if len(sample_paths) != len(ref_paths):
        print(
            f"[compute_custom_metrics] warning: sample/ref counts differ ({len(sample_paths)} vs {len(ref_paths)}). "
            "FID/KID will still run, but LPIPS/SSIM require equal counts."
        )

    if len(args.ref_dirs) != 1:
        raise ValueError("clean-fid path mode expects exactly one merged reference directory. Merge ref dirs first.")
    ref_dir = args.ref_dirs[0]

    fid_score = fid.compute_fid(args.sample_dir, ref_dir, mode="clean", num_workers=0)
    kid_score = fid.compute_kid(args.sample_dir, ref_dir, mode="clean", num_workers=0)

    ssim_score = _compute_ssim(sample_paths, ref_paths)
    lpips_score = _compute_lpips(sample_paths, ref_paths, args.device)

    print(
        {
            "FID": float(fid_score),
            "KID": float(kid_score),
            "LPIPS": float(lpips_score),
            "SSIM": float(ssim_score),
            "num_samples": len(sample_paths),
            "num_refs": len(ref_paths),
        }
    )


if __name__ == "__main__":
    main()
