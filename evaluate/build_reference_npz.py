import argparse
import os

import numpy as np
from PIL import Image


def collect_images(split_dirs):
    images = []
    for split_dir in split_dirs:
        for root, _, files in os.walk(split_dir):
            for name in sorted(files):
                if name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp")):
                    path = os.path.join(root, name)
                    with Image.open(path) as img:
                        images.append(np.asarray(img.convert("RGB"), dtype=np.uint8))
    return images


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split_dirs", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    images = collect_images(args.split_dirs)
    arr = np.stack(images, axis=0)
    np.savez(args.out, arr_0=arr)
    print(f"saved {arr.shape[0]} images to {args.out}")


if __name__ == "__main__":
    main()
