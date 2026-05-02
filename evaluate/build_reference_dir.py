import argparse
import os
import shutil


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split_dirs", nargs="+", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    if os.path.exists(args.out_dir):
        shutil.rmtree(args.out_dir)
    os.makedirs(args.out_dir, exist_ok=True)

    idx = 0
    for split_dir in args.split_dirs:
        for root, _, files in os.walk(split_dir):
            for name in sorted(files):
                if name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp")):
                    src = os.path.join(root, name)
                    ext = os.path.splitext(name)[1].lower()
                    dst = os.path.join(args.out_dir, f"{idx:06d}{ext}")
                    shutil.copy2(src, dst)
                    idx += 1

    print(f"merged {idx} images into {args.out_dir}")


if __name__ == "__main__":
    main()
