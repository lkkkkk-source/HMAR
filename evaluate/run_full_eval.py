import argparse
import ast
import json
import os
import subprocess
import sys


MASK_STAGES = ["10", "15", "20", "30"]


def run(cmd):
    print(f"[run] {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def capture_json_output(cmd):
    print(f"[run] {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, check=True, text=True, capture_output=True)
    stdout = proc.stdout.strip().splitlines()
    if not stdout:
        raise RuntimeError("No output captured from metrics command")
    last_line = stdout[-1].strip()
    return ast.literal_eval(last_line)


def count_images(folder):
    count = 0
    for name in os.listdir(folder):
        if name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp")):
            count += 1
    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_dir", required=True)
    parser.add_argument("--public_hmar_ckpt", required=True)
    parser.add_argument("--vae_ckpt", required=True)
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--sample_config", default="hmar-d16")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--total_samples", type=int, default=8700)
    parser.add_argument("--class_counts", default="0:954,1:1848,2:1602,3:1560,4:1482,5:1254")
    parser.add_argument("--ref_dir", default="ref_all_dir")
    parser.add_argument("--results_json", default="full_eval_results.json")
    args = parser.parse_args()

    run(
        [
            sys.executable,
            "-m",
            "evaluate.build_reference_dir",
            "--split_dirs",
            os.path.join(args.dataset_root, "train"),
            os.path.join(args.dataset_root, "val"),
            os.path.join(args.dataset_root, "test"),
            "--out_dir",
            args.ref_dir,
        ]
    )

    results = {}
    for stage in MASK_STAGES:
        ckpt = os.path.join(args.experiment_dir, f"ar-ckpt-last-mask{stage}.pth")
        if not os.path.exists(ckpt):
            print(f"[skip] checkpoint not found: {ckpt}", flush=True)
            continue

        sample_dir = f"samples_all_8700_mask{stage}"
        run(
            [
                "python",
                "-m",
                "evaluate.generate_finetune_samples",
                "--checkpoint",
                ckpt,
                "--public_hmar_ckpt",
                args.public_hmar_ckpt,
                "--sample_config",
                args.sample_config,
                "--vae_ckpt",
                args.vae_ckpt,
                "--out_dir",
                sample_dir,
                "--total_samples",
                str(args.total_samples),
                "--batch_size",
                str(args.batch_size),
                "--class_counts",
                args.class_counts,
            ]
        )

        sample_count = count_images(sample_dir)
        if sample_count != args.total_samples:
            raise RuntimeError(
                f"{sample_dir} contains {sample_count} images, expected {args.total_samples}"
            )

        metrics = capture_json_output(
            [
                "python",
                "-m",
                "evaluate.compute_custom_metrics",
                "--sample_dir",
                sample_dir,
                "--ref_dirs",
                args.ref_dir,
            ]
        )
        results[f"mask{stage}"] = metrics

    with open(args.results_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
