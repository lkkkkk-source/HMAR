import argparse
import os
import shutil

import torch
import torchvision

from models import build_vae_hmar
from utils.sampling_arg_util import get_args


def build_hmar_from_finetune_ckpt(checkpoint_path: str, vae_ckpt_path: str, device: str):
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    trainer_state = ckpt["trainer"]["transformer_wo_ddp"]

    ns_head_weight = trainer_state["ns_head.weight"]
    embed_dim = ns_head_weight.shape[1]
    depth = embed_dim // 64

    vae_local, hmar = build_vae_hmar(
        V=4096,
        Cvae=32,
        ch=160,
        share_quant_resi=4,
        device=device,
        patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),
        num_classes=6,
        depth=depth,
        shared_aln=False,
        attn_l2_norm=True,
        flash_if_available=True,
        fused_if_available=True,
        n_layers_train=4,
        using_block_sparse_attn=False,
    )

    vae_local.load_state_dict(torch.load(vae_ckpt_path, map_location="cpu"), strict=True)
    hmar.load_base_and_ns_state_dict(trainer_state)
    hmar.load_mask_dict(trainer_state)
    hmar.eval()
    return hmar


def generate_samples(hmar, out_dir: str, total_samples: int, batch_size: int, class_counts):
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    idx = 0
    with torch.inference_mode():
        for class_id, count in class_counts.items():
            remaining = count
            seed_base = class_id * 100000
            while remaining > 0:
                cur_bs = min(batch_size, remaining)
                imgs = hmar.generate(
                    cur_bs,
                    class_id,
                    g_seed=seed_base + idx,
                    num_samples=1,
                    top_k=1000,
                    top_p=0.99,
                    cfg=1.7,
                    more_smooth=False,
                    mask=True,
                    mask_schedule=get_args(cfg_folder="sample").mask_schedule,
                )
                for j in range(cur_bs):
                    torchvision.utils.save_image(imgs[j], os.path.join(out_dir, f"{idx:06d}.png"))
                    idx += 1
                remaining -= cur_bs

    assert idx == total_samples, f"generated {idx} samples, expected {total_samples}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="finetune checkpoint path")
    parser.add_argument("--vae_ckpt", default="vae_ch160v4096z32.pth")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--total_samples", type=int, required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument(
        "--class_counts",
        type=str,
        required=True,
        help='comma separated counts like "0:72,1:138,2:126,3:120,4:114,5:102"',
    )
    args = parser.parse_args()

    class_counts = {}
    for item in args.class_counts.split(","):
        cls, count = item.split(":")
        class_counts[int(cls)] = int(count)

    if sum(class_counts.values()) != args.total_samples:
        raise ValueError("sum(class_counts) must equal total_samples")

    device = "cuda"
    torch.set_default_device(device)
    hmar = build_hmar_from_finetune_ckpt(args.checkpoint, args.vae_ckpt, device=device)
    generate_samples(hmar, args.out_dir, args.total_samples, args.batch_size, class_counts)


if __name__ == "__main__":
    main()
