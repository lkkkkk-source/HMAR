import argparse
import os
import shutil

import torch
import torchvision
import yaml

from models import build_vae_hmar
from utils.arg_util import _get_yaml_loader
from utils.sampling_arg_util import Args


PATCH_NUMS = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)


def _load_public_hmar_weights(hmar, state_dict):
    current_state = hmar.state_dict()
    filtered_state = {}
    skipped = []

    for name, param in state_dict.items():
        if name not in current_state:
            continue
        if name == "class_emb.weight" and param.shape != current_state[name].shape:
            current_state[name][-1].copy_(param[-1])
            skipped.append((name, tuple(param.shape), tuple(current_state[name].shape)))
            continue
        if current_state[name].shape != param.shape:
            skipped.append((name, tuple(param.shape), tuple(current_state[name].shape)))
            continue
        filtered_state[name] = param

    ret = hmar.load_state_dict(filtered_state, strict=False)
    if skipped:
        print(f"[generate_finetune_samples] skipped public shape-mismatched keys: {skipped}")
    if ret is not None:
        missing, unexpected = ret
        print(f"[generate_finetune_samples] public HMAR missing: {missing}")
        print(f"[generate_finetune_samples] public HMAR unexpected: {unexpected}")


def _apply_finetuned_mask_weights(hmar, trainer_state):
    n_layers_train = len(hmar.mask_blocks)
    base_block_count = len(hmar.base_blocks)
    state = hmar.state_dict()

    for name, param in trainer_state.items():
        if name.startswith("blocks."):
            rest = name[len("blocks."):]
            block_idx_str, suffix = rest.split(".", 1)
            block_idx = int(block_idx_str)
            if block_idx >= base_block_count:
                mask_idx = block_idx - base_block_count
                if mask_idx < n_layers_train:
                    state[f"mask_blocks.{mask_idx}.{suffix}"].copy_(param)
        elif name.startswith("head_nm."):
            state["mask_head_nm." + name[len("head_nm."):]].copy_(param)
        elif name.startswith("head."):
            state["mask_head." + name[len("head."):]].copy_(param)
        elif name in {
            "word_embed.weight",
            "word_embed_bias",
            "mask_embed.weight",
            "pos_start",
            "pos_1LC",
        }:
            state[name].copy_(param)
        elif name.startswith("lvl_embed.") or name.startswith("shared_ada_lin."):
            state[name].copy_(param)
        elif name == "class_emb.weight":
            state[name].copy_(param)


def _load_full_hmar_checkpoint(hmar, trainer_state):
    ret = hmar.load_state_dict(trainer_state, strict=False)
    if ret is not None:
        missing, unexpected = ret
        print(f"[generate_finetune_samples] full HMAR missing: {missing}")
        print(f"[generate_finetune_samples] full HMAR unexpected: {unexpected}")


def build_hmar_from_finetune_ckpt(checkpoint_path: str, vae_ckpt_path: str, public_hmar_ckpt: str, device: str):
    finetune_ckpt = torch.load(checkpoint_path, map_location="cpu")
    trainer_state = finetune_ckpt["trainer"]["transformer_wo_ddp"]
    public_state = torch.load(public_hmar_ckpt, map_location="cpu")

    class_emb_weight = trainer_state["class_emb.weight"]
    num_classes = class_emb_weight.shape[0] - 1

    if "head.weight" in trainer_state:
        head_weight = trainer_state["head.weight"]
        checkpoint_style = "masked_prediction"
    elif "ns_head.weight" in trainer_state:
        head_weight = trainer_state["ns_head.weight"]
        checkpoint_style = "full_hmar"
    else:
        raise KeyError("Could not infer checkpoint style from trainer_state")

    embed_dim = head_weight.shape[1]
    depth = embed_dim // 64

    ns_block_ids = {
        int(k.split(".")[1])
        for k in public_state.keys()
        if k.startswith("ns_blocks.") and k.split(".")[1].isdigit()
    }
    if not ns_block_ids:
        raise ValueError("Could not infer HMAR n_layers_train from public checkpoint")
    n_layers_train = max(ns_block_ids) + 1

    vae_local, hmar = build_vae_hmar(
        V=4096,
        Cvae=32,
        ch=160,
        share_quant_resi=4,
        device=device,
        patch_nums=PATCH_NUMS,
        num_classes=num_classes,
        depth=depth,
        shared_aln=False,
        attn_l2_norm=True,
        flash_if_available=True,
        fused_if_available=True,
        n_layers_train=n_layers_train,
        using_block_sparse_attn=False,
    )

    vae_local.load_state_dict(torch.load(vae_ckpt_path, map_location="cpu"), strict=True)
    _load_public_hmar_weights(hmar, public_state)
    if checkpoint_style == "masked_prediction":
        _apply_finetuned_mask_weights(hmar, trainer_state)
    else:
        _load_full_hmar_checkpoint(hmar, trainer_state)
    hmar.eval()
    return hmar


def generate_samples(hmar, out_dir: str, total_samples: int, batch_size: int, class_counts, sample_args):
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
                    top_k=sample_args.top_k,
                    top_p=sample_args.top_p,
                    cfg=sample_args.cfg,
                    more_smooth=sample_args.more_smooth,
                    mask=sample_args.mask,
                    mask_schedule=sample_args.mask_schedule,
                )
                for j in range(cur_bs):
                    torchvision.utils.save_image(imgs[j], os.path.join(out_dir, f"{idx:06d}.png"))
                    idx += 1
                remaining -= cur_bs

    assert idx == total_samples, f"generated {idx} samples, expected {total_samples}"


def load_sampling_args(config_name: str):
    args = Args()
    loader = _get_yaml_loader()
    with open(f"config/sample/{config_name}.yaml", "r") as file:
        config = yaml.load(file, Loader=loader)
    for key, value in config.items():
        if hasattr(args, key):
            setattr(args, key, value)

    args.patch_nums = tuple(map(int, args.pn.replace("-", "_").split("_")))
    args.set_tf32(args.tf32)
    args.seed_everything(benchmark=True)
    return args


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="finetune checkpoint path")
    parser.add_argument("--public_hmar_ckpt", default="hmar-d16.pth")
    parser.add_argument("--sample_config", default="hmar-d16")
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

    sample_args = load_sampling_args(args.sample_config)
    device = "cuda"
    torch.set_default_device(device)
    hmar = build_hmar_from_finetune_ckpt(
        args.checkpoint,
        args.vae_ckpt,
        args.public_hmar_ckpt,
        device=device,
    )
    generate_samples(hmar, args.out_dir, args.total_samples, args.batch_size, class_counts, sample_args)


if __name__ == "__main__":
    main()
