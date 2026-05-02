import gc
import os
import shutil
import sys
import time
import warnings
from functools import partial

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

import dist
from dist import NullDDP
from hmar_trainer import HMARTrainer
from models import HMAR, VQVAE, build_vae_hmar
from utils import arg_util, misc
from utils.amp_sc import AmpOptimizer
from utils.data import build_dataset
from utils.data_sampler import DistInfiniteBatchSampler, EvalDistributedSampler
from utils.finetune_lr_control import filter_params, lr_wd_annealing
from utils.misc import delete_old_ckpts


def infer_public_hmar_n_layers(state_dict):
    ns_ids = {
        int(name.split(".")[1])
        for name in state_dict.keys()
        if name.startswith("ns_blocks.") and name.split(".")[1].isdigit()
    }
    if not ns_ids:
        raise ValueError("Could not infer HMAR architecture split from public checkpoint")
    return max(ns_ids) + 1


def load_public_hmar_weights(hmar_wo_ddp: HMAR, state_dict):
    current_state = hmar_wo_ddp.state_dict()
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

    ret = hmar_wo_ddp.load_state_dict(filtered_state, strict=False)
    if skipped:
        print(f"[load_public_hmar_weights] skipped shape-mismatched keys: {skipped}")
    if ret is not None:
        missing, unexpected = ret
        print(f"[load_public_hmar_weights] missing: {missing}")
        print(f"[load_public_hmar_weights] unexpected: {unexpected}")


def apply_last_k_finetune_policy(hmar_wo_ddp: HMAR, last_k: int):
    for _, param in hmar_wo_ddp.named_parameters():
        param.requires_grad = False

    for param in hmar_wo_ddp.class_emb.parameters():
        param.requires_grad = True
    for param in hmar_wo_ddp.mask_embed.parameters():
        param.requires_grad = True
    for param in hmar_wo_ddp.word_embed.parameters():
        param.requires_grad = True
    hmar_wo_ddp.word_embed_bias.requires_grad = True
    for module in (hmar_wo_ddp.ns_head_nm, hmar_wo_ddp.ns_head, hmar_wo_ddp.mask_head_nm, hmar_wo_ddp.mask_head):
        for param in module.parameters():
            param.requires_grad = True

    if last_k > len(hmar_wo_ddp.ns_blocks):
        raise ValueError(f"Requested last_k={last_k} but HMAR only has {len(hmar_wo_ddp.ns_blocks)} ns/mask blocks")

    for block in list(hmar_wo_ddp.ns_blocks)[-last_k:]:
        for param in block.parameters():
            param.requires_grad = True
    for block in list(hmar_wo_ddp.mask_blocks)[-last_k:]:
        for param in block.parameters():
            param.requires_grad = True


def build_everything(args: arg_util.Args):
    wdb_lg = misc.DistLogger(misc.WandbLogger(args), verbose=True)

    print(f"global bs={args.glb_batch_size}, local bs={args.batch_size}")
    print(f"initial args:\n{str(args)}")

    print(f"[build PT data] ...\n")
    num_classes, dataset_train, dataset_val = build_dataset(
        args.data_path, final_reso=args.data_load_reso, hflip=args.hflip, mid_reso=args.mid_reso
    )
    types = str((type(dataset_train).__name__, type(dataset_val).__name__))

    ld_val = DataLoader(
        dataset_val,
        num_workers=0,
        pin_memory=True,
        batch_size=round(args.batch_size * 1.5),
        sampler=EvalDistributedSampler(dataset_val, num_replicas=dist.get_world_size(), rank=dist.get_rank()),
        shuffle=False,
        drop_last=False,
    )
    del dataset_val

    ld_train = DataLoader(
        dataset=dataset_train,
        num_workers=args.workers,
        pin_memory=True,
        generator=args.get_different_generator_for_each_rank(),
        batch_sampler=DistInfiniteBatchSampler(
            dataset_len=len(dataset_train),
            glb_batch_size=args.glb_batch_size,
            same_seed_for_all_ranks=args.same_seed_for_all_ranks,
            shuffle=True,
            fill_last=True,
            rank=dist.get_rank(),
            world_size=dist.get_world_size(),
            start_ep=0,
            start_it=0,
        ),
    )
    del dataset_train

    print(f"[dataloader multi processing] ...", end="", flush=True)
    stt = time.time()
    iters_train = len(ld_train)
    ld_train = iter(ld_train)
    print(f"     [dataloader multi processing](*) finished! ({time.time()-stt:.2f}s)", flush=True, clean=True)
    print(f"[dataloader] gbs={args.glb_batch_size}, lbs={args.batch_size}, iters_train={iters_train}, types(tr, va)={types}")

    public_hmar_ckpt = os.path.join(args.shared_dir_path, "hmar-d16.pth")
    public_state = torch.load(public_hmar_ckpt, map_location="cpu")
    public_n_layers_train = infer_public_hmar_n_layers(public_state)

    vae_local, hmar_wo_ddp = build_vae_hmar(
        V=4096,
        Cvae=32,
        ch=160,
        share_quant_resi=4,
        device=dist.get_device(),
        patch_nums=args.patch_nums,
        num_classes=num_classes,
        depth=args.depth,
        shared_aln=args.saln,
        attn_l2_norm=args.anorm,
        flash_if_available=args.fuse,
        fused_if_available=args.fuse,
        n_layers_train=public_n_layers_train,
    )

    vae_ckpt = os.path.join(args.shared_dir_path, "vae_ch160v4096z32.pth")
    vae_local.load_state_dict(torch.load(vae_ckpt, map_location="cpu"), strict=True)
    load_public_hmar_weights(hmar_wo_ddp, public_state)
    apply_last_k_finetune_policy(hmar_wo_ddp, args.n_layers_train)

    vae_local: VQVAE = args.compile_model(vae_local, args.vfast)
    hmar_wo_ddp: HMAR = args.compile_model(hmar_wo_ddp, args.tfast)
    hmar: DDP = (DDP if dist.initialized() else NullDDP)(
        hmar_wo_ddp,
        device_ids=[dist.get_local_rank()],
        find_unused_parameters=False,
        broadcast_buffers=False,
    )

    names, paras, para_groups = filter_params(
        hmar_wo_ddp,
        nowd_keys={
            "cls_token",
            "start_token",
            "task_token",
            "cfg_uncond",
            "pos_embed",
            "pos_1LC",
            "pos_start",
            "start_pos",
            "lvl_embed",
            "gamma",
            "beta",
            "ada_gss",
            "moe_bias",
            "scale_mul",
        },
    )
    opt_clz = {
        "adam": partial(torch.optim.AdamW, betas=(0.9, 0.95), fused=args.afuse),
        "adamw": partial(torch.optim.AdamW, betas=(0.9, 0.95), fused=args.afuse),
    }[args.opt.lower().strip()]
    opt_kw = dict(lr=args.tlr, weight_decay=0)
    optimizer = AmpOptimizer(
        mixed_precision=args.fp16,
        optimizer=opt_clz(params=para_groups, **opt_kw),
        names=names,
        paras=paras,
        grad_clip=args.tclip,
        n_gradient_accumulation=args.ac,
    )
    del names, paras, para_groups

    trainer = HMARTrainer(
        device=args.device,
        patch_nums=args.patch_nums,
        resos=args.resos,
        vae_local=vae_local,
        hmar_wo_ddp=hmar_wo_ddp,
        hmar=hmar,
        optimizer=optimizer,
        label_smooth=args.ls,
        reweight_loss=args.reweight_loss,
        loss_reweight_type=args.loss_reweight_type,
    )

    dist.barrier()
    return wdb_lg, trainer, iters_train, ld_train, ld_val


def train_one_ep(ep, args, wdb_lg, ld_or_itrt, iters_train, trainer):
    me = misc.MetricLogger(delimiter="  ")
    me.add_meter("tlr", misc.SmoothedValue(window_size=1, fmt="{value:.2g}"))
    me.add_meter("tnm", misc.SmoothedValue(window_size=1, fmt="{value:.2f}"))
    [me.add_meter(x, misc.SmoothedValue(fmt="{median:.3f} ({global_avg:.3f})")) for x in ["Lm", "Ltail"]]
    [me.add_meter(x, misc.SmoothedValue(fmt="{median:.2f} ({global_avg:.2f})")) for x in ["Accm", "Acct"]]
    header = f"[Ep]: [{ep:4d}/{args.ep}]"

    g_it, max_it = ep * iters_train, args.ep * iters_train
    print(f"  [*] [training]  {header}  [it]: [1/{iters_train}]", flush=True)
    for it, (inp, label) in me.log_every(0, iters_train, ld_or_itrt, 5, header):
        g_it = ep * iters_train + it
        inp = inp.to(args.device, non_blocking=True)
        label = label.to(args.device, non_blocking=True)
        args.cur_it = f"{it+1}/{iters_train}"

        wp_it = args.wp * iters_train
        min_tlr, max_tlr, min_twd, max_twd = lr_wd_annealing(
            args.sche, trainer.optimizer.optimizer, args.tlr, args.twd, args.twde, g_it, wp_it, max_it, wp0=args.wp0, wpe=args.wpe
        )
        args.cur_lr, args.cur_wd = max_tlr, max_twd
        stepping = (g_it + 1) % args.ac == 0
        grad_norm, scale_log2 = trainer.train_step(
            it=it,
            g_it=g_it,
            stepping=stepping,
            metric_lg=me,
            wdb_lg=wdb_lg,
            inp_B3HW=inp,
            label_B=label,
            eval_labels=args.eval_classes,
            log_imgs_iters=args.log_imgs_iters,
        )
        me.update(tlr=max_tlr)
        wdb_lg.set_step(step=g_it)
        wdb_lg.update(head="Optimizer/lr_max", sche_tlr=max_tlr)
        wdb_lg.update(head="Optimizer/wd_max", sche_twd=max_twd)
        wdb_lg.update(head="Optimizer/fp16", scale_log2=scale_log2)
        if args.tclip > 0:
            wdb_lg.update(head="Optimizer/grad", grad_norm=grad_norm)
            wdb_lg.update(head="Optimizer/grad", grad_clip=args.tclip)

    me.synchronize_between_processes()
    return {k: meter.global_avg for k, meter in me.meters.items()}, me.iter_time.time_preds(max_it - (g_it + 1) + (args.ep - ep) * 15)


def finetune_hmar():
    args: arg_util.Args = arg_util.init_dist_and_get_args()
    wdb_lg, trainer, iters_train, ld_train, ld_val = build_everything(args)

    start_time = time.time()
    for ep in range(args.ep):
        wdb_lg.set_step(ep * iters_train)
        stats, (_, remain_time, finish_time) = train_one_ep(ep, args, wdb_lg, ld_train, iters_train, trainer)
        args.L_mean, args.L_tail, args.acc_mean, args.acc_tail, args.grad_norm = (
            stats["Lm"],
            stats["Ltail"],
            stats["Accm"],
            stats["Acct"],
            stats["tnm"],
        )
        args.cur_ep = f"{ep+1}/{args.ep}"
        args.remain_time, args.finish_time = remain_time, finish_time

        if (ep + 1) % args.checkpoint_frequency == 0 or (ep + 1) == args.ep:
            val_loss_mean, val_loss_tail, val_acc_mean, val_acc_tail, val_loss_resos, acc_resos, tot, cost = trainer.eval_ep(ld_val)
            args.vL_mean, args.vL_tail, args.vacc_mean, args.vacc_tail = (
                val_loss_mean,
                val_loss_tail,
                val_acc_mean,
                val_acc_tail,
            )
            if dist.is_master():
                out_ckpt = os.path.join(args.experiment_dir_path, "ar-ckpt-last.pth")
                torch.save(
                    {"epoch": ep + 1, "iter": 0, "trainer": trainer.state_dict(), "args": args.state_dict()},
                    out_ckpt,
                )
                delete_old_ckpts(args.experiment_dir_path, "ar-ckpt-epoch-*.pth", args.max_num_checkpoints)
            dist.barrier()

        args.dump_log()
        wdb_lg.flush()

    print(f"[HMAR finetune finished] Total cost: {(time.time() - start_time) / 3600:.1f}h")
    wdb_lg.close()


if __name__ == "__main__":
    try:
        finetune_hmar()
    finally:
        dist.finalize()
        if isinstance(sys.stdout, misc.SyncPrint) and isinstance(sys.stderr, misc.SyncPrint):
            sys.stdout.close(), sys.stderr.close()
