# Copyright (c) 2025, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the NVIDIA One-Way Noncommercial License v1 (NSCLv1).
# To view a copy of this license, please refer to LICENSE

import gc
import os
import shutil
import sys
import time
import warnings
from functools import partial

import torch
from torch.utils.data import DataLoader

import dist
from dist import NullDDP
from utils import arg_util, misc
from utils.data import build_dataset
from utils.data_sampler import DistInfiniteBatchSampler, EvalDistributedSampler
from utils.misc import auto_resume_finetune, delete_old_ckpts

from torch.nn.parallel import DistributedDataParallel as DDP
from models import MaskedPrediction, VQVAE, build_vae_mp
from utils.amp_sc import AmpOptimizer
from utils.finetune_lr_control import filter_params
from mp_trainer import MaskTrainer
from utils.finetune_lr_control import lr_wd_annealing


def build_everything(args: arg_util.Args):
    # resume
    auto_resume_info, start_ep, start_it, base_ckpt_state, finetune_state, args_state = auto_resume_finetune(args, '*.pth')
    # create wandb logger
    wdb_lg: misc.WandbLogger

    wdb_lg = misc.DistLogger(misc.WandbLogger(args), verbose=True)

    # log args
    print(f'global bs={args.glb_batch_size}, local bs={args.batch_size}')
    print(f'initial args:\n{str(args)}')
    
    print(f'[build PT data] ...\n')
    num_classes, dataset_train, dataset_val = build_dataset(
        args.data_path, final_reso=args.data_load_reso, hflip=args.hflip, mid_reso=args.mid_reso,
    )
    types = str((type(dataset_train).__name__, type(dataset_val).__name__))
    
    ld_val = DataLoader(
        dataset_val, num_workers=0, pin_memory=True,
        batch_size=round(args.batch_size*1.5), sampler=EvalDistributedSampler(dataset_val, num_replicas=dist.get_world_size(), rank=dist.get_rank()),
        shuffle=False, drop_last=False,
    )
    del dataset_val
    
    ld_train = DataLoader(
        dataset=dataset_train, num_workers=args.workers, pin_memory=True,
        generator=args.get_different_generator_for_each_rank(), # worker_init_fn=worker_init_fn,
        batch_sampler=DistInfiniteBatchSampler(
            dataset_len=len(dataset_train), glb_batch_size=args.glb_batch_size, same_seed_for_all_ranks=args.same_seed_for_all_ranks,
            shuffle=True, fill_last=True, rank=dist.get_rank(), world_size=dist.get_world_size(), start_ep=start_ep, start_it=start_it,
        ),
    )
    del dataset_train
    
    [print(line) for line in auto_resume_info]
    print(f'[dataloader multi processing] ...', end='', flush=True)
    stt = time.time()
    iters_train = len(ld_train)
    ld_train = iter(ld_train)
    # noinspection PyArgumentList
    print(f'     [dataloader multi processing](*) finished! ({time.time()-stt:.2f}s)', flush=True, clean=True)
    print(f'[dataloader] gbs={args.glb_batch_size}, lbs={args.batch_size}, iters_train={iters_train}, types(tr, va)={types}')
    
    vae_local, mp_wo_ddp = build_vae_mp(
        V=4096, Cvae=32, ch=160, share_quant_resi=4,        # hard-coded VQVAE hyperparameters
        device=dist.get_device(), patch_nums=args.patch_nums,
        num_classes=num_classes, depth=args.depth, shared_aln=args.saln, attn_l2_norm=args.anorm,
        flash_if_available=args.fuse, fused_if_available=args.fuse,
        init_adaln=args.aln, init_adaln_gamma=args.alng, init_head=args.hd, init_std=args.ini,
        n_layers_train=args.n_layers_train,
        using_block_sparse_attn=False,
    )
    
    vae_ckpt = os.path.join(args.shared_dir_path, 'vae_ch160v4096z32.pth')
    #check if this file exists and if not download it from online into the shared directory
    if dist.is_master() and not os.path.exists(vae_ckpt):
        print(f'File {vae_ckpt} does not exist. Downloading it from online')
        os.system(f'wget https://huggingface.co/FoundationVision/var/resolve/main/vae_ch160v4096z32.pth -P {args.shared_dir_path}')
        
    dist.barrier()
    vae_local.load_state_dict(torch.load(vae_ckpt, map_location='cpu'), strict=True)
    
    vae_local: VQVAE = args.compile_model(vae_local, args.vfast)
    mp_wo_ddp: MaskedPrediction = args.compile_model(mp_wo_ddp, args.tfast) #This should be VMAR
    mp: DDP = (DDP if dist.initialized() else NullDDP)(mp_wo_ddp, device_ids=[dist.get_local_rank()], find_unused_parameters=False, broadcast_buffers=False)
    
    print(f'[INIT] MaskedPrediction model = {mp_wo_ddp}\n\n')
    count_p = lambda m: f'{sum(p.numel() for p in m.parameters())/1e6:.2f}'
    print(f'[INIT][#para] ' + ', '.join([f'{k}={count_p(m)}' for k, m in (('VAE', vae_local), ('VAE.enc', vae_local.encoder), ('VAE.dec', vae_local.decoder), ('VAE.quant', vae_local.quantize))]))
    print(f'[INIT][#para] ' + ', '.join([f'{k}={count_p(m)}' for k, m in (('MaskedPrediction', mp_wo_ddp),)]) + '\n\n')
    
    
    #TODO: determine what parameters are being filtered out here and if to add the masking
    # build optimizer
    names, paras, para_groups = filter_params(mp_wo_ddp, nowd_keys={
        'cls_token', 'start_token', 'task_token', 'cfg_uncond',
        'pos_embed', 'pos_1LC', 'pos_start', 'start_pos', 'lvl_embed',
        'gamma', 'beta',
        'ada_gss', 'moe_bias',
        'scale_mul',
    })
    opt_clz = {
        'adam':  partial(torch.optim.AdamW, betas=(0.9, 0.95), fused=args.afuse),
        'adamw': partial(torch.optim.AdamW, betas=(0.9, 0.95), fused=args.afuse),
    }[args.opt.lower().strip()]
    opt_kw = dict(lr=args.tlr, weight_decay=0)
    print(f'[INIT] optim={opt_clz}, opt_kw={opt_kw}\n')
    
    optimizer = AmpOptimizer(
        mixed_precision=args.fp16, optimizer=opt_clz(params=para_groups, **opt_kw), names=names, paras=paras,
        grad_clip=args.tclip, n_gradient_accumulation=args.ac
    )
    del names, paras, para_groups
    
    trainer = MaskTrainer(
        device=args.device, patch_nums=args.patch_nums, resos=args.resos,
        vae_local=vae_local, mp_wo_ddp=mp_wo_ddp, mp=mp,
        optimizer=optimizer, label_smooth=args.ls,
        reweight_loss=args.reweight_loss,
        loss_reweight_type=args.loss_reweight_type,
    )
    
    #TODO: Check here how this should be done for the mask trainer
    if base_ckpt_state is not None and len(base_ckpt_state):
        trainer.load_state_dict(base_ckpt_state, finetune_state, strict=False, skip_vae=True) # don't load vae again
    del vae_local, mp_wo_ddp, mp, optimizer
    
    #print the number of trainable parameters
    total_params = sum(p.numel() for p in trainer.transformer_wo_ddp.parameters() if p.requires_grad)
    print(f'[INIT] Total trainable parameters: {total_params}')
    dist.barrier()
    return (
        wdb_lg, trainer, start_ep, start_it,
        iters_train, ld_train, ld_val
    )


def finetune():
    args: arg_util.Args = arg_util.init_dist_and_get_args()
    
    (   
        wdb_lg, trainer,
        start_ep, start_it,
        iters_train, ld_train, ld_val
    ) = build_everything(args)
    

    # train
    start_time = time.time()
    best_L_mean, best_L_tail, best_acc_mean, best_acc_tail = 999., 999., -1., -1.
    best_val_loss_mean, best_val_loss_tail, best_val_acc_mean, best_val_acc_tail = 999, 999, -1, -1
    
    L_mean, L_tail = -1, -1
    for ep in range(start_ep, args.ep):
        if hasattr(ld_train, 'sampler') and hasattr(ld_train.sampler, 'set_epoch'):
            ld_train.sampler.set_epoch(ep)
        wdb_lg.set_step(ep * iters_train)
        
        stats, (sec, remain_time, finish_time) = train_one_ep(
            ep, ep == start_ep, start_it if ep == start_ep else 0, args, wdb_lg, ld_train, iters_train, trainer
        )
        
        L_mean, L_tail, acc_mean, acc_tail, grad_norm = stats['Lm'], stats['Lt'], stats['Accm'], stats['Acct'], stats['tnm']
        best_L_mean, best_acc_mean = min(best_L_mean, L_mean), max(best_acc_mean, acc_mean)
        if L_tail != -1: best_L_tail, best_acc_tail = min(best_L_tail, L_tail), max(best_acc_tail, acc_tail)
        args.L_mean, args.L_tail, args.acc_mean, args.acc_tail, args.grad_norm = L_mean, L_tail, acc_mean, acc_tail, grad_norm
        args.cur_ep = f'{ep+1}/{args.ep}'
        args.remain_time, args.finish_time = remain_time, finish_time
        
        log_ckpt_to_wandb = ep == start_ep or (ep + 1) % args.log_ckpt_to_wandb_every == 0 or (ep + 1) == args.ep or (ep + 1) == 1
        is_val_and_also_saving = (ep + 1) % args.checkpoint_frequency == 0 or log_ckpt_to_wandb or (ep + 1) == args.ep
        
        if is_val_and_also_saving:
            val_loss_mean, val_loss_tail, val_acc_mean, val_acc_tail, val_loss_resos, acc_resos, tot, cost = trainer.eval_ep(ld_val)
            best_updated = best_val_loss_tail > val_loss_tail
            best_val_loss_mean, best_val_loss_tail = min(best_val_loss_mean, val_loss_mean), min(best_val_loss_tail, val_loss_tail)
            best_val_acc_mean, best_val_acc_tail = max(best_val_acc_mean, val_acc_mean), max(best_val_acc_tail, val_acc_tail)
            val_loss_and_acc = dict(
                vL_mean=val_loss_mean, vL_tail=val_loss_tail, vacc_mean=val_acc_mean, vacc_tail=val_acc_tail,
            )
            
            args.vL_mean, args.vL_tail, args.vacc_mean, args.vacc_tail = val_loss_mean, val_loss_tail, val_acc_mean, val_acc_tail
            print(f' [*] [ep{ep}]  (val {tot})  Lm: {L_mean:.4f}, Lt: {L_tail:.4f}, Acc m&t: {acc_mean:.2f} {acc_tail:.2f},  Val cost: {cost:.2f}s')
            
            wdb_lg.update(head='Validation Loss & Accuracy', step=(ep + 1) * iters_train, **val_loss_and_acc)
            
            kw_loss = {f'vL_{pn}x{pn}': val_loss_resos[i] for i, pn in enumerate(args.patch_nums)}
            kw_acc = {f'vacc_{pn}x{pn}': acc_resos[i] for i, pn in enumerate(args.patch_nums)}
            
            wdb_lg.update(head='Validation Loss & Accuracy', step=(ep + 1) * iters_train, **kw_loss, **kw_acc)
            
            if dist.is_master():
                out_ckpt = os.path.join(args.experiment_dir_path, f'ar-ckpt-last.pth')
                out_ckpt_best = os.path.join(args.experiment_dir_path, 'ar-ckpt-best.pth')
                print(f'[saving ckpt] ...', end='', flush=True)
                torch.save({
                    'epoch':    ep+1,
                    'iter':     0,
                    'trainer':  trainer.state_dict(),
                    'args':     args.state_dict(),
                }, out_ckpt)
                if best_updated:
                    shutil.copy(out_ckpt, out_ckpt_best)
                print(f'     [saving ckpt](*) finished!  @ {out_ckpt}', flush=True, clean=True)
                
                if log_ckpt_to_wandb:
                    out_wandb_ckpt = os.path.join(args.experiment_dir_path, f'ar-ckpt-epoch-{ep+1}.pth')
                    shutil.copy(out_ckpt, out_wandb_ckpt)
                    wdb_lg.log_file(out_wandb_ckpt)  
                    
                    log_files = os.path.join(args.experiment_dir_path, f'*.txt')  
                    wdb_lg.log_file(log_files, policy='live')     
                      
                delete_old_ckpts(args.experiment_dir_path, 'ar-ckpt-epoch-*.pth', args.max_num_checkpoints)
            dist.barrier()
        
        print(    f'     [ep{ep}]  (training )  Lm: {best_L_mean:.3f} ({L_mean:.3f}), Lt: {best_L_tail:.3f} ({L_tail:.3f}),  Acc m&t: {best_acc_mean:.2f} {best_acc_tail:.2f},  Remain: {remain_time},  Finish: {finish_time}', flush=True)
        
        args.dump_log(); wdb_lg.flush()
    
    total_time = f'{(time.time() - start_time) / 60 / 60:.1f}h'
    print('\n\n')
    print(f'  [*] [PT finished]  Total cost: {total_time},   Lm: {best_L_mean:.3f} ({L_mean}),   Lt: {best_L_tail:.3f} ({L_tail})')
    print('\n\n')
    
    del stats
    del iters_train, ld_train
    time.sleep(300), gc.collect(), torch.cuda.empty_cache(), time.sleep(300)
    
    args.remain_time, args.finish_time = '-', time.strftime("%Y-%m-%d %H:%M", time.localtime(time.time() - 60))
    print(f'final args:\n\n{str(args)}')
    args.dump_log(); wdb_lg.flush()
    dist.barrier()  
    wdb_lg.log_file(log_files, policy='now')
    wdb_lg.flush()
    time.sleep(300)
    wdb_lg.close()
    dist.barrier()


def train_one_ep(ep: int, is_first_ep: bool, start_it: int, args: arg_util.Args, wdb_lg: misc.WandbLogger, ld_or_itrt, iters_train: int, trainer : MaskTrainer):
    step_cnt = 0
    me = misc.MetricLogger(delimiter='  ')
    me.add_meter('tlr', misc.SmoothedValue(window_size=1, fmt='{value:.2g}'))
    me.add_meter('tnm', misc.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    [me.add_meter(x, misc.SmoothedValue(fmt='{median:.3f} ({global_avg:.3f})')) for x in ['Lm', 'Lt']]
    [me.add_meter(x, misc.SmoothedValue(fmt='{median:.2f} ({global_avg:.2f})')) for x in ['Accm', 'Acct']]
    header = f'[Ep]: [{ep:4d}/{args.ep}]'
    
    if is_first_ep:
        warnings.filterwarnings('ignore', category=DeprecationWarning)
        warnings.filterwarnings('ignore', category=UserWarning)
    g_it, max_it = ep * iters_train, args.ep * iters_train
    
    print(f'  [*] [training]  {header}  [it]: [{start_it+1}/{iters_train}]', flush=True)
    for it, (inp, label) in me.log_every(start_it, iters_train, ld_or_itrt, 30 if iters_train > 8000 else 5, header):
        g_it = ep * iters_train + it
        if it < start_it: continue
        if is_first_ep and it == start_it: warnings.resetwarnings()
        
        inp = inp.to(args.device, non_blocking=True)
        label = label.to(args.device, non_blocking=True)
        
        args.cur_it = f'{it+1}/{iters_train}'
        
        wp_it = args.wp * iters_train
        min_tlr, max_tlr, min_twd, max_twd = lr_wd_annealing(args.sche, trainer.optimizer.optimizer, args.tlr, args.twd, args.twde, g_it, wp_it, max_it, wp0=args.wp0, wpe=args.wpe)
        args.cur_lr, args.cur_wd = max_tlr, max_twd
        
        stepping = (g_it + 1) % args.ac == 0
        step_cnt += int(stepping)
        
        grad_norm, scale_log2 = trainer.train_step(
            it=it, g_it=g_it, stepping=stepping, metric_lg=me, wdb_lg=wdb_lg,
            inp_B3HW=inp, label_B=label,
            eval_labels=args.eval_classes,
            log_imgs_iters=args.log_imgs_iters,
        )

        me.update(tlr=max_tlr)
        wdb_lg.set_step(step=g_it)
        wdb_lg.update(head='Optimizer/lr_min', sche_tlr=min_tlr)
        wdb_lg.update(head='Optimizer/lr_max', sche_tlr=max_tlr)
        wdb_lg.update(head='Optimizer/wd_max', sche_twd=max_twd)
        wdb_lg.update(head='Optimizer/wd_min', sche_twd=min_twd)
        wdb_lg.update(head='Optimizer/fp16', scale_log2=scale_log2)
        
        if args.tclip > 0:
            wdb_lg.update(head='Optimizer/grad', grad_norm=grad_norm)
            wdb_lg.update(head='Optimizer/grad', grad_clip=args.tclip)
    
    me.synchronize_between_processes()
    return {k: meter.global_avg for k, meter in me.meters.items()}, me.iter_time.time_preds(max_it - (g_it + 1) + (args.ep - ep) * 15)  

if __name__ == '__main__':
    try: finetune()
    finally:
        dist.finalize()
        if isinstance(sys.stdout, misc.SyncPrint) and isinstance(sys.stderr, misc.SyncPrint):
            sys.stdout.close(), sys.stderr.close()
