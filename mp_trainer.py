# Copyright (c) 2025, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the NVIDIA One-Way Noncommercial License v1 (NSCLv1).
# To view a copy of this license, please refer to LICENSE

import time
from typing import List, Optional, Tuple, Union

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

import dist
from models import MaskedPrediction, VQVAE
from utils.amp_sc import AmpOptimizer
from utils.misc import MetricLogger, WandbLogger
from trainer import Trainer
import random

Ten = torch.Tensor
FTen = torch.Tensor
ITen = torch.LongTensor
BTen = torch.BoolTensor


class MaskTrainer(Trainer):
    def __init__(
        self, device, patch_nums: Tuple[int, ...], resos: Tuple[int, ...],
        vae_local: VQVAE, mp_wo_ddp: MaskedPrediction, mp: DDP,
        optimizer: AmpOptimizer, label_smooth: float, reweight_loss: bool = False,
        loss_reweight_type: str = 'mask_unweighted',
    ):
        super(MaskTrainer, self).__init__(
            device, patch_nums, resos, vae_local, mp_wo_ddp, mp, optimizer, label_smooth, reweight_loss, loss_reweight_type
        )
        
        self.loss_reweight_type = loss_reweight_type    

    @torch.no_grad()
    def eval_ep(self, ld_val: DataLoader, p_mask: float = 0.5):
        tot = 0
        L_mean, L_tail, acc_mean, acc_tail = 0, 0, 0, 0
        L_resos = [0] * len(self.resos)
        acc_resos = [0] * len(self.resos)
        
        stt = time.time()
        
        training = self.transformer_wo_ddp.training
        self.transformer_wo_ddp.eval()
        for inp_B3HW, label_B in ld_val:
            B, V = label_B.shape[0], self.vae_local.vocab_size
            inp_B3HW = inp_B3HW.to(dist.get_device(), non_blocking=True)
            label_B = label_B.to(dist.get_device(), non_blocking=True)
            
            gt_idx_Bl: List[ITen] = self.vae_local.img_to_idxBl(inp_B3HW)
            gt_BL = torch.cat(gt_idx_Bl, dim=1)
            x_ns = self.quantize_local.idxBl_to_ns_input(gt_idx_Bl)
            x_mask, idx_to_mask_list = self.quantize_local.idxBl_to_mask_input(gt_idx_Bl, p_mask=p_mask)
            x = torch.cat([x_ns, x_mask], dim=0)

            idx_to_mask = torch.cat(idx_to_mask_list, dim=0)
            
            with torch.autocast('cuda', enabled=True, cache_enabled=True):
                logits_BLV = self.transformer_wo_ddp(label_B, x, idx_to_mask)
            
            idx_to_mask_list_plus_1 = [idx + 1 for idx in idx_to_mask_list]
            idx_to_mask_plus_1 = idx_to_mask + 1
            
            L_mean += self.val_loss(logits_BLV[:, idx_to_mask_plus_1, :].view(-1, V), gt_BL[:, idx_to_mask_plus_1].view(-1)) * B
            L_tail += self.val_loss(logits_BLV.data[:, idx_to_mask_list_plus_1[-1], :].reshape(-1, V), gt_BL[:, idx_to_mask_list_plus_1[-1]].reshape(-1)) * B
            
            acc_mean += (logits_BLV.data[:, idx_to_mask_plus_1].argmax(dim=-1) == gt_BL[:, idx_to_mask_plus_1]).sum() * (100 / idx_to_mask_plus_1.shape[0]) 

            acc_tail += (logits_BLV.data[:, idx_to_mask_list_plus_1[-1]].argmax(dim=-1) == gt_BL[:, idx_to_mask_list_plus_1[-1]]).sum() * (100 / idx_to_mask_list_plus_1[-1].shape[0])

            idx_to_mask_list_plus_1 = [torch.tensor([0], device=dist.get_device(), dtype=torch.long)] + idx_to_mask_list_plus_1
            for si, (bg, ed) in enumerate(self.begin_ends):
                L_resos[si] += self.val_loss(logits_BLV.data[:, idx_to_mask_list_plus_1[si], :].reshape(-1, V), gt_BL[:, idx_to_mask_list_plus_1[si]].reshape(-1)) * B
                acc_resos[si] += (logits_BLV.data[:, idx_to_mask_list_plus_1[si]].argmax(dim=-1) == gt_BL[:, idx_to_mask_list_plus_1[si]]).sum() * (100 / (ed - bg))
            tot += B
            
        self.transformer_wo_ddp.train(training)
        
        stats = L_mean.new_tensor(L_resos + acc_resos + [L_mean.item(), L_tail.item(), acc_mean.item(), acc_tail.item(), tot])
        dist.allreduce(stats)
        tot = round(stats[-1].item())
        stats /= tot
        
        L_mean, L_tail, acc_mean, acc_tail, _ = stats.tolist()[len(self.resos*2):]
        L_resos = stats.tolist()[:len(self.resos)]
        acc_resos = stats.tolist()[len(self.resos):len(self.resos*2)]
        return L_mean, L_tail, acc_mean, acc_tail, L_resos, acc_resos, tot, time.time()-stt
    
    def train_step(
        self, it: int, g_it: int, stepping: bool, metric_lg: MetricLogger, wdb_lg: WandbLogger,
        inp_B3HW: FTen, label_B: Union[ITen, FTen], eval_labels: List[int], log_imgs_iters: int,
    ) -> Tuple[Optional[Union[Ten, float]], Optional[float]]:

        B, V = label_B.shape[0], self.vae_local.vocab_size
        self.transformer.require_backward_grad_sync = stepping
        
        gt_idx_Bl: List[ITen] = self.vae_local.img_to_idxBl(inp_B3HW)
        gt_BL = torch.cat(gt_idx_Bl, dim=1)
        
        x_ns = self.quantize_local.idxBl_to_ns_input(gt_idx_Bl)
        p_mask = random.random()
        x_mask, idx_to_mask_list = self.quantize_local.idxBl_to_mask_input(gt_idx_Bl, p_mask=p_mask)
        x = torch.cat([x_ns, x_mask], dim=0)

        idx_to_mask_list_plus_1 = [idx + 1 for idx in idx_to_mask_list]
        idx_to_mask = torch.cat(idx_to_mask_list, dim=0)
        idx_to_mask_plus_1 = idx_to_mask + 1
        
        with self.optimizer.amp_ctx:
            with torch.autocast('cuda', enabled=True, cache_enabled=True):
                logits_BLV = self.transformer(label_B, x, idx_to_mask)
            loss = self.train_loss(logits_BLV[:, idx_to_mask_plus_1, :].view(-1, V), gt_BL[:, idx_to_mask_plus_1].view(-1)).view(B, -1) 
            if self.loss_reweight_type == 'mask_unweighted':
                L = idx_to_mask.shape[0]
                loss_weight = torch.ones(1, L, device=dist.get_device()) / L
            else:
                loss_weight = self.loss_weight[:, idx_to_mask]
            loss = loss.mul(loss_weight).sum(dim=-1).mean()
        
        # backward
        grad_norm, scale_log2 = self.optimizer.backward_clip_step(loss=loss, stepping=stepping)
        
        # log
        pred_BL = logits_BLV.data.argmax(dim=-1)
        if it == 0 or it in metric_lg.log_iters:
            Lmean = self.val_loss(logits_BLV[:, idx_to_mask_plus_1, :].data.view(-1, V), gt_BL[:, idx_to_mask_plus_1].view(-1)).item()
            acc_mean = (pred_BL[:, idx_to_mask_plus_1] == gt_BL[:, idx_to_mask_plus_1]).float().mean().item() * 100

            Ltail = self.val_loss(logits_BLV.data[:, idx_to_mask_list_plus_1[-1], :].reshape(-1, V), gt_BL[:, idx_to_mask_list_plus_1[-1]].reshape(-1)).item()
            acc_tail = (pred_BL[:, idx_to_mask_list_plus_1[-1]] == gt_BL[:, idx_to_mask_list_plus_1[-1]]).float().mean().item() * 100

            grad_norm = grad_norm.item() if grad_norm is not None else 0.0
            metric_lg.update(Lm=Lmean, Ltail=Ltail, Accm=acc_mean, Acct=acc_tail, tnm=grad_norm)
        
        # log to wandb
        if g_it == 0 or (g_it + 1) % 500 == 0:
            if dist.is_master():
                kw = {}

                tce = self.val_loss(logits_BLV[:, idx_to_mask_plus_1, :].data.view(-1, V), gt_BL[:, idx_to_mask_plus_1].view(-1)).item()
                tacc = (pred_BL[:, idx_to_mask_plus_1] == gt_BL[:, idx_to_mask_plus_1]).float().mean().item() * 100

                wdb_lg.update(head='Training Loss & Accuracy', **{'Total Loss': tce, 'Total Accuracy': tacc}, step=g_it)
    
                idx_to_mask_list_plus_1 = [torch.tensor([0], device=dist.get_device(), dtype=torch.long)] + idx_to_mask_list_plus_1
                
                for si, (_, _) in enumerate(self.begin_ends):
                    pred, tar = logits_BLV.data[:, idx_to_mask_list_plus_1[si], :].reshape(-1, V), gt_BL[:, idx_to_mask_list_plus_1[si]].reshape(-1)
                    acc = (pred.argmax(dim=-1) == tar).float().mean().item() * 100
                    ce = self.val_loss(pred, tar).item()
                    kw[f'L_{self.patch_nums[si]}x{self.patch_nums[si]}'] = ce
                    kw[f'acc_{self.patch_nums[si]}x{self.patch_nums[si]}'] = acc

                wdb_lg.update(head='Masking Ratio', **{'Masking': p_mask}, step=g_it)
                wdb_lg.update(head='Resolution Training Loss & Accuracy', **kw, step=g_it)
                
                if wdb_lg.initialized() and g_it == 0 or (g_it + 1) % log_imgs_iters == 0:
                    #visualize image reconstruction
                    n_images = min(8, B)
                    x_mask, idx_to_mask_list = self.quantize_local.idxBl_to_mask_input(gt_idx_Bl, p_mask=0.5)
                    x = torch.cat([x_ns, x_mask], dim=0)

                    idx_to_mask = torch.cat(idx_to_mask_list, dim=0)
                    idx_to_mask_plus_1 = idx_to_mask + 1

                    #visualize the original images, take only the first 8 of them 
                    orig_imgs = inp_B3HW[:n_images, ...]

                    #visualize the images reconstructed by the VQVAE these tell us the upper bound of the construction/generation
                    recon_imgs = self.vae_local.img_to_reconstructed_img(inp_B3HW, last_one=True)[:n_images]

                    #Need to take the logits again because we have already done a backward pass, so need to use the newest weights
                    with torch.no_grad():
                        with torch.autocast('cuda', enabled=True, cache_enabled=True):
                            logits_BLV = self.transformer(label_B, x, idx_to_mask)
                    pred_BL = logits_BLV.data.argmax(dim=-1)

                    #visualize the image gotten by taking the predictions for the next scale
                    pred_BL[:, 0] = gt_BL[:, 0]  #make this deterministic by having the first token be the same as the original
                    ns_imgs = self.vae_local.idxBL_to_fhat_or_img(pred_BL, last_only=True, to_img=True)[:n_images]

                    #visualize the images gotten by taking the predictions from masking
                    tmp = gt_BL.clone()
                    tmp[:, idx_to_mask_plus_1] = pred_BL[:, idx_to_mask_plus_1]
                    mask_imgs = self.vae_local.idxBL_to_fhat_or_img(tmp, last_only=True, to_img=True)[:n_images]
                
                    #combine the images from the reconstruction catetory into a single tensor for viewing
                    imgs = torch.cat([orig_imgs, recon_imgs, ns_imgs, mask_imgs], dim=0)

                    wdb_lg.log_images('Visualization/Reconstruction', imgs, nrow=n_images, step=g_it)
                        
        return grad_norm, scale_log2
    
    def load_state_dict(self, base_ckpt_state, finetune_state, strict=True, skip_vae=False):
        #load the base checkpoint into the model 
        if not finetune_state:
            m = getattr(self, 'transformer_wo_ddp')
            if m is not None:
                if hasattr(m, '_orig_mod'):
                    m = m._orig_mod
                ret = getattr(self, 'transformer_wo_ddp').load_state_dict_with_word_embed(base_ckpt_state, strict=strict)
                if ret is not None:
                    missing, unexpected = ret
                    print(f'[MaskTrainer.load_state_dict] transformer_wo_ddp missing:  {missing}')
                    print(f'[MaskTrainer.load_state_dict] transformer_wo_ddp unexpected:  {unexpected}')
        else:
            for k in ('transformer_wo_ddp', 'vae_local', 'optimizer'):
                if skip_vae and 'vae' in k: continue
                m = getattr(self, k)
                if m is not None:
                    if hasattr(m, '_orig_mod'):
                        m = m._orig_mod
                    ret = m.load_state_dict(finetune_state[k], strict=strict)
                    if ret is not None:
                        missing, unexpected = ret
                        print(f'[MaskTrainer.load_state_dict] {k} missing:  {missing}')
                        print(f'[MaskTrainer.load_state_dict] {k} unexpected:  {unexpected}')
        
            config: dict = finetune_state.pop('config', None)
            if config is not None:
                for k, v in self.get_config().items():
                    if config.get(k, None) != v:
                        err = f'[MaskedPrediction.load_state_dict] config mismatch:  this.{k}={v} (ckpt.{k}={config.get(k, None)})'
                        if strict: raise AttributeError(err)
                        else: print(err)
