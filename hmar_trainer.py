import random
import time
from typing import List, Optional, Tuple, Union

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

import dist
from models import HMAR, VQVAE
from utils.amp_sc import AmpOptimizer
from utils.misc import MetricLogger, WandbLogger
from trainer import Trainer

Ten = torch.Tensor
FTen = torch.Tensor
ITen = torch.LongTensor
BTen = torch.BoolTensor


class HMARTrainer(Trainer):
    def __init__(
        self,
        device,
        patch_nums: Tuple[int, ...],
        resos: Tuple[int, ...],
        vae_local: VQVAE,
        hmar_wo_ddp: HMAR,
        hmar: DDP,
        optimizer: AmpOptimizer,
        label_smooth: float,
        reweight_loss: bool = False,
        loss_reweight_type: str = "mask_unweighted",
    ):
        super(HMARTrainer, self).__init__(
            device, patch_nums, resos, vae_local, hmar_wo_ddp, hmar, optimizer, label_smooth, reweight_loss, loss_reweight_type
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
            idx_to_mask_plus_1 = idx_to_mask + 1
            idx_to_mask_list_plus_1 = [idx + 1 for idx in idx_to_mask_list]

            with torch.autocast("cuda", enabled=True, cache_enabled=True):
                logits_ns, logits_mask = self.transformer_wo_ddp(label_B, x, idx_to_mask)

            loss_ns = self.val_loss(logits_ns.view(-1, V), gt_BL.view(-1))
            loss_mask = self.val_loss(
                logits_mask[:, idx_to_mask_plus_1, :].reshape(-1, V),
                gt_BL[:, idx_to_mask_plus_1].reshape(-1),
            )
            logits_for_eval = logits_mask

            L_mean += (loss_ns + loss_mask) * B
            L_tail += self.val_loss(
                logits_for_eval.data[:, idx_to_mask_list_plus_1[-1], :].reshape(-1, V),
                gt_BL[:, idx_to_mask_list_plus_1[-1]].reshape(-1),
            ) * B
            acc_mean += (logits_for_eval.data[:, idx_to_mask_plus_1].argmax(dim=-1) == gt_BL[:, idx_to_mask_plus_1]).sum() * (
                100 / idx_to_mask_plus_1.shape[0]
            )
            acc_tail += (
                logits_for_eval.data[:, idx_to_mask_list_plus_1[-1]].argmax(dim=-1)
                == gt_BL[:, idx_to_mask_list_plus_1[-1]]
            ).sum() * (100 / idx_to_mask_list_plus_1[-1].shape[0])

            idx_to_mask_list_plus_1 = [torch.tensor([0], device=dist.get_device(), dtype=torch.long)] + idx_to_mask_list_plus_1
            for si, (_, ed) in enumerate(self.begin_ends):
                L_resos[si] += self.val_loss(
                    logits_for_eval.data[:, idx_to_mask_list_plus_1[si], :].reshape(-1, V),
                    gt_BL[:, idx_to_mask_list_plus_1[si]].reshape(-1),
                ) * B
                acc_resos[si] += (
                    logits_for_eval.data[:, idx_to_mask_list_plus_1[si]].argmax(dim=-1)
                    == gt_BL[:, idx_to_mask_list_plus_1[si]]
                ).sum() * (100 / (ed - self.begin_ends[si][0]))
            tot += B

        self.transformer_wo_ddp.train(training)

        stats = L_mean.new_tensor(L_resos + acc_resos + [L_mean.item(), L_tail.item(), acc_mean.item(), acc_tail.item(), tot])
        dist.allreduce(stats)
        tot = round(stats[-1].item())
        stats /= tot
        L_mean, L_tail, acc_mean, acc_tail, _ = stats.tolist()[len(self.resos * 2) :]
        L_resos = stats.tolist()[: len(self.resos)]
        acc_resos = stats.tolist()[len(self.resos) : len(self.resos * 2)]
        return L_mean, L_tail, acc_mean, acc_tail, L_resos, acc_resos, tot, time.time() - stt

    def train_step(
        self,
        it: int,
        g_it: int,
        stepping: bool,
        metric_lg: MetricLogger,
        wdb_lg: WandbLogger,
        inp_B3HW: FTen,
        label_B: Union[ITen, FTen],
        eval_labels: List[int],
        log_imgs_iters: int,
    ) -> Tuple[Optional[Union[Ten, float]], Optional[float]]:
        B, V = label_B.shape[0], self.vae_local.vocab_size
        self.transformer.require_backward_grad_sync = stepping

        gt_idx_Bl: List[ITen] = self.vae_local.img_to_idxBl(inp_B3HW)
        gt_BL = torch.cat(gt_idx_Bl, dim=1)
        x_ns = self.quantize_local.idxBl_to_ns_input(gt_idx_Bl)
        p_mask = random.random()
        x_mask, idx_to_mask_list = self.quantize_local.idxBl_to_mask_input(gt_idx_Bl, p_mask=p_mask)
        x = torch.cat([x_ns, x_mask], dim=0)
        idx_to_mask = torch.cat(idx_to_mask_list, dim=0)
        idx_to_mask_plus_1 = idx_to_mask + 1
        idx_to_mask_list_plus_1 = [idx + 1 for idx in idx_to_mask_list]

        with self.optimizer.amp_ctx:
            with torch.autocast("cuda", enabled=True, cache_enabled=True):
                logits_ns, logits_mask = self.transformer(label_B, x, idx_to_mask)

            loss_ns = self.train_loss(logits_ns.view(-1, V), gt_BL.view(-1)).view(B, -1)
            loss_ns = loss_ns.mul(self.loss_weight).sum(dim=-1).mean()

            loss_mask = self.train_loss(
                logits_mask[:, idx_to_mask_plus_1, :].reshape(-1, V),
                gt_BL[:, idx_to_mask_plus_1].reshape(-1),
            ).view(B, -1)
            if self.loss_reweight_type == "mask_unweighted":
                L = idx_to_mask.shape[0]
                loss_weight = torch.ones(1, L, device=dist.get_device()) / L
            else:
                loss_weight = self.loss_weight[:, idx_to_mask]
            loss_mask = loss_mask.mul(loss_weight).sum(dim=-1).mean()
            loss = loss_ns + loss_mask

        grad_norm, scale_log2 = self.optimizer.backward_clip_step(loss=loss, stepping=stepping)

        pred_BL = logits_mask.data.argmax(dim=-1)
        if it == 0 or it in metric_lg.log_iters:
            Lmean = (
                self.val_loss(logits_ns.data.view(-1, V), gt_BL.view(-1)).item()
                + self.val_loss(
                    logits_mask[:, idx_to_mask_plus_1, :].data.view(-1, V),
                    gt_BL[:, idx_to_mask_plus_1].view(-1),
                ).item()
            )
            acc_mean = (pred_BL[:, idx_to_mask_plus_1] == gt_BL[:, idx_to_mask_plus_1]).float().mean().item() * 100
            Ltail = self.val_loss(
                logits_mask.data[:, idx_to_mask_list_plus_1[-1], :].reshape(-1, V),
                gt_BL[:, idx_to_mask_list_plus_1[-1]].reshape(-1),
            ).item()
            acc_tail = (pred_BL[:, idx_to_mask_list_plus_1[-1]] == gt_BL[:, idx_to_mask_list_plus_1[-1]]).float().mean().item() * 100
            grad_norm = grad_norm.item() if grad_norm is not None else 0.0
            metric_lg.update(Lm=Lmean, Ltail=Ltail, Accm=acc_mean, Acct=acc_tail, tnm=grad_norm)

        return grad_norm, scale_log2

    def load_state_dict(self, state, strict=True, skip_vae=False):
        for k in ("transformer_wo_ddp", "vae_local", "optimizer"):
            if skip_vae and "vae" in k:
                continue
            m = getattr(self, k)
            if m is not None:
                if hasattr(m, "_orig_mod"):
                    m = m._orig_mod
                ret = m.load_state_dict(state[k], strict=strict)
                if ret is not None:
                    missing, unexpected = ret
                    print(f"[HMARTrainer.load_state_dict] {k} missing:  {missing}")
                    print(f"[HMARTrainer.load_state_dict] {k} unexpected:  {unexpected}")

        config: dict = state.pop("config", None)
        if config is not None:
            for k, v in self.get_config().items():
                if config.get(k, None) != v:
                    err = f"[HMAR.load_state_dict] config mismatch:  this.{k}={v} (ckpt.{k}={config.get(k, None)})"
                    if strict:
                        raise AttributeError(err)
                    else:
                        print(err)
