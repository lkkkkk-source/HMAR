# Copyright (c) 2025, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the NVIDIA One-Way Noncommercial License v1 (NSCLv1).
# To view a copy of this license, please refer to LICENSE

import math
import torch
from typing import Any, Mapping, Optional, Union, Tuple
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from models.vqvae import VQVAE
from utils.misc import does_not_contain_substrings
import dist
from functools import partial
from models.transformer_blocks import AdaLNBeforeHead, AdaLNSelfAttn
from models.vqvae import VQVAE, VectorQuantizer2
from models.helpers import sample_with_top_k_top_p_


class SharedAdaLin(nn.Linear):
    def forward(self, cond_BD):
        C = self.weight.shape[0] // 6
        return super().forward(cond_BD).view(-1, 1, 6, C)  # B16C


class HMAR(nn.Module):
    def __init__(
        self,
        vae_local: VQVAE,
        num_classes=1000,
        depth=16,
        embed_dim=1024,
        num_heads=16,
        mlp_ratio=4.0,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_eps=1e-6,
        shared_aln=False,
        cond_drop_rate=0.1,
        attn_l2_norm=False,
        patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),  # 10 steps by default
        flash_if_available=True,
        fused_if_available=True,
        n_layers_train=8,  # how many layers to not freezee for finetune
    ):
        super().__init__()
        # 0. hyperparameters
        assert embed_dim % num_heads == 0
        self.Cvae, self.V = vae_local.Cvae, vae_local.vocab_size
        self.depth, self.C, self.D, self.num_heads = (
            depth,
            embed_dim,
            embed_dim,
            num_heads,
        )

        self.cond_drop_rate = cond_drop_rate
        self.patch_nums: Tuple[int] = patch_nums
        self.L = sum(pn**2 for pn in self.patch_nums)
        self.first_l = self.patch_nums[0] ** 2
        self.begin_ends = []
        cur = 0
        for _, pn in enumerate(self.patch_nums):
            self.begin_ends.append((cur, cur + pn**2))
            cur += pn**2

        self.num_stages_minus_1 = len(self.patch_nums) - 1
        self.rng = torch.Generator(device=dist.get_device())

        # 1. input (word) embedding
        quant: VectorQuantizer2 = vae_local.quantize
        self.vae_proxy: Tuple[VQVAE] = (vae_local,)
        self.vae_quant_proxy: Tuple[VectorQuantizer2] = (quant,)

        # 2. class embedding
        init_std = math.sqrt(1 / self.C / 3)
        self.num_classes = num_classes
        self.uniform_prob = torch.full(
            (1, num_classes),
            fill_value=1.0 / num_classes,
            dtype=torch.float32,
            device=dist.get_device(),
        )
        self.class_emb = nn.Embedding(self.num_classes + 1, self.C)
        nn.init.trunc_normal_(self.class_emb.weight.data, mean=0, std=init_std)
        self.pos_start = nn.Parameter(torch.empty(1, self.first_l, self.C))
        nn.init.trunc_normal_(self.pos_start.data, mean=0, std=init_std)

        # 3. absolute position embedding
        pos_1LC = []
        for _, pn in enumerate(self.patch_nums):
            pe = torch.empty(1, pn * pn, self.C)
            nn.init.trunc_normal_(pe, mean=0, std=init_std)
            pos_1LC.append(pe)
        pos_1LC = torch.cat(pos_1LC, dim=1)  # 1, L, C
        assert tuple(pos_1LC.shape) == (1, self.L, self.C)
        self.pos_1LC = nn.Parameter(pos_1LC)

        # level embedding (similar to GPT's segment embedding, used to distinguish different levels of token pyramid)
        self.lvl_embed = nn.Embedding(len(self.patch_nums), self.C)
        nn.init.trunc_normal_(self.lvl_embed.weight.data, mean=0, std=init_std)

        # 4. backbone blocks
        self.shared_ada_lin = (
            nn.Sequential(nn.SiLU(inplace=False), SharedAdaLin(self.D, 6 * self.C))
            if shared_aln
            else nn.Identity()
        )

        norm_layer = partial(nn.LayerNorm, eps=norm_eps)
        self.drop_path_rate = drop_path_rate
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule (linearly increasing)

        self.base_blocks = nn.ModuleList(
            [
                AdaLNSelfAttn(
                    cond_dim=self.D,
                    shared_aln=shared_aln,
                    block_idx=block_idx,
                    embed_dim=self.C,
                    norm_layer=norm_layer,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[block_idx],
                    last_drop_p=0 if block_idx == 0 else dpr[block_idx - 1],
                    attn_l2_norm=attn_l2_norm,
                    flash_if_available=flash_if_available,
                    fused_if_available=fused_if_available,
                )
                for block_idx in range(depth - n_layers_train)
            ]
        )

        self.ns_blocks = nn.ModuleList(
            [
                AdaLNSelfAttn(
                    cond_dim=self.D,
                    shared_aln=shared_aln,
                    block_idx=block_idx,
                    embed_dim=self.C,
                    norm_layer=norm_layer,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[block_idx],
                    last_drop_p=0 if block_idx == 0 else dpr[block_idx - 1],
                    attn_l2_norm=attn_l2_norm,
                    flash_if_available=flash_if_available,
                    fused_if_available=fused_if_available,
                )
                for block_idx in range(n_layers_train)
            ]
        )

        self.mask_blocks = nn.ModuleList(
            [
                AdaLNSelfAttn(
                    cond_dim=self.D,
                    shared_aln=shared_aln,
                    block_idx=block_idx,
                    embed_dim=self.C,
                    norm_layer=norm_layer,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[block_idx],
                    last_drop_p=0 if block_idx == 0 else dpr[block_idx - 1],
                    attn_l2_norm=attn_l2_norm,
                    flash_if_available=flash_if_available,
                    fused_if_available=fused_if_available,
                )
                for block_idx in range(n_layers_train)
            ]
        )

        d: torch.Tensor = torch.cat(
            [torch.full((pn * pn,), i) for i, pn in enumerate(self.patch_nums)]
        ).view(1, self.L, 1)
        dT = d.transpose(1, 2)  # dT: 11L
        lvl_1L = dT[:, 0].contiguous()
        self.register_buffer("lvl_1L", lvl_1L)

        attn_bias_for_masking = torch.where(d == dT, 0.0, -torch.inf).reshape(
            1, 1, self.L, self.L
        )
        self.register_buffer(
            "attn_bias_for_masking", attn_bias_for_masking.contiguous()
        )

        # 6. classifier head
        self.ns_head_nm = AdaLNBeforeHead(self.C, self.D, norm_layer=norm_layer)
        self.ns_head = nn.Linear(self.C, self.V)

        self.mask_head_nm = AdaLNBeforeHead(self.C, self.D, norm_layer=norm_layer)
        self.mask_head = nn.Linear(self.C, self.V)

        self.word_embed = nn.Linear(self.Cvae, self.C, bias=False)
        self.word_embed_bias = nn.Parameter(torch.zeros(self.C))

        init_std = math.sqrt(1 / self.C / 3)

        # mask embedding
        self.mask_embed = nn.Embedding(1, self.C)
        nn.init.trunc_normal_(self.mask_embed.weight.data, mean=0, std=init_std)

        assert (
            n_layers_train <= self.depth
        ), f"n_layers_train should be less than depth {self.depth}"

        ns_blocks_train = [f"ns_blocks.{i}" for i in range(n_layers_train)]
        mask_blocks_train = [f"mask_blocks.{i}" for i in range(n_layers_train)]
        self.train_params = [
            "mask_embed",
            "class_emb",
            "ns_head",
            "ns_head_nm",
            "mask_head",
            "mask_head_nm",
        ] + ns_blocks_train + mask_blocks_train
        self.n_layers_train = n_layers_train
        # freeze all params except those to be finetuned
        for name, param in self.named_parameters():
            if param.requires_grad and does_not_contain_substrings(
                name, self.train_params
            ):
                param.requires_grad = False

        self.copied_params = []

    def get_word_embed(self, x: torch.Tensor, idx_to_mask) -> torch.Tensor:
        B = x.shape[0]
        x_ns_we_wo_bias = self.word_embed(x[: B // 2, ...].float())

        x_mask_we_wo_bias = self.word_embed(x[B // 2 :, ...].float())
        x_mask_we_wo_bias[:, idx_to_mask, :] = self.mask_embed(
            torch.tensor(0, device=x.device, dtype=torch.long)
        )
        x_mask_we = x_ns_we_wo_bias + x_mask_we_wo_bias + self.word_embed_bias

        return x_mask_we

    def _embed_prefix(self, label_B: torch.LongTensor, B: int):
        label_B = torch.where(
            torch.rand(B, device=label_B.device) < self.cond_drop_rate,
            self.num_classes,
            label_B,
        )
        sos = cond_BD = self.class_emb(label_B)
        sos = sos.unsqueeze(1).expand(B, self.first_l, -1) + self.pos_start.expand(
            B, self.first_l, -1
        )
        cond_BD_or_gss = self.shared_ada_lin(cond_BD)
        return sos, cond_BD, cond_BD_or_gss

    def _run_blocks(self, blocks, x_BLC, cond_BD_or_gss, attn_bias, use_no_grad=False):
        iterator = blocks
        if use_no_grad:
            with torch.no_grad():
                for b in iterator:
                    x_BLC = b(
                        x=x_BLC,
                        cond_BD=cond_BD_or_gss,
                        attn_bias=attn_bias,
                        using_block_sparse_attn=False,
                    )
            return x_BLC.detach()

        for b in iterator:
            if self.training:
                x_BLC = checkpoint(
                    self._forward_block,
                    b,
                    x_BLC,
                    cond_BD_or_gss,
                    attn_bias,
                    use_reentrant=False,
                )
            else:
                x_BLC = self._forward_block(b, x_BLC, cond_BD_or_gss, attn_bias)
        return x_BLC

    def _forward_block(self, block, x_BLC, cond_BD_or_gss, attn_bias):
        return block(
            x=x_BLC,
            cond_BD=cond_BD_or_gss,
            attn_bias=attn_bias,
            using_block_sparse_attn=False,
        )

    def forward_ns(self, label_B: torch.LongTensor, x_BLCv_wo_first_l: torch.Tensor):
        B = x_BLCv_wo_first_l.shape[0]
        with torch.amp.autocast("cuda", enabled=False):
            sos, cond_BD, cond_BD_or_gss = self._embed_prefix(label_B, B)
            x_BLC = torch.cat(
                (sos, self.word_embed(x_BLCv_wo_first_l.float()) + self.word_embed_bias), dim=1
            )
            x_BLC += self.lvl_embed(self.lvl_1L.expand(B, -1)) + self.pos_1LC

        attn_bias = self.attn_bias_for_masking
        temp = x_BLC.new_ones(8, 8)
        main_type = torch.matmul(temp, temp).dtype
        x_BLC = x_BLC.to(dtype=main_type)
        cond_BD_or_gss = cond_BD_or_gss.to(dtype=main_type)
        attn_bias = attn_bias.to(dtype=main_type)

        x_BLC = self._run_blocks(self.base_blocks, x_BLC, cond_BD_or_gss, attn_bias, use_no_grad=True)
        x_BLC = self._run_blocks(self.ns_blocks, x_BLC, cond_BD_or_gss, attn_bias, use_no_grad=False)
        return self.get_ns_logits(x_BLC.float(), cond_BD)

    def forward_mask(
        self, label_B: torch.LongTensor, x_BLCv_wo_first_l: torch.Tensor, idx_to_mask: torch.Tensor
    ):
        B = x_BLCv_wo_first_l.shape[0] // 2
        with torch.amp.autocast("cuda", enabled=False):
            sos, cond_BD, cond_BD_or_gss = self._embed_prefix(label_B, B)
            x_BLC = torch.cat(
                (sos, self.get_word_embed(x_BLCv_wo_first_l, idx_to_mask)), dim=1
            )
            x_BLC += self.lvl_embed(self.lvl_1L.expand(B, -1)) + self.pos_1LC

        attn_bias = self.attn_bias_for_masking
        temp = x_BLC.new_ones(8, 8)
        main_type = torch.matmul(temp, temp).dtype
        x_BLC = x_BLC.to(dtype=main_type)
        cond_BD_or_gss = cond_BD_or_gss.to(dtype=main_type)
        attn_bias = attn_bias.to(dtype=main_type)

        x_BLC = self._run_blocks(self.base_blocks, x_BLC, cond_BD_or_gss, attn_bias, use_no_grad=True)
        x_BLC = self._run_blocks(self.mask_blocks, x_BLC, cond_BD_or_gss, attn_bias, use_no_grad=False)
        return self.get_mask_logits(x_BLC.float(), cond_BD)

    def get_ns_logits(
        self,
        h_or_h_and_residual: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        cond_BD: Optional[torch.Tensor],
    ):
        if not isinstance(h_or_h_and_residual, torch.Tensor):
            h, resi = h_or_h_and_residual  # fused_add_norm must be used
            h = resi + self.ns_blocks[-1].drop_path(h)
        else:  # fused_add_norm is not used
            h = h_or_h_and_residual
        return self.ns_head(self.ns_head_nm(h.float(), cond_BD).float()).float()

    def get_mask_logits(
        self,
        h_or_h_and_residual: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        cond_BD: Optional[torch.Tensor],
    ):
        if not isinstance(h_or_h_and_residual, torch.Tensor):
            h, resi = h_or_h_and_residual  # fused_add_norm must be used
            h = resi + self.mask_blocks[-1].drop_path(h)
        else:  # fused_add_norm is not used
            h = h_or_h_and_residual
        return self.mask_head(self.mask_head_nm(h.float(), cond_BD).float()).float()

    def forward(
        self,
        label_B: torch.LongTensor,
        x_BLCv_wo_first_l: torch.Tensor,
        idx_to_mask: torch.Tensor,
        mode: str = "both",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B = x_BLCv_wo_first_l.shape[0] // 2
        x_ns = x_BLCv_wo_first_l[:B]
        if mode == "ns":
            return self.forward_ns(label_B, x_ns)
        if mode == "mask":
            return self.forward_mask(label_B, x_BLCv_wo_first_l, idx_to_mask)
        if mode == "both":
            return self.forward_ns(label_B, x_ns), self.forward_mask(label_B, x_BLCv_wo_first_l, idx_to_mask)
        raise ValueError(f"Unknown HMAR forward mode: {mode}")

    @torch.no_grad()
    def generate(
        self,
        B: int,
        label_B: Optional[Union[int, torch.LongTensor]],
        g_seed: Optional[int] = None,
        cfg=1.5,
        top_k=1100,
        top_p=0.999,
        more_smooth=False,
        num_samples=1,
        mask=True,
        mask_schedule=None,
        kv_cache=False # Only used to benchmark and compare performance to VAR
    ) -> torch.Tensor:
        
        #TODO: Support sampling with gumbel_softmax like in MaskGIT and VAR, when more_smooth is True.
        
        """
        only used for inference, on autoregressive mode
        :param B: batch size
        :param label_B: imagenet label; if None, randomly sampled
        :param g_seed: random seed
        :param cfg: classifier-free guidance ratio
        :param top_k: top-k sampling
        :param top_p: top-p sampling
        :param more_smooth: smoothing the pred using gumbel softmax; only used in visualization, not used in FID/IS benchmarking
        :return: if returns_vemb: list of embedding h_BChw := vae_embed(idx_Bl), else: list of idx_Bl
        """
        if g_seed is None:
            rng = None
        else:
            self.rng.manual_seed(g_seed)
            rng = self.rng

        if label_B is None:
            label_B = torch.multinomial(
                self.uniform_prob, num_samples=B, replacement=True, generator=rng
            ).reshape(B)
        elif isinstance(label_B, int):
            label_B = torch.full(
                (B,),
                fill_value=self.num_classes if label_B < 0 else label_B,
                device=self.lvl_1L.device,
            )

        sos = cond_BD = self.class_emb(
            torch.cat(
                (label_B, torch.full_like(label_B, fill_value=self.num_classes)), dim=0
            )
        )
        cond_BD_or_gss = self.shared_ada_lin(
            cond_BD
        )
        lvl_pos = self.lvl_embed(self.lvl_1L) + self.pos_1LC
        next_token_map = (
            sos.unsqueeze(1).expand(2 * B, self.first_l, -1)
            + self.pos_start.expand(2 * B, self.first_l, -1)
            + lvl_pos[:, : self.first_l].expand(2 * B, self.first_l, -1)
        )

        cur_L = 0
        f_hat = sos.new_zeros(B, self.Cvae, self.patch_nums[-1], self.patch_nums[-1])
    
        ntokens_per_steps =  mask_schedule
       
        # This is only used for benchmarking to compare the performance to VAR
        if kv_cache:
            for b in self.base_blocks: b.attn.kv_caching(True)
            for b in self.ns_blocks: b.attn.kv_caching(True)
            
        for si, pn in enumerate(self.patch_nums):  # si: i-th segment
            ratio = si / self.num_stages_minus_1
            x = next_token_map
            
            for b in self.base_blocks:
                x = b(
                    x=x,
                    cond_BD=cond_BD_or_gss,
                    using_block_sparse_attn=False,
                    attn_bias=None,
                )
                
            for b in self.ns_blocks:
                x = b(
                    x=x,
                    cond_BD=cond_BD_or_gss,
                    using_block_sparse_attn=False,
                    attn_bias=None,
                )

            logits_BlV = self.get_ns_logits(x, cond_BD)

            t = cfg * ratio
            logits_BlV = (1 + t) * logits_BlV[:B] - t * logits_BlV[B:]

            idx_Bl = sample_with_top_k_top_p_(
                logits_BlV, rng=rng, top_k=top_k, top_p=top_p, num_samples=num_samples
            )
            idx_Bl = idx_Bl[:, :, 0]
            h_BChw = self.vae_quant_proxy[0].embedding(idx_Bl)

            h_BChw = h_BChw.transpose_(1, 2).reshape(B, self.Cvae, pn, pn)

            if mask and pn * pn > 1 and si < len(self.patch_nums):
                n_steps = len(ntokens_per_steps[si])
                n_tokens_mask = sum(ntokens_per_steps[si][1:])
                probs = torch.nn.functional.softmax(logits_BlV, dim=-1)
                probs_sampled = torch.gather(probs, 2, idx_Bl.unsqueeze(-1)).squeeze(-1)
                idx_to_mask = torch.argsort(probs_sampled, dim=-1)[:, :n_tokens_mask]
                
                for step in range(1, n_steps):
                    ratio_step =  1e-6 #TODO: Remove this from being hardcoded
                    f_hat_mask, next_token_map_mask = self.vae_quant_proxy[
                        0
                    ].get_next_mask_input(si, len(self.patch_nums), f_hat, h_BChw)
                    next_token_map_mask = next_token_map_mask.view(
                        B, self.Cvae, -1
                    ).transpose(1, 2)
                    f_hat_mask = f_hat_mask.view(B, self.Cvae, -1).transpose(1, 2)
                    f_hat_mask = self.word_embed(f_hat_mask)
                    next_token_map_mask = self.word_embed(next_token_map_mask)
                    next_token_map_mask = torch.scatter(
                        next_token_map_mask,
                        1,
                        idx_to_mask.unsqueeze(-1).expand(-1, -1, self.C),
                        self.mask_embed(
                            torch.tensor(0, device=dist.get_device(), dtype=torch.int)
                        ).expand(B, pn * pn, -1),
                    )
                    next_token_map_mask = (
                        f_hat_mask
                        + next_token_map_mask
                        + self.word_embed_bias
                        + lvl_pos[:, cur_L : cur_L + self.patch_nums[si] ** 2]
                    )
                    next_token_map_mask = next_token_map_mask.repeat(2, 1, 1)

                    x = next_token_map_mask
                    
                    for b in self.base_blocks:
                        x = b(
                            x=x,
                            cond_BD=cond_BD_or_gss,
                            using_block_sparse_attn=False,
                            attn_bias=None,
                        )
                    for b in self.mask_blocks:
                        x = b(
                            x=x,
                            cond_BD=cond_BD_or_gss,
                            using_block_sparse_attn=False,
                            attn_bias=None,
                        )
                    logits_BlV_mask = self.get_mask_logits(x, cond_BD)

                    t = cfg * ratio_step
                    logits_BlV_mask = (1 + t) * logits_BlV_mask[
                        :B
                    ] - t * logits_BlV_mask[B:]

                    idx_Bl_mask = sample_with_top_k_top_p_(
                        logits_BlV_mask,
                        rng=rng,
                        top_k=top_k,
                        top_p=top_p,
                        num_samples=num_samples,
                    )
                    idx_Bl_mask = idx_Bl_mask[:, :, 0]

                    idx_Bl[torch.arange(B).unsqueeze(1), idx_to_mask] = idx_Bl_mask[
                        torch.arange(B).unsqueeze(1), idx_to_mask
                    ]
                    h_BChw = self.vae_quant_proxy[0].embedding(idx_Bl)

                    h_BChw = h_BChw.transpose_(1, 2).reshape(B, self.Cvae, pn, pn)

                    if step != n_steps - 1:
                        n_tokens_mask = sum(ntokens_per_steps[si][step + 1 :])
                        probs = torch.softmax(logits_BlV_mask, dim=-1)
                        probs_sampled = torch.gather(
                            probs, 2, idx_Bl.unsqueeze(-1)
                        ).squeeze(-1)
                        probs_sampled_masked = probs_sampled[
                            torch.arange(B).unsqueeze(1), idx_to_mask
                        ]
                        idx_sampled_sorted = torch.argsort(
                            probs_sampled_masked, dim=-1
                        )[:, :n_tokens_mask]
                        idx_to_mask = idx_to_mask[
                            torch.arange(B).unsqueeze(1), idx_sampled_sorted
                        ]

            f_hat, next_token_map = self.vae_quant_proxy[
                0
            ].get_next_autoregressive_input(si, len(self.patch_nums), f_hat, h_BChw)

            cur_L += pn * pn

            if si != self.num_stages_minus_1:  # prepare for next stage
                next_token_map = next_token_map.view(B, self.Cvae, -1).transpose(1, 2)
                next_token_map = (
                    self.word_embed(next_token_map)
                    + self.word_embed_bias
                    + lvl_pos[:, cur_L : cur_L + self.patch_nums[si + 1] ** 2]
                )
                next_token_map = next_token_map.repeat(
                    2, 1, 1
                )  # double the batch sizes due to CFG

         # This is only used for benchmarking to compare the performance to VAR
        if kv_cache:
            for b in self.base_blocks: b.attn.kv_caching(False)
            for b in self.ns_blocks: b.attn.kv_caching(False)
            
        return (
            self.vae_proxy[0].fhat_to_img(f_hat).add_(1).mul_(0.5)
        )  # de-normalize, from [-1, 1] to [0, 1]

    def load_base_and_ns_state_dict(self, state_dict: Mapping[str, Any]):
        # load the word embedding and bias
        self.copied_params = []

        for name, param in state_dict.items():
            if "head" in name:
                self.state_dict()[f"ns_{name}"].copy_(param)
                self.copied_params.append(f"ns_{name}")
            elif "word_embed.weight" in name:
                self.word_embed.weight.copy_(param)
                self.copied_params.append(name)
            elif "word_embed.bias" in name:
                self.word_embed_bias.copy_(param)
                self.copied_params.append(name)
            elif "blocks" in name and name.split(".")[1] in [
                f"{i}" for i in range(self.depth - self.n_layers_train)
            ]:
                new_name = "base_blocks." + ".".join(name.split(".")[1:])
                self.state_dict()[new_name].copy_(param)
                self.copied_params.append(new_name)
            elif "blocks" in name and name.split(".")[1] in [
                f"{i}" for i in range(self.depth - self.n_layers_train, self.depth)
            ]:
                new_name = f'ns_blocks.{int(name.split(".")[1])-(self.depth - self.n_layers_train)}.{".".join(name.split(".")[2:])}'
                self.state_dict()[new_name].copy_(param)
                self.copied_params.append(new_name)
            elif name in self.state_dict().keys():
                self.state_dict()[name].copy_(param)
                self.copied_params.append(name)

    def load_mask_dict(self, state_dict: Mapping[str, Any]):
        for name, param in state_dict.items():
            if "head" in name:
                self.state_dict()[f"mask_{name}"].copy_(param)
                self.copied_params.append(f"mask_{name}")
            elif "mask" in name:
                self.state_dict()[name].copy_(param)
                self.copied_params.append(name)
            elif "blocks" in name and name.split(".")[1] in [
                f"{i}" for i in range(self.depth - self.n_layers_train, self.depth)
            ]:
                new_name = f'mask_blocks.{int(name.split(".")[1])-(self.depth - self.n_layers_train)}.{".".join(name.split(".")[2:])}'
                self.state_dict()[new_name].copy_(param)
                self.copied_params.append(new_name)
        print(set(self.state_dict().keys()) - set(self.copied_params))
