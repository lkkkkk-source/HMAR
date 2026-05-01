# Copyright (c) 2025, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the NVIDIA One-Way Noncommercial License v1 (NSCLv1).
# To view a copy of this license, please refer to LICENSE

import math
import re
import torch
from typing import Any, Mapping
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from models.vqvae import VQVAE
from .transformer import Transformer
from utils.misc import does_not_contain_substrings


class MaskedPrediction(Transformer):
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
        using_block_sparse_attn=True,
        n_layers_train=2,  # how many layers to not freezee for finetune
    ):
        super(MaskedPrediction, self).__init__(
            vae_local,
            num_classes=num_classes,
            depth=depth,
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_eps=norm_eps,
            shared_aln=shared_aln,
            cond_drop_rate=cond_drop_rate,
            attn_l2_norm=attn_l2_norm,
            patch_nums=patch_nums,
            flash_if_available=flash_if_available,
            fused_if_available=fused_if_available,
            using_block_sparse_attn=using_block_sparse_attn,
        )

        self.word_embed = nn.Linear(self.Cvae, self.C, bias=False)
        self.word_embed_bias = nn.Parameter(torch.zeros(self.C))

        init_std = math.sqrt(1 / self.C / 3)
        # mask embedding
        self.mask_embed = nn.Embedding(1, self.C)
        nn.init.trunc_normal_(self.mask_embed.weight.data, mean=0, std=init_std)

        assert (
            n_layers_train <= self.depth
        ), f"n_layers_train should be less than depth {self.depth}"
        self.n_layers_train = n_layers_train

        # transformer blocks to keep for finetune
        blocks_train = [
            f"blocks.{i}" for i in range(self.depth - n_layers_train, self.depth)
        ]
        self.train_params = ["mask_embed", "head"] + blocks_train

        # freeze all params except those to be finetuned
        for name, param in self.named_parameters():
            if param.requires_grad and does_not_contain_substrings(
                name, self.train_params
            ):
                param.requires_grad = False

    def get_word_embed(self, x: torch.Tensor, idx_to_mask) -> torch.Tensor:
        B = x.shape[0]
        x_ns_we_wo_bias = self.word_embed(x[: B // 2, ...].float())

        x_mask_we_wo_bias = self.word_embed(x[B // 2 :, ...].float())
        x_mask_we_wo_bias[:, idx_to_mask, :] = self.mask_embed(
            torch.tensor(0, device=x.device, dtype=torch.long)
        )
        x_mask_we = x_ns_we_wo_bias + x_mask_we_wo_bias + self.word_embed_bias

        return x_mask_we

    def forward(
        self,
        label_B: torch.LongTensor,
        x_BLCv_wo_first_l: torch.Tensor,
        idx_to_mask: torch.Tensor,
    ) -> torch.Tensor:  # returns logits_BLV
        """
        :param label_B: label_B
        :param x_BLCv_wo_first_l: teacher forcing input (B, self.L-self.first_l, self.Cvae)
        :idx_to_mask: index to mask
        :return: logits BLV, V is vocab_size
        """
        B = x_BLCv_wo_first_l.shape[0] // 2
        with torch.amp.autocast('cuda', enabled=False):
            label_B = torch.where(
                torch.rand(B, device=label_B.device) < self.cond_drop_rate,
                self.num_classes,
                label_B,
            )
            sos = cond_BD = self.class_emb(label_B)
            sos = sos.unsqueeze(1).expand(B, self.first_l, -1) + self.pos_start.expand(
                B, self.first_l, -1
            )

            x_BLC = torch.cat(
                (sos, self.get_word_embed(x_BLCv_wo_first_l, idx_to_mask)), dim=1
            )

            x_BLC += self.lvl_embed(self.lvl_1L.expand(B, -1)) + self.pos_1LC

        attn_bias = self.attn_bias_for_masking
        cond_BD_or_gss = self.shared_ada_lin(cond_BD)

        # hack: get the dtype if mixed precision is used
        temp = x_BLC.new_ones(8, 8)
        main_type = torch.matmul(temp, temp).dtype

        x_BLC = x_BLC.to(dtype=main_type)
        cond_BD_or_gss = cond_BD_or_gss.to(dtype=main_type)
        attn_bias = attn_bias.to(dtype=main_type)

        frozen_block_count = self.depth - self.n_layers_train
        if frozen_block_count > 0:
            with torch.no_grad():
                for b in self.blocks[:frozen_block_count]:
                    x_BLC = b(
                        x=x_BLC,
                        cond_BD=cond_BD_or_gss,
                        using_block_sparse_attn=self.using_block_sparse_attn,
                        attn_bias=attn_bias,
                    )
            x_BLC = x_BLC.detach()

        for b in self.blocks[frozen_block_count:]:
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
        x_BLC = self.get_logits(x_BLC.float(), cond_BD)

        return x_BLC

    def _forward_block(self, block, x_BLC, cond_BD_or_gss, attn_bias):
        return block(
            x=x_BLC,
            cond_BD=cond_BD_or_gss,
            using_block_sparse_attn=self.using_block_sparse_attn,
            attn_bias=attn_bias,
        )

    def load_state_dict_with_word_embed(
        self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False
    ):
        if any(name.startswith("base_blocks.") or name.startswith("mask_blocks.") for name in state_dict):
            state_dict = self._convert_hmar_state_dict(state_dict)

        current_state = self.state_dict()
        filtered_state = {}
        skipped = []
        for name, param in state_dict.items():
            if name not in current_state:
                continue
            if name == "class_emb.weight" and param.shape != current_state[name].shape:
                # Reuse the unconditional embedding from the released HMAR checkpoint,
                # but keep class-specific rows randomly initialized for the new dataset.
                current_state[name][-1].copy_(param[-1])
                continue
            if current_state[name].shape != param.shape:
                skipped.append((name, tuple(param.shape), tuple(current_state[name].shape)))
                continue
            filtered_state[name] = param

        ret = self.load_state_dict(filtered_state, strict=False)
        if skipped:
            print(f'[MaskedPrediction.load_state_dict_with_word_embed] skipped shape-mismatched keys: {skipped}')
        for name, param in state_dict.items():
            if "word_embed.weight" in name:
                self.word_embed.weight.copy_(param)
            if "word_embed.bias" in name:
                self.word_embed_bias.copy_(param)
        return ret

    def _convert_hmar_state_dict(self, state_dict: Mapping[str, Any]):
        converted = {}
        base_ids = sorted({
            int(m.group(1))
            for name in state_dict.keys()
            for m in [re.match(r"base_blocks\.(\d+)\.", name)]
            if m is not None
        })
        mask_ids = sorted({
            int(m.group(1))
            for name in state_dict.keys()
            for m in [re.match(r"mask_blocks\.(\d+)\.", name)]
            if m is not None
        })
        base_offset = len(base_ids)

        for name, param in state_dict.items():
            if name.startswith("base_blocks."):
                converted["blocks." + name[len("base_blocks."):]] = param
            elif name.startswith("mask_blocks."):
                rest = name[len("mask_blocks."):]
                block_idx, suffix = rest.split(".", 1)
                converted[f"blocks.{base_offset + int(block_idx)}.{suffix}"] = param
            elif name.startswith("mask_head_nm."):
                converted["head_nm." + name[len("mask_head_nm."):]] = param
            elif name.startswith("mask_head."):
                converted["head." + name[len("mask_head."):]] = param
            elif name in {
                "class_emb.weight",
                "pos_start",
                "pos_1LC",
                "word_embed.weight",
                "word_embed_bias",
                "mask_embed.weight",
            } or name.startswith("lvl_embed.") or name.startswith("shared_ada_lin."):
                converted[name] = param

        return converted
