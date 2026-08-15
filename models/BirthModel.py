#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 18 11:46:47 2026

@author: jacob
"""

import torch
from torch import nn


class BirthModel(nn.Module):
    def __init__(
        self,
        vit_model,
        tabular_embedding_block,
        preterm_heads,
        aux_task_heads,
        aux_method="append",
    ):

        super().__init__()

        self.tabular_embedding_block = tabular_embedding_block
        self.vit_model = vit_model
        self.preterm_heads = preterm_heads
        self.aux_task_heads = aux_task_heads
        self.aux_method = aux_method

        if self.aux_method == "append":
            """
            Early fusion, where EHR/Img-meta data is appended to the patch embeddings
            """
            self.forward_ = self.forward_append

        else:
            raise RuntimeError(f'Unknown fusion type f"{self.aux_method}"')

    def forward_append(self, img, tabular_data, ehr, patient_ids=None):
        embeddings = [self.tabular_embedding_block(tabular_data)]
        if len(embeddings) > 0:
            vision_features = self.vit_model(img, append_tokens=embeddings)
        else:
            vision_features = self.vit_model(img)

        outputs = {"preterm": {}, "aux_tasks": {}}

        for GA, preterm_head in self.preterm_heads.items():
            outputs["preterm"][GA] = preterm_head(vision_features)

        for var, aux_task_head in self.aux_task_heads.items():
            outputs["aux_tasks"][var] = aux_task_head(vision_features)

        return outputs, vision_features

    def freeze_model(self, model):
        for p in model.parameters():
            p.requires_grad = False

    def unfreeze_vit(self, model, n, cfg):
        total_blocks = len(model.blocks)

        # Always unfreeze final normalization layer
        for p in model.norm.parameters():
            p.requires_grad = True

        # Maximum number of blocks that are allowed to be trainable
        if cfg.training.top_n_blocks == -1:
            max_blocks = total_blocks
        else:
            max_blocks = cfg.training.top_n_blocks

        # Number of blocks to unfreeze
        if cfg.training.blocks_per_step == -1:
            n_blocks = max_blocks
        else:
            n_blocks = min(max_blocks, cfg.training.blocks_per_step * (n // cfg.training.every_n_epochs))

        if n_blocks > 0:
            for block in model.blocks[-n_blocks:]:
                for p in block.parameters():
                    p.requires_grad = True

        # Unfreeze embeddings once the whole backbone is trainable
        if n_blocks == total_blocks:
            model.cls_token.requires_grad = True
            model.pos_embed.requires_grad = True
            model.register_tokens.requires_grad = True

    def forward(self, img, img_data, ehr, patient_ids=None):
        return self.forward_(img, img_data, ehr, patient_ids)
