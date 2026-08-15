#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 18 16:30:13 2026

@author: jacob
"""

from models.layers.FCLayer import FCLayer
from torch import nn


class EmbeddingBlock(nn.Module):
    def __init__(self, num_inputs, num_outputs, layer_dims=[]):
        super().__init__()

        self.num_tokens = layer_dims[-1] // 768
        if self.num_tokens < 1:
            raise ValueError(f"num_tokens must be >= 1, got {self.num_tokens}")

        self.num_outputs = num_outputs
        self.num_inputs = num_inputs
        self.layer_dims = layer_dims
        layers = []
        last_dim = num_inputs

        for i in range(len(self.layer_dims)):
            layers.append(FCLayer(last_dim, self.layer_dims[i]))
            last_dim = self.layer_dims[i]

        self.fc = nn.Sequential(*layers)
        print(self.fc)

    def forward(self, x):
        # (B, 1, C) or (B, C) → (B, num_tokens, num_outputs)
        features = self.fc(x)
        if features.dim() == 2:
            features = features.unsqueeze(1)
        batch = features.shape[0]
        features = features.reshape(batch, self.num_tokens, self.num_outputs)
        return features
