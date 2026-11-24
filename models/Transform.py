#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 18 16:30:13 2026

@author: jacob
"""

from torch import nn

from models.layers.FCLayer import FCLayer
        
class Transform(nn.Module):
    def __init__(self, num_inputs, num_outputs, layer_dims=[], normalize_inputs=False):
        super().__init__()

        self.num_inputs = num_inputs
        self.num_ouputs = num_outputs
        self.layer_dims = layer_dims + [num_outputs]

        #Standardise raw inputs (e.g. GA in days), otherwise the ViT's LayerNorm erases their value
        #BatchNorm1d can't take zero inputs (ehr_data: [])
        if normalize_inputs and num_inputs > 0:
            self.norm = nn.BatchNorm1d(num_inputs)
        else:
            self.norm = nn.Identity()

        layers = []
        last_dim = num_inputs
        
        for i in range(len(self.layer_dims)):
            layers.append(FCLayer(last_dim, self.layer_dims[i]))
            last_dim = self.layer_dims[i]
           
        self.fc = nn.Sequential(*layers)
        
    def forward(self, x):
        #x is (B, 1, num_inputs) and BatchNorm1d normalises dim 1
        x = self.norm(x.transpose(1, 2)).transpose(1, 2)
        features = self.fc(x)
        return features

class CNNTransform(nn.Module):
    def __init__(self, img_size, num_outputs, channel_dims=[]):
        super().__init__()

        self.input_channels = 1
        self.num_outputs = num_outputs
        self.channel_dims = channel_dims

        layers = []
        temp_dim = self.input_channels

        for dim in self.channel_dims:
            layers.extend([nn.Conv2d(temp_dim, dim, kernel_size=3, padding=1),
                           nn.GELU(),
                           nn.MaxPool2d(2)])
            temp_dim = dim

        if self.channel_dims:
            layers.append(nn.AdaptiveAvgPool2d(1))
            final_dim = temp_dim
        else:
            final_dim = img_size[0] * img_size[1]

        layers.extend([nn.Flatten(start_dim=1),
                       nn.Linear(final_dim, num_outputs)])

        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        return self.fc(x)