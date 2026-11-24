#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 12:08:41 2026

@author: jacob
"""
import torch 
from omegaconf import ListConfig

class ModelLoss():
    def __init__(self, cfg):
        self.loss_fns = get_loss(cfg)
        self.device = cfg.device.type
        self.pos_weight = cfg.tasks.preterm.pos_weight
        self.pos_weight_multiplier = cfg.tasks.preterm.pos_weight_multiplier
        tasks = {}

        for task_name, task_cfg in cfg.tasks.items():
            if task_name == "aux_tasks":
                tasks[task_name] = [[aux_cfg.var, aux_cfg.weights, aux_cfg.mask_value] for aux_cfg in task_cfg]
            
            elif task_name == 'preterm':
                cutoffs = task_cfg.get("cutoffs", [None])
                cutoff_weights = task_cfg.get("cutoff_weights", 1)

                if isinstance(cutoff_weights, (int, float)):
                    cutoff_weights = [cutoff_weights] * len(cutoffs)
    
                tasks[task_name] = [[cutoff, cutoff_weight, task_cfg.label_smoothing_value, task_cfg.distance_weight] 
                                    for cutoff, cutoff_weight in zip(cutoffs, cutoff_weights)]
        
        self.tasks = tasks
        self.loss = self.loss = {task_name: {'loss': [], 'count': []} for task_name in self.tasks}
        
    def __call__(self, outputs, data):
        loss = 0
        for task_name, task_values in self.tasks.items():
            if task_name == 'preterm':
                for cutoff, cutoff_weight, ls_val, distance_weight in task_values:
                    labels = self.label_smoothing(data, cutoff, ls_val).to(self.device)
                    logits = outputs[task_name][str(cutoff)]['logits']
                    mask = data['masks'][str(cutoff)].to(self.device)
                    if mask.sum() > 0:
                        weights = self.distance_weighting(data, cutoff, alpha=distance_weight).to(self.device)
                        temp_loss = (self.loss_fns[task_name](logits, labels)*weights)[mask].mean()*cutoff_weight
                        loss += temp_loss
                        self.loss[task_name]['loss'].append(temp_loss.item())
                        self.loss[task_name]['count'].append(mask.sum().item())

            elif task_name == 'segmentation':
                continue
            
            elif task_name == 'aux_tasks':
                for var, weight, mask_val in task_values:
                    var = var if isinstance(var, str) else '_'.join(var)
                    labels = data['labels'][var].to(self.device)
                    logits = outputs[task_name][var]['logits']
                    mask = self.mask_values(labels, mask_val)
                    if mask.sum () > 0:
                        temp_loss = self.loss_fns[var](logits, labels)[mask].mean()*weight
                        loss += temp_loss
                        self.loss[task_name]['loss'].append(temp_loss.item())
                        self.loss[task_name]['count'].append(mask.sum().item())

        return loss

    def get_loss(self, reset=True):
        loss_dict = {}

        for task_name, values in self.loss.items():
            counts = values['count']
            losses = values['loss']
    
            total_count = sum(counts)
            loss_dict[task_name] = (sum(l*c for l, c in zip(losses, counts)) / total_count if total_count > 0 else 0.0)

        if reset:
            self.loss = self.loss = {task_name: {'loss': [], 'count': []} for task_name in self.tasks}

        return loss_dict

    def label_smoothing(self, data, cutoff, smoothing=2, window=2):
        ga = data['labels']['GA_weeks']
        labels = (ga < cutoff).float()
        if smoothing == 0:
            return labels

        #Centred between the last preterm week (cutoff - 1) and the first term week (cutoff)
        mask = (ga >= cutoff - window) & (ga < cutoff + window)
        soft_labels = torch.sigmoid((cutoff - 0.5 - ga) / smoothing)
        labels[mask] = soft_labels[mask]

        return labels
    
    def mask_values(self, labels, mask_value):
        if mask_value is None:
            return torch.ones_like(labels, dtype=torch.bool)
        else:
            return labels != mask_value
    
    def distance_weighting(self, data, cutoff, alpha=0.5):
        distance = torch.abs(data['labels']['GA_weeks'] - cutoff)
        weights = 1 + alpha * (distance / 6).clamp(max=1)
        return weights
    
    def update_weight(self):
        self.loss_fns['preterm'].pos_weight.mul_(self.pos_weight_multiplier)
    
    def reset_weight(self):
        self.loss_fns['preterm'].pos_weight.copy_(self.pos_weight)
    
class DiceLoss(torch.nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, target):
        probs = torch.sigmoid(logits)

        # Flatten per sample
        probs = probs.flatten(1)
        target = target.float().flatten(1)

        intersection = (probs * target).sum(dim=1)
        denominator = probs.sum(dim=1) + target.sum(dim=1)

        dice = (2 * intersection + self.smooth) / (denominator + self.smooth)

        return 1 - dice

def get_loss(cfg):
    loss_map = {'bce': torch.nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor(cfg.tasks.preterm.pos_weight,
                                                                                            dtype=torch.float32)),
                'l2': torch.nn.MSELoss(reduction='none'),
                'l1': torch.nn.L1Loss(reduction='none'),
                'dice': DiceLoss()}
    
    #Move to CUDA
    loss_map = {name: fn.to(cfg.device.type) for name, fn in loss_map.items()}
    
    losses = {}
    
    for task_name, config in cfg.tasks.items():
        tasks = config if isinstance(config, (list, ListConfig)) else [config]
        
        for task in tasks:            
            if task_name == 'aux_tasks':
                loss_name = task.var
            else:
                loss_name = task_name
                
            if isinstance(task['loss'], (list, ListConfig)):
                for name in task['loss']:
                    if name not in loss_map:
                        raise ValueError(f"Loss type '{name}' not implemented")
                
                def combined_loss(logits, labels, names=task['loss']):
                    result = 0
                    for name in names:
                        loss = loss_map[name](logits, labels)

                        # Reduce each component to one value per sample
                        if loss.ndim > 1:
                            loss = loss.flatten(1).mean(dim=1)

                        result = result + loss

                    return result.unsqueeze(1)

                losses[loss_name] = combined_loss
                
            else:
                if task['loss'] not in loss_map:
                    raise ValueError(f"Loss type '{task['loss']}' not implemented")
               
                losses[loss_name] = loss_map[task['loss']]
    
    
    return losses




    
