"""
Loss functions for cough detection
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_loss_func(loss_type):
    """Get loss function"""
    if loss_type == 'clip_bce':
        return clip_bce
    elif loss_type == 'clip_nll':
        return clip_nll
    else:
        raise Exception('Incorrect loss type!')


def clip_bce(output_dict, target_dict):
    """Binary cross entropy loss"""
    clipwise_output = output_dict['clipwise_output']
    target = target_dict['target']
    
    loss = F.binary_cross_entropy(clipwise_output, target)
    return loss


def clip_nll(output_dict, target_dict):
    """Negative log likelihood loss"""
    clipwise_output = output_dict['clipwise_output']
    target = target_dict['target']
    
    # Convert target to class indices for NLL loss
    if target.dim() > 1 and target.size(1) > 1:
        target = torch.argmax(target, dim=1)
    
    loss = F.nll_loss(clipwise_output, target)
    return loss
