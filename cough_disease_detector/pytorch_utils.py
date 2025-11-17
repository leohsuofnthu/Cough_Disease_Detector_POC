"""
PyTorch utilities for cough detection
"""
import torch


def move_data_to_device(x, device):
    """Move data to device"""
    if isinstance(x, dict):
        for key, value in x.items():
            x[key] = move_data_to_device(value, device)
        return x
    elif isinstance(x, list):
        return [move_data_to_device(item, device) for item in x]
    elif isinstance(x, torch.Tensor):
        return x.to(device)
    else:
        return x


def do_mixup(x, mixup_lambda):
    """Mixup augmentation"""
    if mixup_lambda is not None:
        lam = mixup_lambda
        batch_size = x.size()[0]
        index = torch.randperm(batch_size).to(x.device)
        mixed_x = lam * x + (1 - lam) * x[index, :]
        return mixed_x
    else:
        return x


