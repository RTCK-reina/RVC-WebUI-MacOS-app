import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os


from .e2e import E2E

def get_rmvpe(model_path, device, is_half=True):
    try:
        model = E2E(4, 1, (2, 2))
        ckpt = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt)
        model.eval()
        if is_half:
            model = model.half()
        model = model.to(device)
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load RMVPE model: {str(e)}")
