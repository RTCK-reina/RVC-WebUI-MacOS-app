"""Builder for the RMVPE pitch-estimation network.

The earlier macOS port shipped a Conv2d placeholder here that never matched
the real 741-tensor ``rmvpe.pt`` state dict, so every F0 extraction using
the ``rmvpe`` method raised ``RuntimeError: Failed to load model`` on load.

This rebuilds the correct architecture: a DeepUnet (5-level encoder/decoder
with 4 ConvBlockRes per level, 4 intermediate blocks, 16 base channels) fed
into a bidirectional GRU plus Linear+Sigmoid head — exactly the 741-parameter
layout of the upstream RMVPE checkpoint. Hyperparameters verified by an
exact ``load_state_dict(..., strict=True)`` match against ``rmvpe.pt``.
"""

import os

import torch

from .e2e import E2E


def get_rmvpe(model_path, device, is_half=True):
    """Load the RMVPE pitch model from ``rmvpe.pt`` onto ``device``.

    Returns the underlying :class:`E2E` module. For end-user F0 extraction
    use :class:`rvc.f0.rmvpe.RMVPE`, which wraps this model with the mel
    front-end, decode grid, and the F0Predictor contract.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"RMVPE model file not found: {model_path}")

    model = E2E(n_blocks=4, n_gru=1, kernel_size=(2, 2))
    state_dict = torch.load(model_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state_dict)
    model.eval()
    if is_half:
        model = model.half()
    return model.to(device)
