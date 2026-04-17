import pickle
from io import BytesIO
from collections import OrderedDict
import os

import torch


class _SafeJITUnpickler(pickle.Unpickler):
    """Restrict globals allowed during JIT cache unpickling.

    JIT cache files only need primitive containers plus OrderedDict metadata.
    Denying arbitrary globals reduces code-execution risk from tampered cache
    files while keeping backward compatibility with our own cache format.
    """

    _ALLOWED_GLOBALS = {
        ("collections", "OrderedDict"): OrderedDict,
    }

    def find_class(self, module, name):
        key = (module, name)
        if key in self._ALLOWED_GLOBALS:
            return self._ALLOWED_GLOBALS[key]
        raise pickle.UnpicklingError(
            f"disallowed global during JIT cache load: {module}.{name}"
        )


def load_pickle(path: str):
    with open(path, "rb") as f:
        ckpt = _SafeJITUnpickler(f).load()

    if not isinstance(ckpt, (dict, OrderedDict)):
        raise ValueError("invalid JIT cache payload: expected mapping")
    if "model" not in ckpt or not isinstance(ckpt["model"], (bytes, bytearray)):
        raise ValueError("invalid JIT cache payload: missing model bytes")
    if "is_half" in ckpt and not isinstance(ckpt["is_half"], bool):
        raise ValueError("invalid JIT cache payload: is_half must be bool")
    if "device" in ckpt and not isinstance(ckpt["device"], str):
        raise ValueError("invalid JIT cache payload: device must be string")
    return ckpt


def save_pickle(ckpt: dict, save_path: str):
    with open(save_path, "wb") as f:
        pickle.dump(ckpt, f)


def load_inputs(path: str | BytesIO, device: str, is_half=False):
    parm = torch.load(path, map_location=torch.device("cpu"))
    for key in parm.keys():
        parm[key] = parm[key].to(device)
        if is_half and parm[key].dtype == torch.float32:
            parm[key] = parm[key].half()
        elif not is_half and parm[key].dtype == torch.float16:
            parm[key] = parm[key].float()
    return parm


def export_jit_model(
    model: torch.nn.Module,
    mode: str = "trace",
    inputs: dict = None,
    device=torch.device("cpu"),
    is_half: bool = False,
) -> dict:
    model = model.half() if is_half else model.float()
    model.eval()
    if mode == "trace":
        assert inputs is not None
        model_jit = torch.jit.trace(model, example_kwarg_inputs=inputs)
    elif mode == "script":
        model_jit = torch.jit.script(model)
    model_jit.to(device)
    model_jit = model_jit.half() if is_half else model_jit.float()
    buffer = BytesIO()
    # model_jit=model_jit.cpu()
    torch.jit.save(model_jit, buffer)
    del model_jit
    cpt = OrderedDict()
    cpt["model"] = buffer.getvalue()
    cpt["is_half"] = is_half
    return cpt


def get_jit_model(model_path: str, is_half: bool, device: str, exporter):
    jit_model_path = model_path.rstrip(".pth")
    jit_model_path += ".half.jit" if is_half else ".jit"
    ckpt = None

    if os.path.exists(jit_model_path):
        ckpt = load_pickle(jit_model_path)
        model_device = ckpt["device"]
        if model_device != str(device):
            del ckpt
            ckpt = None

    if ckpt is None:
        ckpt = exporter(
            model_path=model_path,
            mode="script",
            inputs_path=None,
            save_path=jit_model_path,
            device=device,
            is_half=is_half,
        )

    return ckpt
