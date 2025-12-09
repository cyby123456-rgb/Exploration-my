import logging, torch
from typing import List, Optional, Sequence, Union
from transformers import PreTrainedModel
logger = logging.getLogger(__name__)

def _get_decoder_layers(model: PreTrainedModel):
    for root_name in ("model", "transformer", "decoder", "backbone"):
        root = getattr(model, root_name, None)
        if root is None:
            continue
        for layers_name in ("layers", "h", "block"):
            layers = getattr(root, layers_name, None)
            if layers is not None:
                return layers
    raise AttributeError("Cannot locate decoder layers for noise injection")

def register_hidden_state_noise(
    model: PreTrainedModel,
    *,
    std: float,
    layer_idx: Union[int, Sequence[int], None] = None,
    apply_phase: str = "train",
    train_only: Optional[bool] = None,
    all_layers: bool = False,
):
    if std is None or std <= 0:
        return None

    if train_only is not None:
        apply_phase = "train" if train_only else "both"
    apply_phase = (apply_phase or "train").lower()
    if apply_phase not in ("train", "eval", "both"):
        raise ValueError(f"Unknown apply_phase={apply_phase}")

    layers = _get_decoder_layers(model)
    num_layers = len(layers)
    target_indices: List[int]
    if all_layers:
        target_indices = list(range(num_layers))
    elif isinstance(layer_idx, (list, tuple)):
        target_indices = []
        for idx in layer_idx:
            if idx is None:
                continue
            idx_int = int(idx)
            if idx_int < 0:
                idx_int += num_layers
            target_indices.append(idx_int)
    else:
        idx = layer_idx if layer_idx is not None else num_layers // 2
        idx = int(idx)
        if idx < 0:
            idx += num_layers
        target_indices = [idx]

    def _hook(module, inputs, output):
        if apply_phase == "train" and not module.training:
            return output
        if apply_phase == "eval" and module.training:
            return output
        noise = lambda x: x + torch.randn_like(x) * std
        if isinstance(output, tuple):
            head, *tail = output
            return (noise(head), *tail)
        return noise(output)

    handles: List = []
    for idx in target_indices:
        target_layer = layers[idx]
        handle = target_layer.register_forward_hook(_hook)
        handles.append(handle)
    logger.info("Injecting Gaussian noise std=%s on layers %s (phase=%s)", std, target_indices, apply_phase)
    if len(handles) == 1:
        return handles[0]
    return handles
