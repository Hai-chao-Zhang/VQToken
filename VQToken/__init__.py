"""Public runtime metadata for VQToken integrations."""

import json
from pathlib import Path

VQTOKEN_RUNTIME_VERSION = "2"
VQTOKEN_CAPABILITIES = {
    "modes": ("centroids", "attention"),
    "selection_methods": ("fixed", "elbow", "silhouette"),
}

_RELEASED_ATTENTION_KEYS = {
    "model.cross_attention.to_q_proj.bias",
    "model.cross_attention.to_q_proj.weight",
    "model.cross_attention.transformer_decoder.layers.0.linear1.bias",
    "model.cross_attention.transformer_decoder.layers.0.linear1.weight",
    "model.cross_attention.transformer_decoder.layers.0.linear2.bias",
    "model.cross_attention.transformer_decoder.layers.0.linear2.weight",
    "model.cross_attention.transformer_decoder.layers.0.multihead_attn.in_proj_bias",
    "model.cross_attention.transformer_decoder.layers.0.self_attn.in_proj_weight",
    "model.cross_attention.transformer_decoder.layers.0.multihead_attn.in_proj_weight",
    "model.cross_attention.transformer_decoder.layers.0.multihead_attn.out_proj.bias",
    "model.cross_attention.transformer_decoder.layers.0.multihead_attn.out_proj.weight",
    "model.cross_attention.transformer_decoder.layers.0.norm1.bias",
    "model.cross_attention.transformer_decoder.layers.0.norm1.weight",
    "model.cross_attention.transformer_decoder.layers.0.norm2.bias",
    "model.cross_attention.transformer_decoder.layers.0.norm2.weight",
    "model.cross_attention.transformer_decoder.layers.0.norm3.bias",
    "model.cross_attention.transformer_decoder.layers.0.norm3.weight",
    "model.cross_attention.transformer_decoder.layers.0.self_attn.in_proj_bias",
    "model.cross_attention.transformer_decoder.layers.0.self_attn.out_proj.bias",
    "model.cross_attention.transformer_decoder.layers.0.self_attn.out_proj.weight",
}


def _local_checkpoint_keys(model_path: str) -> set[str]:
    """Read local checkpoint headers without materializing model tensors."""

    model_dir = Path(model_path)
    if not model_dir.is_dir():
        return set()
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.is_file():
        try:
            weight_map = json.loads(index_path.read_text())["weight_map"]
        except (KeyError, OSError, ValueError):
            return set()
        return set(weight_map) if isinstance(weight_map, dict) else set()

    try:
        from safetensors import SafetensorError, safe_open
    except ImportError:
        return set()
    keys: set[str] = set()
    for checkpoint_path in model_dir.glob("*.safetensors"):
        try:
            with safe_open(checkpoint_path, framework="pt", device="cpu") as checkpoint:
                keys.update(checkpoint.keys())
        except (OSError, ValueError, SafetensorError):
            return set()
    return keys


def has_released_vq_attention_weights(model_path: str) -> bool:
    """Return whether a local checkpoint contains the released VQ-Attention."""

    keys = _local_checkpoint_keys(model_path)
    attention_keys = {key for key in keys if key.startswith("model.cross_attention.")}
    return attention_keys == _RELEASED_ATTENTION_KEYS


def has_embedded_vision_weights(model_path: str) -> bool:
    """Inspect local checkpoint headers without materializing their tensors."""
    model_dir = Path(model_path)
    config_path = model_dir / "config.json"
    if not config_path.is_file():
        return False
    try:
        config = json.loads(config_path.read_text())
    except (OSError, ValueError):
        return False
    if not isinstance(config, dict):
        return False
    if config.get("mm_vision_tower") != "google/siglip-so400m-patch14-384" or config.get("mm_hidden_size") != 1152:
        return False

    return any("vision_tower.vision_tower." in key for key in _local_checkpoint_keys(model_path))


__all__ = [
    "VQTOKEN_CAPABILITIES",
    "VQTOKEN_RUNTIME_VERSION",
    "has_embedded_vision_weights",
    "has_released_vq_attention_weights",
]
