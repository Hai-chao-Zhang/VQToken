"""Public runtime metadata for VQToken integrations."""

import json
from pathlib import Path

VQTOKEN_RUNTIME_VERSION = "1"
VQTOKEN_CAPABILITIES = {
    "modes": ("centroids",),
    "selection_methods": ("fixed", "elbow", "silhouette"),
}


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

    index_path = model_dir / "model.safetensors.index.json"
    if index_path.is_file():
        try:
            weight_map = json.loads(index_path.read_text())["weight_map"]
        except (KeyError, OSError, ValueError):
            return False
        if not isinstance(weight_map, dict):
            return False
        return any("vision_tower.vision_tower." in key for key in weight_map)

    try:
        from safetensors import safe_open
    except ImportError:
        return False
    for checkpoint_path in model_dir.glob("*.safetensors"):
        try:
            with safe_open(checkpoint_path, framework="pt", device="cpu") as checkpoint:
                if any("vision_tower.vision_tower." in key for key in checkpoint.keys()):
                    return True
        except (OSError, ValueError):
            continue
    return False


__all__ = ["VQTOKEN_CAPABILITIES", "VQTOKEN_RUNTIME_VERSION", "has_embedded_vision_weights"]
