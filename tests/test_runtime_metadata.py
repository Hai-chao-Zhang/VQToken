import json

from VQToken import (
    VQTOKEN_CAPABILITIES,
    VQTOKEN_RUNTIME_VERSION,
    has_embedded_vision_weights,
    has_released_vq_attention_weights,
)


def test_runtime_capabilities_include_released_attention_path():
    assert VQTOKEN_RUNTIME_VERSION == "2"
    assert VQTOKEN_CAPABILITIES == {
        "modes": ("centroids", "attention"),
        "selection_methods": ("fixed", "elbow", "silhouette"),
    }


def test_embedded_vision_detection_from_safetensors_index(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "mm_vision_tower": "google/siglip-so400m-patch14-384",
                "mm_hidden_size": 1152,
            }
        )
    )
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"model.vision_tower.vision_tower.embeddings.patch_embedding.weight": "model-00001-of-00002.safetensors"}}))

    assert has_embedded_vision_weights(str(tmp_path)) is True


def test_embedded_vision_detection_is_safe_for_unknown_paths(tmp_path):
    assert has_embedded_vision_weights(str(tmp_path / "missing")) is False
    (tmp_path / "config.json").write_text("[]")
    assert has_embedded_vision_weights(str(tmp_path)) is False


def test_released_attention_detection_from_safetensors_index(tmp_path):
    keys = {
        "model.cross_attention.to_q_proj.weight",
        "model.cross_attention.transformer_decoder.layers.0.self_attn.in_proj_weight",
        "model.cross_attention.transformer_decoder.layers.0.multihead_attn.in_proj_weight",
        "model.cross_attention.transformer_decoder.layers.0.norm3.weight",
    }
    keys.update(f"model.cross_attention.placeholder_{index}" for index in range(16))
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {key: "model.safetensors" for key in keys}})
    )

    assert has_released_vq_attention_weights(str(tmp_path)) is True


def test_released_attention_detection_fails_closed(tmp_path):
    assert has_released_vq_attention_weights(str(tmp_path / "missing")) is False
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "model.cross_attention.to_q_proj.weight": "model.safetensors",
                }
            }
        )
    )
    assert has_released_vq_attention_weights(str(tmp_path)) is False
    (tmp_path / "config.json").write_text("{}")
    assert has_embedded_vision_weights(str(tmp_path)) is False
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "mm_vision_tower": "google/siglip-so400m-patch14-384",
                "mm_hidden_size": 1152,
            }
        )
    )
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": None}))
    assert has_embedded_vision_weights(str(tmp_path)) is False
