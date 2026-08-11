import json

from VQToken import VQTOKEN_CAPABILITIES, VQTOKEN_RUNTIME_VERSION, has_embedded_vision_weights


def test_runtime_capabilities_are_centroid_only():
    assert VQTOKEN_RUNTIME_VERSION == "1"
    assert VQTOKEN_CAPABILITIES == {
        "modes": ("centroids",),
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
