from unittest.mock import patch

import pytest

from llava.model.builder import _resolve_checkpoint_file


def test_local_checkpoint_file_is_resolved_without_hub_access(tmp_path):
    checkpoint = tmp_path / "mm_projector.bin"
    checkpoint.write_bytes(b"test")

    with patch("huggingface_hub.hf_hub_download") as hub_download:
        assert _resolve_checkpoint_file(str(tmp_path), checkpoint.name) == str(checkpoint)

    hub_download.assert_not_called()


def test_missing_local_checkpoint_fails_without_hub_access(tmp_path):
    with patch("huggingface_hub.hf_hub_download") as hub_download:
        with pytest.raises(FileNotFoundError, match="non_lora_trainables.bin"):
            _resolve_checkpoint_file(str(tmp_path), "non_lora_trainables.bin")

    hub_download.assert_not_called()


def test_hub_checkpoint_uses_huggingface_download():
    with patch("huggingface_hub.hf_hub_download", return_value="/cache/mm_projector.bin") as hub_download:
        result = _resolve_checkpoint_file("org/model", "mm_projector.bin")

    assert result == "/cache/mm_projector.bin"
    hub_download.assert_called_once_with(repo_id="org/model", filename="mm_projector.bin")
