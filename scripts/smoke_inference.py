#!/usr/bin/env python3
"""Run one bounded VQToken generation on the bundled demo video."""

from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path

KNOWN_REVISIONS = {
    "haichaozhang/VQ-Token-llava-ov-0.5b": "ab20666864dbe71c931a9c1236190e90493f02fe",
    "lmms-lab/llava-onevision-qwen2-0.5b-ov": "381d9947148efb1e58a577f451c05705ceec666e",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained",
        default="lmms-lab/llava-onevision-qwen2-0.5b-ov",
        help="A Hugging Face repo ID or local model directory. The default is an ungated public base checkpoint.",
    )
    parser.add_argument("--revision", default=None, help="Pinned Hugging Face revision; known defaults are pinned automatically.")
    parser.add_argument(
        "--video",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "playground/demo/xU25MMA2N4aVtYay.mp4",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--prompt", default="Describe what happens in this video in one sentence.")
    parser.add_argument("--selection", choices=["fixed", "elbow", "silhouette"], default="fixed")
    return parser.parse_args()


def resolve_model(pretrained: str, revision: str | None) -> str:
    from huggingface_hub import snapshot_download

    local_path = Path(pretrained).expanduser()
    if local_path.is_dir():
        return str(local_path.resolve())

    resolved_revision = revision or KNOWN_REVISIONS.get(pretrained)
    try:
        return snapshot_download(
            repo_id=pretrained,
            revision=resolved_revision,
            allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"],
            ignore_patterns=["global_step*", "*.pth", "training_args.bin", "trainer_state.json"],
        )
    except Exception as exc:
        message = str(exc)
        if "gated" in message.lower() or "401" in message:
            raise RuntimeError(
                f"{pretrained} is gated. Accept its access terms and set HF_TOKEN, "
                "or use the default public base checkpoint for a code-path smoke test."
            ) from exc
        raise


def load_frames(video_path: Path, frame_count: int):
    import numpy as np
    from decord import VideoReader, cpu

    if frame_count < 1:
        raise ValueError("--frames must be positive")
    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")

    reader = VideoReader(str(video_path), ctx=cpu(0))
    indices = np.linspace(0, len(reader) - 1, min(frame_count, len(reader)), dtype=int)
    return reader.get_batch(indices.tolist()).asnumpy()


def has_embedded_vision_weights(model_path: str) -> bool:
    """Check safetensors headers without materializing checkpoint tensors."""

    import json

    model_dir = Path(model_path)
    config_path = model_dir / "config.json"
    if not config_path.is_file():
        return False
    config = json.loads(config_path.read_text())
    if config.get("mm_vision_tower") != "google/siglip-so400m-patch14-384" or config.get("mm_hidden_size") != 1152:
        return False

    index_path = model_dir / "model.safetensors.index.json"
    if index_path.is_file():
        weight_map = json.loads(index_path.read_text())["weight_map"]
        return any("vision_tower.vision_tower." in key for key in weight_map)

    from safetensors import safe_open

    for checkpoint_path in model_dir.glob("*.safetensors"):
        with safe_open(checkpoint_path, framework="pt", device="cpu") as checkpoint:
            if any("vision_tower.vision_tower." in key for key in checkpoint.keys()):
                return True
    return False


def main() -> None:
    args = parse_args()

    import torch

    from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
    from llava.conversation import conv_templates
    from llava.mm_utils import tokenizer_image_token
    from llava.model.builder import load_pretrained_model

    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("This full-model smoke test requires an available CUDA GPU")

    torch.manual_seed(0)
    model_path = resolve_model(args.pretrained, args.revision)
    embedded_vision = has_embedded_vision_weights(model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path,
        None,
        "llava_qwen",
        device_map=args.device,
        torch_dtype="auto",
        attn_implementation="sdpa",
        multimodal=True,
        overwrite_config={
            "use_vqtoken": True,
            "vqtoken_mode": "centroids",
            "vqtoken_selection_method": args.selection,
            "vqtoken_min_clusters": 12,
            "vqtoken_max_clusters": 32,
            "use_embedded_vision": embedded_vision,
        },
    )
    model.eval()

    frames_np = load_frames(args.video, args.frames)
    frames = image_processor.preprocess(frames_np, return_tensors="pt")["pixel_values"].to(
        device=args.device,
        dtype=model.dtype,
    )

    conversation = copy.deepcopy(conv_templates["qwen_1_5"])
    conversation.append_message(conversation.roles[0], f"{DEFAULT_IMAGE_TOKEN}\n{args.prompt}")
    conversation.append_message(conversation.roles[1], None)
    prompt = conversation.get_prompt()
    input_ids = tokenizer_image_token(
        prompt,
        tokenizer,
        IMAGE_TOKEN_INDEX,
        return_tensors="pt",
    ).unsqueeze(0).to(args.device)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=[frames],
            image_sizes=[(int(frames_np.shape[2]), int(frames_np.shape[1]))],
            modalities=["video"],
            do_sample=False,
            temperature=0,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
        )

    output = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    print(f"model={args.pretrained}")
    print(f"revision={args.revision or KNOWN_REVISIONS.get(args.pretrained, 'local/default')}")
    print(f"frames={frames_np.shape[0]}")
    print(f"embedded_vision={embedded_vision}")
    print(f"output={output}")


if __name__ == "__main__":
    # Avoid accidentally enabling paid evaluator APIs; this script uses none.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
