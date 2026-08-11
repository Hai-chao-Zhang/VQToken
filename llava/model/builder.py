#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import os
import warnings

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, PretrainedConfig
import torch
# from ..llava.model import *
from llava.model import *
from llava.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.utils import rank0_print


def _load_config(config_cls, model_path, customized_config=None, overwrite_config=None):
    """Build one concrete LLaVA config and apply runtime overrides once."""

    if customized_config is None:
        config = config_cls.from_pretrained(model_path)
    elif isinstance(customized_config, config_cls):
        config = customized_config
    elif isinstance(customized_config, PretrainedConfig):
        config = config_cls.from_dict(customized_config.to_dict())
    elif isinstance(customized_config, dict):
        config = config_cls.from_dict(customized_config)
    else:
        # A directory, Hub id, or explicit JSON file is accepted by
        # PretrainedConfig.from_pretrained.
        config = config_cls.from_pretrained(customized_config)

    if overwrite_config:
        rank0_print(f"Overwriting config with {overwrite_config}")
        for key, value in overwrite_config.items():
            setattr(config, key, value)
    return config


def _infer_model_family(model_path, model_name, customized_config=None):
    """Augment ambiguous checkpoint names using config architectures."""

    lowered = model_name.lower()
    families = ("mixtral", "mistral", "zephyr", "qwen", "quyen", "gemma", "llama")
    if any(family in lowered for family in families):
        return model_name

    try:
        if isinstance(customized_config, PretrainedConfig):
            config_dict = customized_config.to_dict()
        elif isinstance(customized_config, dict):
            config_dict = customized_config
        else:
            config_source = customized_config or model_path
            config_dict, _ = PretrainedConfig.get_config_dict(config_source)
        architecture_hint = " ".join(config_dict.get("architectures") or []).lower()
        if "qwenmoe" in architecture_hint or "qwen_moe" in architecture_hint:
            return f"{model_name}_qwen_moe"
        for family in ("mixtral", "mistral", "qwen", "gemma", "llama"):
            if family in architecture_hint:
                return f"{model_name}_{family}"
    except Exception as exc:
        rank0_print(f"Could not infer model family from config: {exc}")
    return model_name


def _resolve_checkpoint_file(model_path, filename):
    """Resolve an auxiliary checkpoint without treating local paths as Hub ids."""

    if os.path.isdir(model_path):
        local_path = os.path.join(model_path, filename)
        if not os.path.isfile(local_path):
            raise FileNotFoundError(f"Required checkpoint file not found: {local_path}")
        return local_path

    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=model_path, filename=filename)


def load_pretrained_model(model_path, model_base, model_name, load_8bit=False, load_4bit=False, device_map="auto", torch_dtype="float16", attn_implementation="sdpa", customized_config=None, overwrite_config=None, **kwargs):
    kwargs["device_map"] = device_map

    if load_8bit:
        kwargs["load_in_8bit"] = True
    elif load_4bit:
        kwargs["load_in_4bit"] = True
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
    elif torch_dtype == "float16":
        kwargs["torch_dtype"] = torch.float16
    elif torch_dtype == "bfloat16":
        kwargs["torch_dtype"] = torch.bfloat16
    elif torch_dtype == "auto":
        kwargs["torch_dtype"] = "auto"
    elif isinstance(torch_dtype, torch.dtype):
        kwargs["torch_dtype"] = torch_dtype
    else:
        raise ValueError(f"Unsupported torch_dtype: {torch_dtype!r}")

    if "config" in kwargs:
        if customized_config is not None:
            raise ValueError("Pass only one of customized_config or config")
        customized_config = kwargs.pop("config")

    is_multimodal = bool(kwargs.pop("multimodal", False))
    model_name = model_name or os.path.basename(str(model_path).rstrip("/"))
    model_name = _infer_model_family(model_path, model_name, customized_config)
    lower_model_name = model_name.lower()

    if "llava" in model_name.lower() or is_multimodal:
        # Load LLaVA model
        if "lora" in model_name.lower() and model_base is None:
            warnings.warn(
                "There is `lora` in model name but no `model_base` is provided. If you are loading a LoRA model, please provide the `model_base` argument. Detailed instruction: https://github.com/haotian-liu/LLaVA#launch-a-model-worker-lora-weights-unmerged."
            )
        if "lora" in model_name.lower() and model_base is not None:
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            rank0_print("Loading LLaVA from base model...")
            if "mixtral" in model_name.lower():
                from llava.model.language_model.llava_mixtral import LlavaMixtralConfig

                lora_cfg_pretrained = _load_config(
                    LlavaMixtralConfig, model_path, customized_config, overwrite_config
                )
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
                model = LlavaMixtralForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, attn_implementation=attn_implementation, **kwargs)
            elif "mistral" in model_name.lower():
                from llava.model.language_model.llava_mistral import LlavaMistralConfig

                lora_cfg_pretrained = _load_config(
                    LlavaMistralConfig, model_path, customized_config, overwrite_config
                )
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
                model = LlavaMistralForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, attn_implementation=attn_implementation, **kwargs)
            elif "gemma" in model_name.lower():
                from llava.model.language_model.llava_gemma import LlavaGemmaConfig

                lora_cfg_pretrained = _load_config(
                    LlavaGemmaConfig, model_path, customized_config, overwrite_config
                )
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
                model = LlavaGemmaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, attn_implementation=attn_implementation, **kwargs)
            elif "qwen" in model_name.lower() or "quyen" in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
                if "moe" in lower_model_name or "a14b" in lower_model_name:
                    from llava.model.language_model.llava_qwen_moe import LlavaQwenMoeConfig

                    lora_cfg_pretrained = _load_config(
                        LlavaQwenMoeConfig, model_path, customized_config, overwrite_config
                    )
                    model = LlavaQwenMoeForCausalLM.from_pretrained(
                        model_base,
                        low_cpu_mem_usage=True,
                        config=lora_cfg_pretrained,
                        attn_implementation=attn_implementation,
                        **kwargs,
                    )
                else:
                    from llava.model.language_model.llava_qwen import LlavaQwenConfig

                    lora_cfg_pretrained = _load_config(
                        LlavaQwenConfig, model_path, customized_config, overwrite_config
                    )
                    model = LlavaQwenForCausalLM.from_pretrained(
                        model_base,
                        low_cpu_mem_usage=True,
                        config=lora_cfg_pretrained,
                        attn_implementation=attn_implementation,
                        **kwargs,
                    )
            else:
                from llava.model.language_model.llava_llama import LlavaConfig

                lora_cfg_pretrained = _load_config(
                    LlavaConfig, model_path, customized_config, overwrite_config
                )
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
                model = LlavaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, attn_implementation=attn_implementation, **kwargs)

            token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
            if model.lm_head.weight.shape[0] != token_num:
                model.lm_head.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
                model.model.embed_tokens.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))

            rank0_print("Loading additional LLaVA weights...")
            non_lora_trainables = torch.load(
                _resolve_checkpoint_file(model_path, "non_lora_trainables.bin"),
                map_location="cpu",
                weights_only=True,
            )
            non_lora_trainables = {(k[11:] if k.startswith("base_model.") else k): v for k, v in non_lora_trainables.items()}
            if any(k.startswith("model.model.") for k in non_lora_trainables):
                non_lora_trainables = {(k[6:] if k.startswith("model.") else k): v for k, v in non_lora_trainables.items()}
            model.load_state_dict(non_lora_trainables, strict=False)

            from peft import PeftModel

            rank0_print("Loading LoRA weights...")
            model = PeftModel.from_pretrained(model, model_path)
            rank0_print("Merging LoRA weights...")
            model = model.merge_and_unload()
            rank0_print("Model is loaded...")
        elif model_base is not None:  # this may be mm projector only, loading projector with preset language mdoel
            rank0_print(f"Loading LLaVA from base model {model_base}...")
            if "mixtral" in model_name.lower():
                from llava.model.language_model.llava_mixtral import LlavaMixtralConfig

                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
                cfg_pretrained = _load_config(
                    LlavaMixtralConfig, model_path, customized_config, overwrite_config
                )
                model = LlavaMixtralForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, attn_implementation=attn_implementation, **kwargs)
            elif "mistral" in model_name.lower() or "zephyr" in model_name.lower():
                from llava.model.language_model.llava_mistral import LlavaMistralConfig

                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
                cfg_pretrained = _load_config(
                    LlavaMistralConfig, model_path, customized_config, overwrite_config
                )
                model = LlavaMistralForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, attn_implementation=attn_implementation, **kwargs)
            elif "gemma" in model_name.lower():
                from llava.model.language_model.llava_gemma import LlavaGemmaConfig

                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
                cfg_pretrained = _load_config(
                    LlavaGemmaConfig, model_path, customized_config, overwrite_config
                )
                model = LlavaGemmaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, attn_implementation=attn_implementation, **kwargs)
            elif "qwen" in model_name.lower() or "quyen" in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
                if "moe" in lower_model_name or "a14b" in lower_model_name:
                    from llava.model.language_model.llava_qwen_moe import LlavaQwenMoeConfig

                    cfg_pretrained = _load_config(
                        LlavaQwenMoeConfig, model_path, customized_config, overwrite_config
                    )
                    model = LlavaQwenMoeForCausalLM.from_pretrained(
                        model_base,
                        low_cpu_mem_usage=True,
                        config=cfg_pretrained,
                        attn_implementation=attn_implementation,
                        **kwargs,
                    )
                else:
                    from llava.model.language_model.llava_qwen import LlavaQwenConfig

                    cfg_pretrained = _load_config(
                        LlavaQwenConfig, model_path, customized_config, overwrite_config
                    )
                    model = LlavaQwenForCausalLM.from_pretrained(
                        model_base,
                        low_cpu_mem_usage=True,
                        config=cfg_pretrained,
                        attn_implementation=attn_implementation,
                        **kwargs,
                    )
            elif (
                "wizardlm-2" in model_name.lower()
                and "vicuna" in model_name.lower()
                or "llama" in model_name.lower()
                or "yi" in model_name.lower()
                or "nous-hermes" in model_name.lower()
                or "llava-v1.6-34b" in model_name.lower()
                or "llava-v1.5" in model_name.lower()
            ):
                from llava.model.language_model.llava_llama import LlavaConfig

                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
                llava_cfg = _load_config(
                    LlavaConfig, model_path, customized_config, overwrite_config
                )
                if "v1.5" in model_name.lower():
                    llava_cfg.delay_load = True
                model = LlavaLlamaForCausalLM.from_pretrained(
                    model_base,
                    low_cpu_mem_usage=True,
                    config=llava_cfg,
                    attn_implementation=attn_implementation,
                    **kwargs,
                )
            else:
                raise ValueError(f"Model {model_name} not supported")

            mm_projector_path = _resolve_checkpoint_file(model_path, "mm_projector.bin")
            mm_projector_weights = torch.load(
                mm_projector_path,
                map_location="cpu",
                weights_only=True,
            )
            mm_projector_weights = {k: v.to(torch.float16) for k, v in mm_projector_weights.items()}
            model.load_state_dict(mm_projector_weights, strict=False)
        else:
            rank0_print(f"Loaded LLaVA model: {model_path}")
            if "mixtral" in model_name.lower():
                from llava.model.language_model.llava_mixtral import LlavaMixtralConfig

                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                llava_cfg = _load_config(
                    LlavaMixtralConfig,
                    model_path,
                    customized_config,
                    overwrite_config,
                )

                tokenizer = AutoTokenizer.from_pretrained(model_path)
                model = LlavaMixtralForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, attn_implementation=attn_implementation, config=llava_cfg, **kwargs)

            elif "mistral" in model_name.lower() or "zephyr" in model_name.lower():
                from llava.model.language_model.llava_mistral import LlavaMistralConfig

                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                llava_cfg = _load_config(
                    LlavaMistralConfig,
                    model_path,
                    customized_config,
                    overwrite_config,
                )
                model = LlavaMistralForCausalLM.from_pretrained(
                    model_path,
                    low_cpu_mem_usage=True,
                    attn_implementation=attn_implementation,
                    config=llava_cfg,
                    **kwargs,
                )
            elif (
                "wizardlm-2" in model_name.lower()
                and "vicuna" in model_name.lower()
                or "llama" in model_name.lower()
                or "yi" in model_name.lower()
                or "nous-hermes" in model_name.lower()
                or "llava-v1.6-34b" in model_name.lower()
                or "llava-v1.5" in model_name.lower()
            ):
                from llava.model.language_model.llava_llama import LlavaConfig

                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                llava_cfg = _load_config(
                    LlavaConfig,
                    model_path,
                    customized_config,
                    overwrite_config,
                )
                if "v1.5" in model_name.lower():
                    llava_cfg.delay_load = True  # a workaround for correctly loading v1.5 models

                model = LlavaLlamaForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, attn_implementation=attn_implementation, config=llava_cfg, **kwargs)

            elif "qwen" in model_name.lower() or "quyen" in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                if "moe" in lower_model_name or "a14b" in lower_model_name:
                    from llava.model.language_model.llava_qwen_moe import LlavaQwenMoeConfig
                    llava_cfg = _load_config(
                        LlavaQwenMoeConfig,
                        model_path,
                        customized_config,
                        overwrite_config,
                    )
                    model = LlavaQwenMoeForCausalLM.from_pretrained(
                        model_path,
                        low_cpu_mem_usage=True,
                        attn_implementation=attn_implementation,
                        config=llava_cfg,
                        **kwargs,
                    )

                else:
                    from llava.model.language_model.llava_qwen import LlavaQwenConfig
                    llava_cfg = _load_config(
                        LlavaQwenConfig,
                        model_path,
                        customized_config,
                        overwrite_config,
                    )
                    model = LlavaQwenForCausalLM.from_pretrained(
                        model_path,
                        low_cpu_mem_usage=True,
                        attn_implementation=attn_implementation,
                        config=llava_cfg,
                        **kwargs,
                    )

            elif "gemma" in model_name.lower():
                from llava.model.language_model.llava_gemma import LlavaGemmaConfig

                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                llava_cfg = _load_config(
                    LlavaGemmaConfig,
                    model_path,
                    customized_config,
                    overwrite_config,
                )
                model = LlavaGemmaForCausalLM.from_pretrained(
                    model_path,
                    low_cpu_mem_usage=True,
                    config=llava_cfg,
                    attn_implementation=attn_implementation,
                    **kwargs,
                )
            else:
                try:
                    from llava.model.language_model.llava_llama import LlavaConfig

                    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                    llava_cfg = _load_config(
                        LlavaConfig,
                        model_path,
                        customized_config,
                        overwrite_config,
                    )
                    if "v1.5" in model_path.lower():
                        llava_cfg.delay_load = True  # a workaround for correctly loading v1.5 models
                    model = LlavaLlamaForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, attn_implementation=attn_implementation, config=llava_cfg, **kwargs)
                except Exception as exc:
                    raise ValueError(f"Model {model_name} not supported") from exc

    else:
        # Load language model
        if model_base is not None:
            # PEFT model
            from peft import PeftModel

            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            model = AutoModelForCausalLM.from_pretrained(model_base, torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map="auto")
            print(f"Loading LoRA weights from {model_path}")
            model = PeftModel.from_pretrained(model, model_path)
            print(f"Merging weights")
            model = model.merge_and_unload()
            print("Convert to FP16...")
            model.to(torch.float16)
        else:
            use_fast = False
            if "mpt" in model_name.lower().replace("prompt", ""):
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, trust_remote_code=True, **kwargs)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)

    rank0_print(f"Model Class: {model.__class__.__name__}")
    image_processor = None

    if "llava" in model_name.lower() or is_multimodal:
        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
        if mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
        model.resize_token_embeddings(len(tokenizer))

        vision_tower = model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model(device_map=device_map)
        if device_map != "auto":
            vision_tower.to(device=model.device, dtype=model.dtype)
        image_processor = vision_tower.image_processor

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    elif hasattr(model.config, "max_position_embeddings"):
        context_len = model.config.max_position_embeddings
    elif hasattr(model.config, "tokenizer_model_max_length"):
        context_len = model.config.tokenizer_model_max_length
    else:
        context_len = 2048

    return tokenizer, model, image_processor, context_len
