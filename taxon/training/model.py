"""The models: tiny random weights for development on CPU, the pinned model behind the GPU flag."""

from __future__ import annotations

import torch
from peft import LoraConfig
from transformers import AutoModelForImageTextToText, AutoProcessor, Qwen3_5Config, set_seed

from taxon.pins import MODEL, MODEL_REVISION

TINY = "tiny-random"
# Attention projections in the text model only. The vision tower stays frozen.
LORA_TARGETS = r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|in_proj_qkv|out_proj)"
LORA_RANK = 16
_TINY_HIDDEN = 32


def load_processor():
    return AutoProcessor.from_pretrained(MODEL, revision=MODEL_REVISION)


def tiny_model(processor, seed: int):
    """Random weights with the real architecture, sized to `processor`'s vocabulary.

    The same seed gives the same weights, so a run can be reloaded from its adapter alone.
    """
    tokenizer = processor.tokenizer
    config = Qwen3_5Config(
        text_config={
            "vocab_size": len(tokenizer),
            "hidden_size": _TINY_HIDDEN,
            "intermediate_size": 2 * _TINY_HIDDEN,
            "num_hidden_layers": 4,
            "layer_types": ["linear_attention"] * 3 + ["full_attention"],
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": _TINY_HIDDEN,
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 2,
            "linear_key_head_dim": 8,
            "linear_value_head_dim": 8,
            "max_position_embeddings": 4096,
            "eos_token_id": tokenizer.convert_tokens_to_ids("<|im_end|>"),
            "pad_token_id": tokenizer.pad_token_id,
            "tie_word_embeddings": True,
            "rope_parameters": {
                "rope_type": "default",
                "rope_theta": 10_000_000,
                "partial_rotary_factor": 0.25,
                "mrope_section": [2, 1, 1],
                "mrope_interleaved": True,
            },
        },
        vision_config={
            "depth": 1,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_heads": 2,
            "out_hidden_size": _TINY_HIDDEN,
        },
        image_token_id=tokenizer.convert_tokens_to_ids("<|image_pad|>"),
        video_token_id=tokenizer.convert_tokens_to_ids("<|video_pad|>"),
        vision_start_token_id=tokenizer.convert_tokens_to_ids("<|vision_start|>"),
        vision_end_token_id=tokenizer.convert_tokens_to_ids("<|vision_end|>"),
        tie_word_embeddings=True,
    )
    set_seed(seed)
    return AutoModelForImageTextToText.from_config(config, dtype=torch.float32)


def real_model():
    return AutoModelForImageTextToText.from_pretrained(
        MODEL, revision=MODEL_REVISION, dtype=torch.bfloat16, device_map={"": 0}
    )


def lora_config(rank: int = LORA_RANK) -> LoraConfig:
    return LoraConfig(
        r=rank,
        lora_alpha=2 * rank,
        lora_dropout=0.05,
        target_modules=LORA_TARGETS,
        task_type="CAUSAL_LM",
    )
