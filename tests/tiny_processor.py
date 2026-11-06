"""A real processor with a byte-level vocabulary, so training tests need no downloads."""

from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast
from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor
from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor
from transformers.models.qwen3_vl.video_processing_qwen3_vl import Qwen3VLVideoProcessor

SPECIAL = [
    "<|endoftext|>",
    "<|im_start|>",
    "<|im_end|>",
    "<|vision_start|>",
    "<|vision_end|>",
    "<|image_pad|>",
    "<|video_pad|>",
]

# The shape of the real template, for the parts training uses.
TEMPLATE = """\
{%- for message in messages -%}
{{- '<|im_start|>' + message.role + '\n' -}}
{%- for part in message.content -%}
{%- if part.type == 'image' -%}{{- '<|vision_start|><|image_pad|><|vision_end|>' -}}
{%- else -%}{{- part.text -}}{%- endif -%}
{%- endfor -%}
{{- '<|im_end|>\n' -}}
{%- endfor -%}
{%- if add_generation_prompt -%}
{{- '<|im_start|>assistant\n' -}}
{%- if enable_thinking is defined and enable_thinking is true -%}{{- '<think>\n' -}}
{%- else -%}{{- '<think>\n\n</think>\n\n' -}}{%- endif -%}
{%- endif -%}
"""


def tiny_processor():
    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    tokenizer = Tokenizer(models.BPE(vocab={c: i for i, c in enumerate(alphabet)}, merges=[]))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.add_special_tokens(SPECIAL)
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, pad_token="<|endoftext|>", eos_token="<|im_end|>"
    )
    sizes = {
        "size": {"shortest_edge": 32 * 32, "longest_edge": 64 * 64},
        "patch_size": 16,
        "temporal_patch_size": 2,
        "merge_size": 2,
        "image_mean": [0.5, 0.5, 0.5],
        "image_std": [0.5, 0.5, 0.5],
    }
    return Qwen3VLProcessor(
        image_processor=Qwen2VLImageProcessor(**sizes),
        tokenizer=fast,
        video_processor=Qwen3VLVideoProcessor(**sizes),
        chat_template=TEMPLATE,
    )
