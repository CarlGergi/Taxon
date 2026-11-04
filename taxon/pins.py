"""Every external input, pinned to an exact version and checksum."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
DERIVED_DIR = DATA_DIR / "derived"
RESULTS_DIR = ROOT / "results"

SEED = 20261103


@dataclass(frozen=True)
class PinnedFile:
    url: str
    sha256: str
    relative_path: str


TAXONOMY = PinnedFile(
    url="https://github.com/Shopify/product-taxonomy/releases/download/v2026-10/taxonomy.en.json.gz",
    sha256="ea03651eb8d3a967a1c18a6a10936c25fae0f967722a46fccf3707f86ca05d3e",
    relative_path="taxonomy/v2026-10/taxonomy.en.json.gz",
)


def _hf_tokenizer(model: str, revision: str, sha256: str) -> PinnedFile:
    return PinnedFile(
        url=f"https://huggingface.co/{model}/resolve/{revision}/tokenizer.json",
        sha256=sha256,
        relative_path=f"tokenizers/{model.replace('/', '--')}/{revision}/tokenizer.json",
    )


# Qwen3.5-2B and Qwen3.5-4B share this vocabulary and merge table, so one trie serves both.
TOKENIZERS = {
    "qwen3.5": _hf_tokenizer(
        "Qwen/Qwen3.5-2B",
        "15852e8c16360a2fea060d615a32b45270f8a8fc",
        "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42",
    ),
    "qwen2.5-vl": _hf_tokenizer(
        "Qwen/Qwen2.5-VL-3B-Instruct",
        "66285546d2b821cf421d4f5eb2576359d3770cd3",
        "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539",
    ),
}

AMAZON_REPO = "McAuley-Lab/Amazon-Reviews-2023"
AMAZON_REVISION = "2b6d039ed471f2ba5fd2acb718bf33b0a7e5598e"

# Byte size of each item-metadata file at AMAZON_REVISION. Left out: Unknown, which has no
# domain signal, and the digital-only or membership domains Gift_Cards, Digital_Music,
# Kindle_Store, Magazine_Subscriptions and Subscription_Boxes.
AMAZON_FILE_SIZES = {
    "All_Beauty": 212990142,
    "Amazon_Fashion": 1422365805,
    "Appliances": 285298475,
    "Arts_Crafts_and_Sewing": 2170215242,
    "Automotive": 5345616548,
    "Baby_Products": 691003271,
    "Beauty_and_Personal_Care": 2835194976,
    "Books": 14697581022,
    "CDs_and_Vinyl": 948861684,
    "Cell_Phones_and_Accessories": 4015159840,
    "Clothing_Shoes_and_Jewelry": 17960532903,
    "Electronics": 5246144134,
    "Grocery_and_Gourmet_Food": 1382105675,
    "Handmade_Products": 399261431,
    "Health_and_Household": 2468505069,
    "Health_and_Personal_Care": 118005874,
    "Home_and_Kitchen": 11788767944,
    "Industrial_and_Scientific": 1130006344,
    "Movies_and_TV": 1285744674,
    "Musical_Instruments": 631877970,
    "Office_Products": 2146468843,
    "Patio_Lawn_and_Garden": 2714096874,
    "Pet_Supplies": 1571293235,
    "Software": 256220771,
    "Sports_and_Outdoors": 4132263769,
    "Tools_and_Home_Improvement": 4850689793,
    "Toys_and_Games": 2637909430,
    "Video_Games": 437375785,
}


def amazon_url(domain: str) -> str:
    return (
        f"https://huggingface.co/datasets/{AMAZON_REPO}/resolve/{AMAZON_REVISION}"
        f"/raw/meta_categories/meta_{domain}.jsonl"
    )
