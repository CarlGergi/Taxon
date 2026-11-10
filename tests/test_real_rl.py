"""The constrained decoder's assumption, checked on the real tokenizer: inside a reply, a category's
tokens are exactly its standalone tokens, so the trie built from standalone paths fits."""

import random

import pytest

from taxon.fetch import fetch
from taxon.pins import TAXONOMY
from taxon.rl.decoding import CATEGORY_OPEN, _closers, category_trie, category_value, generate_one
from taxon.taxonomy import Taxonomy
from taxon.training.examples import LabeledProduct, request_row
from taxon.training.fields import CATEGORY
from taxon.training.model import load_processor, tiny_model

pytestmark = pytest.mark.data


@pytest.fixture(scope="module")
def processor():
    return load_processor()


@pytest.fixture(scope="module")
def paths():
    return sorted(random.Random(7).sample(Taxonomy.load(fetch(TAXONOMY)).paths(), 2_000))


@pytest.mark.parametrize("after", ['"}', '", "color": null}'])
def test_a_path_in_a_reply_tokenizes_as_it_does_alone(processor, paths, after):
    tokenizer = processor.tokenizer

    def encode(text):
        return tokenizer.encode(text, add_special_tokens=False)

    opening = encode(CATEGORY_OPEN)
    broken = []
    for path in paths:
        whole = encode(CATEGORY_OPEN + path + after)
        alone = encode(path)
        if (
            whole[: len(opening)] != opening
            or whole[len(opening) : len(opening) + len(alone)] != alone
        ):
            broken.append(path)
        closing = whole[len(opening) + len(alone)]
        if closing not in _closers(tokenizer):
            broken.append(path)
    assert not broken, broken[:5]


def test_constrained_decoding_with_the_real_tokenizer_gives_a_real_path(processor, paths):
    trie = category_trie(processor.tokenizer, paths)
    item = LabeledProduct(
        {"parent_asin": "A", "title": "Red tote", "features": [], "description": "", "details": {}},
        paths[0],
        {},
    )
    text = generate_one(
        tiny_model(processor, 1).eval(),
        processor,
        request_row(item, [CATEGORY], lambda _p: None),
        trie,
        80,
    )
    assert category_value(text) in set(paths)
