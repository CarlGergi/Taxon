import pytest
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from taxon.trie import ROOT, TokenTrie


class MergingTokenizer:
    """Tokenizes greedily against a phrase list, so a string's tokens depend on what
    follows it: "Bags" is one token, while "Bag Straps" starts with a different one."""

    PIECES = [
        "Bags",
        "Bag",
        " Straps",
        " > ",
        "Apparel",
        "Home",
        " Totes",
        "Totes",
        "Kitchen",
        "Knives",
    ]

    def __init__(self):
        self.vocab = {piece: i for i, piece in enumerate(self.PIECES)}
        self.pieces = sorted(self.PIECES, key=len, reverse=True)

    def encode(self, text):
        tokens = []
        while text:
            piece = next(p for p in self.pieces if text.startswith(p))
            tokens.append(self.vocab[piece])
            text = text[len(piece) :]
        return tokens

    def decode(self, tokens):
        return "".join(self.PIECES[t] for t in tokens)


@pytest.fixture
def tokenizer():
    return MergingTokenizer()


@pytest.fixture
def trie(taxonomy, tokenizer):
    return TokenTrie.build(taxonomy.paths(), tokenizer.encode, tokenizer.decode)


def tiny_bpe(paths):
    """A real tokenizers BPE, trained on the toy paths, with the same pipeline shape as Qwen's."""
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=300, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False
    )
    tokenizer.train_from_iterator(paths, trainer)
    return tokenizer


def test_accepts_exactly_the_taxonomy(taxonomy, tokenizer, trie):
    assert trie.n_paths == len(taxonomy)
    accepted = dict(trie.sequences())
    assert sorted(accepted.values()) == taxonomy.paths()
    assert {tokens: path for tokens, path in accepted.items()} == {
        tuple(tokenizer.encode(p)): p for p in taxonomy.paths()
    }


def test_every_state_reaches_a_terminal(trie):
    def reaches_terminal(state):
        return trie.terminal(state) is not None or any(
            reaches_terminal(trie.step(state, token)) for token in trie.allowed(state)
        )

    assert all(reaches_terminal(state) for state in range(trie.n_states))


def test_a_full_walk_lands_on_the_right_path(taxonomy, tokenizer, trie):
    for path in taxonomy.paths():
        assert trie.path_for(tokenizer.encode(path)) == path


def test_internal_node_is_terminal_and_can_continue(tokenizer, trie):
    state = trie.walk(tokenizer.encode("Apparel > Bags"))
    assert trie.terminal(state) == "Apparel > Bags"
    assert trie.allowed(state) == tokenizer.encode(" > ")


def test_partial_path_is_a_state_but_not_a_terminal(tokenizer, trie):
    state = trie.walk(tokenizer.encode("Apparel > "))
    assert state is not None
    assert trie.terminal(state) is None


def test_invalid_paths_are_rejected(tokenizer, trie):
    for invalid in ["Home > Totes", "Totes", "Home > Kitchen > Bags", "Knives > Kitchen"]:
        assert trie.path_for(tokenizer.encode(invalid)) is None


def test_step_matches_walk_and_rejects_off_path_tokens(tokenizer, trie):
    tokens = tokenizer.encode("Home > Kitchen")
    state = ROOT
    for token in tokens:
        state = trie.step(state, token)
    assert state == trie.walk(tokens)
    assert trie.step(ROOT, tokenizer.encode("Knives")[0]) is None


def test_allowed_at_the_root_is_the_first_token_of_each_vertical(tokenizer, trie):
    assert trie.allowed(ROOT) == sorted(
        {tokenizer.encode("Apparel")[0], tokenizer.encode("Home")[0]}
    )


def test_branches_where_tokenization_diverges_on_a_shared_string_prefix(tokenizer, trie):
    # "Bags" and "Bag Straps" share the characters "Bag" but not a first token.
    state = trie.walk(tokenizer.encode("Apparel > "))
    assert trie.allowed(state) == sorted(
        {tokenizer.encode("Bags")[0], tokenizer.encode("Bag Straps")[0]}
    )


def test_same_name_under_two_parents_gets_two_terminals(tokenizer, trie):
    assert trie.path_for(tokenizer.encode("Apparel > Bags")) == "Apparel > Bags"
    assert trie.path_for(tokenizer.encode("Home > Bags")) == "Home > Bags"


def test_rejects_two_paths_with_one_tokenization():
    with pytest.raises(ValueError, match="share a tokenization"):
        TokenTrie.build(["a", "b"], lambda _: [1])


def test_rejects_an_empty_tokenization():
    with pytest.raises(ValueError, match="no tokens"):
        TokenTrie.build(["a"], lambda _: [])


def test_rejects_a_path_that_does_not_round_trip():
    with pytest.raises(ValueError, match="round-trip"):
        TokenTrie.build(["a"], lambda _: [1], lambda _: "b")


def test_stats_describe_the_trie_and_are_deterministic(taxonomy, tokenizer):
    first = TokenTrie.build(taxonomy.paths(), tokenizer.encode).stats()
    second = TokenTrie.build(reversed(taxonomy.paths()), tokenizer.encode).stats()
    assert first == second
    assert first["paths"] == len(taxonomy)
    assert first["max_branching"] == 2


def test_builds_on_a_real_bpe_tokenizer(taxonomy):
    bpe = tiny_bpe(taxonomy.paths())
    trie = TokenTrie.build(
        taxonomy.paths(),
        lambda text: bpe.encode(text).ids,
        lambda ids: bpe.decode(ids),
    )
    assert trie.n_paths == len(taxonomy)
    for path in taxonomy.paths():
        assert trie.path_for(bpe.encode(path).ids) == path
