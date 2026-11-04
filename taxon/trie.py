"""A trie over the token sequences of every valid category path.

The trie is built from full-path tokenizations rather than by joining per-segment
tokens, because a BPE tokenizer does not tokenize a string independently of what
surrounds it. Paths are valid stopping points at every category, internal nodes
included, so a state can be terminal and still have children.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable, Iterator, Sequence

ROOT = 0


class TokenTrie:
    def __init__(self) -> None:
        self._children: list[dict[int, int]] = [{}]
        self._terminal: list[str | None] = [None]
        self.n_paths = 0

    @classmethod
    def build(
        cls,
        paths: Iterable[str],
        encode: Callable[[str], Sequence[int]],
        decode: Callable[[Sequence[int]], str] | None = None,
    ) -> TokenTrie:
        """Build from `paths`. With `decode`, also require each path to round-trip."""
        trie = cls()
        for path in paths:
            tokens = list(encode(path))
            if decode is not None and decode(tokens) != path:
                raise ValueError(f"path does not round-trip through the tokenizer: {path!r}")
            trie.insert(tokens, path)
        return trie

    def insert(self, tokens: Sequence[int], path: str) -> None:
        if not tokens:
            raise ValueError(f"path has no tokens: {path!r}")
        state = ROOT
        for token in tokens:
            next_state = self._children[state].get(token)
            if next_state is None:
                next_state = len(self._children)
                self._children[state][token] = next_state
                self._children.append({})
                self._terminal.append(None)
            state = next_state
        if self._terminal[state] is not None:
            raise ValueError(f"paths share a tokenization: {self._terminal[state]!r} and {path!r}")
        self._terminal[state] = path
        self.n_paths += 1

    def step(self, state: int, token: int) -> int | None:
        """The state after `token`, or None if no valid path continues that way."""
        return self._children[state].get(token)

    def walk(self, tokens: Iterable[int], state: int = ROOT) -> int | None:
        for token in tokens:
            state = self._children[state].get(token)
            if state is None:
                return None
        return state

    def allowed(self, state: int) -> list[int]:
        """The tokens that keep a generation inside the set of valid paths."""
        return sorted(self._children[state])

    def terminal(self, state: int) -> str | None:
        """The category path that ends at `state`, if stopping here is valid."""
        return self._terminal[state]

    def path_for(self, tokens: Iterable[int]) -> str | None:
        state = self.walk(tokens)
        return None if state is None else self._terminal[state]

    def sequences(self) -> Iterator[tuple[tuple[int, ...], str]]:
        """Every accepted token sequence with its path, in sorted token order."""
        stack: list[tuple[int, tuple[int, ...]]] = [(ROOT, ())]
        while stack:
            state, prefix = stack.pop()
            path = self._terminal[state]
            if path is not None:
                yield prefix, path
            for token in sorted(self._children[state], reverse=True):
                stack.append((self._children[state][token], (*prefix, token)))

    @property
    def n_states(self) -> int:
        return len(self._children)

    @property
    def max_branching(self) -> int:
        return max(len(children) for children in self._children)

    def stats(self) -> dict:
        lengths = [len(tokens) for tokens, _ in self.sequences()]
        digest = hashlib.sha256()
        for tokens, path in self.sequences():
            digest.update(f"{','.join(map(str, tokens))}\t{path}\n".encode())
        return {
            "paths": self.n_paths,
            "states": self.n_states,
            "max_tokens": max(lengths),
            "mean_tokens": round(sum(lengths) / len(lengths), 3),
            "max_branching": self.max_branching,
            "sha256": digest.hexdigest(),
        }
