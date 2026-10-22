"""YAML loading that a hostile file cannot blow up.

yaml.safe_load accepts aliases, so a few hundred bytes of nested aliases ("billion
laughs") expand into billions of values as soon as anything walks the result. Here
the document's expanded size is measured before any value is built, and recursive
aliases and runaway nesting are refused. Ordinary files (no aliases, or a few) are
unaffected.
"""

from __future__ import annotations

from typing import Any, Iterator

import yaml

MAX_NODES = 1_000_000  # values after expanding every alias


class YAMLTooLarge(yaml.YAMLError):
    pass


def _expanded_size(node: yaml.Node, memo: dict, active: set) -> int:
    key = id(node)
    if key in memo:
        return memo[key]
    if key in active:
        raise YAMLTooLarge("recursive aliases (an anchor used inside itself) are not supported")
    active.add(key)
    if isinstance(node, yaml.SequenceNode):
        size = 1 + sum(_expanded_size(child, memo, active) for child in node.value)
    elif isinstance(node, yaml.MappingNode):
        size = 1 + sum(_expanded_size(k, memo, active) + _expanded_size(v, memo, active) for k, v in node.value)
    else:
        size = 1
    active.discard(key)
    memo[key] = size
    if size > MAX_NODES:
        raise YAMLTooLarge(f"the document expands to more than {MAX_NODES:,} values (aliases); refusing to load it")
    return size


def _checked(node: yaml.Node) -> yaml.Node:
    try:
        _expanded_size(node, {}, set())
    except RecursionError:
        raise YAMLTooLarge("the document is nested too deeply") from None
    return node


class _Loader(yaml.SafeLoader):
    pass


def load(text: str) -> Any:
    """Like yaml.safe_load, refusing documents that expand beyond MAX_NODES values."""
    loader = _Loader(text)
    try:
        node = loader.get_single_node()
        return None if node is None else loader.construct_document(_checked(node))
    except RecursionError:
        raise YAMLTooLarge("the document is nested too deeply") from None
    finally:
        loader.dispose()


def load_all(text: str) -> Iterator[Any]:
    """Like yaml.safe_load_all, with the same limits per document."""
    loader = _Loader(text)
    try:
        while loader.check_node():
            yield loader.construct_document(_checked(loader.get_node()))
    except RecursionError:
        raise YAMLTooLarge("the document is nested too deeply") from None
    finally:
        loader.dispose()
