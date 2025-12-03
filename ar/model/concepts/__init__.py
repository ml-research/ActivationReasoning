from __future__ import annotations

from typing import List, Union

from ar.config import LogicConfig

from .topk import TopKConcepts
from .tree import TreeConcepts

ConceptInstance = Union[TopKConcepts, TreeConcepts]


class Concepts:
    """
    Compatibility wrapper that dispatches to the concrete concept implementation
    based on the configured search strategy.
    """

    def __new__(
        cls,
        concepts: List[str],
        cache_dir: str,
        config: LogicConfig,
        verbose: bool = True,
    ) -> ConceptInstance:
        if config.concept_dictionary == "top_k":
            return TopKConcepts(concepts, cache_dir, config, verbose)
        if config.concept_dictionary == "tree":
            return TreeConcepts(concepts, cache_dir, config, verbose)
        raise ValueError(
            f"Unsupported concept_dictionary '{config.concept_dictionary}'. "
            "Expected 'top_k' or 'tree'."
        )


__all__ = ["Concepts", "TopKConcepts", "TreeConcepts"]
