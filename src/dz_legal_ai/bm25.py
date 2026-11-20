"""Small, dependency-free BM25 ranker for article-level legal chunks.

BM25 is computed over the candidate chunks of a single query. It replaces the
raw coverage count as the lexical component of the hybrid score, so a term
that appears in many articles contributes less than a discriminating one.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Iterable, Sequence


class BM25Index:
    """Okapi BM25 over an in-memory list of tokenized documents."""

    def __init__(
        self,
        documents: Sequence[Sequence[str]],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.k1 = k1
        self.b = b
        self.size = len(documents)
        self.term_frequencies = [Counter(document) for document in documents]
        self.lengths = [len(document) for document in documents]
        self.average_length = (
            sum(self.lengths) / self.size if self.size else 0.0
        )
        document_frequency: Counter[str] = Counter()
        for frequencies in self.term_frequencies:
            document_frequency.update(frequencies.keys())
        self.idf = {
            term: math.log(1.0 + (self.size - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequency.items()
        }

    def score(self, query_terms: Iterable[str], document_index: int) -> float:
        """Return the BM25 score of one document for a query."""
        if document_index < 0 or document_index >= self.size:
            return 0.0
        frequencies = self.term_frequencies[document_index]
        length = self.lengths[document_index]
        if not length:
            return 0.0
        normalization = self.k1 * (
            1.0 - self.b + self.b * length / self.average_length
        ) if self.average_length else self.k1
        total = 0.0
        for term in set(query_terms):
            frequency = frequencies.get(term, 0)
            if not frequency:
                continue
            inverse = self.idf.get(term, 0.0)
            total += inverse * (frequency * (self.k1 + 1.0)) / (
                frequency + normalization
            )
        return total
