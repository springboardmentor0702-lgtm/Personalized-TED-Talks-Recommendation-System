"""Information Retrieval evaluation metrics for top-K recommendation lists.

Implements:
- Precision@K
- Recall@K
- NDCG@K (Normalized Discounted Cumulative Gain)
- AP@K (Average Precision at K; averaged across queries to obtain MAP@K)
"""
from typing import Iterable, Optional, Sequence, Set, Tuple, TypeVar
import math

T = TypeVar("T")


def sanitize_recommendations(
    recommended: Sequence[T],
    k: int,
    query_item: Optional[T] = None,
) -> Tuple[T, ...]:
    """Deduplicate recommendations in order, exclude query_item, and truncate to k."""
    if k <= 0 or not recommended:
        return ()
    seen: Set[T] = set()
    cleaned = []
    for item in recommended:
        if query_item is not None and item == query_item:
            continue
        if item not in seen:
            seen.add(item)
            cleaned.append(item)
            if len(cleaned) == k:
                break
    return tuple(cleaned)


def precision_at_k(
    recommended: Sequence[T],
    relevant: Iterable[T],
    k: int,
    query_item: Optional[T] = None,
) -> float:
    """Compute Precision@K = (# relevant items in top-K) / K.

    Returns 0.0 when k <= 0 or recommended list is empty after sanitization.
    """
    if k <= 0:
        return 0.0
    rel_set = set(relevant)
    recs = sanitize_recommendations(recommended, k, query_item=query_item)
    if not recs or not rel_set:
        return 0.0
    hits = sum(1 for item in recs if item in rel_set)
    return float(hits / k)


def recall_at_k(
    recommended: Sequence[T],
    relevant: Iterable[T],
    k: int,
    query_item: Optional[T] = None,
) -> float:
    """Compute Recall@K = (# relevant items in top-K) / |relevant|.

    Returns 0.0 when k <= 0 or relevant set is empty (prevents division by zero).
    """
    if k <= 0:
        return 0.0
    rel_set = set(relevant)
    if not rel_set:
        return 0.0
    recs = sanitize_recommendations(recommended, k, query_item=query_item)
    if not recs:
        return 0.0
    hits = sum(1 for item in recs if item in rel_set)
    return float(hits / len(rel_set))


def ndcg_at_k(
    recommended: Sequence[T],
    relevant: Iterable[T],
    k: int,
    query_item: Optional[T] = None,
) -> float:
    """Compute NDCG@K using binary relevance and log2(rank + 1) discounting.

    DCG@K  = sum_{j=1}^{m} rel_j / log2(j + 1)
    IDCG@K = sum_{j=1}^{min(K, |relevant|)} 1 / log2(j + 1)
    NDCG@K = DCG@K / IDCG@K (0.0 if IDCG@K == 0).
    """
    if k <= 0:
        return 0.0
    rel_set = set(relevant)
    if not rel_set:
        return 0.0
    recs = sanitize_recommendations(recommended, k, query_item=query_item)
    if not recs:
        return 0.0

    dcg = sum(1.0 / math.log2(rank + 1) for rank, item in enumerate(recs, start=1) if item in rel_set)
    ideal_hits = min(k, len(rel_set))
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    if idcg == 0.0:
        return 0.0
    return float(dcg / idcg)


def average_precision_at_k(
    recommended: Sequence[T],
    relevant: Iterable[T],
    k: int,
    query_item: Optional[T] = None,
) -> float:
    """Compute Average Precision at K (AP@K) for a single query.

    AP@K = (1 / min(K, |relevant|)) * sum_{j=1}^{m} Precision@j * rel_j
    Averaging AP@K across all eligible queries gives MAP@K.
    Returns 0.0 when k <= 0 or relevant set is empty.
    """
    if k <= 0:
        return 0.0
    rel_set = set(relevant)
    if not rel_set:
        return 0.0
    recs = sanitize_recommendations(recommended, k, query_item=query_item)
    if not recs:
        return 0.0

    hits = 0
    precision_sum = 0.0
    for rank, item in enumerate(recs, start=1):
        if item in rel_set:
            hits += 1
            precision_sum += hits / rank
    denom = min(k, len(rel_set))
    return float(precision_sum / denom) if denom > 0 else 0.0


def evaluate_single_query(
    recommended: Sequence[T],
    relevant: Iterable[T],
    k: int,
    query_item: Optional[T] = None,
) -> dict:
    """Return Precision@K, Recall@K, NDCG@K, and AP@K for one query."""
    return {
        f"precision@{k}": precision_at_k(recommended, relevant, k, query_item=query_item),
        f"recall@{k}": recall_at_k(recommended, relevant, k, query_item=query_item),
        f"ndcg@{k}": ndcg_at_k(recommended, relevant, k, query_item=query_item),
        f"ap@{k}": average_precision_at_k(recommended, relevant, k, query_item=query_item),
    }

