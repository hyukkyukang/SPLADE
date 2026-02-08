"""Scoring functions for index-based search."""

import numba
import numpy as np


def _estimate_posting_length(term_ptr: np.ndarray, q_indices: np.ndarray) -> int:
    """Estimate total postings length for the query terms."""
    if q_indices.size == 0:
        return 0
    term_next: np.ndarray = term_ptr[q_indices + 1]
    term_curr: np.ndarray = term_ptr[q_indices]
    total: int = int(np.sum(term_next - term_curr))
    return total


@numba.njit
def _accumulate_scores(
    term_ptr: np.ndarray,
    post_doc_ids: np.ndarray,
    post_weights: np.ndarray,
    q_indices: np.ndarray,
    q_values: np.ndarray,
    scores: np.ndarray,
    seen: np.ndarray,
    touched: np.ndarray,
) -> int:
    # Accumulate scores and track which docs were touched.
    touched_count: int = 0
    for idx in range(q_indices.shape[0]):
        term_id: int = int(q_indices[idx])
        q_weight: float = float(q_values[idx])
        start: int = int(term_ptr[term_id])
        end: int = int(term_ptr[term_id + 1])
        for pos in range(start, end):
            doc_id: int = int(post_doc_ids[pos])
            if seen[doc_id] == 0:
                seen[doc_id] = 1
                touched[touched_count] = doc_id
                touched_count += 1
            scores[doc_id] += q_weight * float(post_weights[pos])
    return touched_count


def score_query_postings(
    term_ptr: np.ndarray,
    post_doc_ids: np.ndarray,
    post_weights: np.ndarray,
    q_indices: np.ndarray,
    q_values: np.ndarray,
    *,
    scores: np.ndarray,
    seen: np.ndarray,
    top_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Score a query against an inverted index and return top-k results."""
    if top_k <= 0:
        empty_docs: np.ndarray = np.zeros((0,), dtype=np.int32)
        empty_scores: np.ndarray = np.zeros((0,), dtype=np.float32)
        return empty_docs, empty_scores
    if q_indices.size == 0:
        empty_docs = np.zeros((0,), dtype=np.int32)
        empty_scores = np.zeros((0,), dtype=np.float32)
        return empty_docs, empty_scores

    total_postings: int = _estimate_posting_length(term_ptr, q_indices)
    if total_postings <= 0:
        empty_docs = np.zeros((0,), dtype=np.int32)
        empty_scores = np.zeros((0,), dtype=np.float32)
        return empty_docs, empty_scores

    touched: np.ndarray = np.empty(int(total_postings), dtype=np.int32)
    touched_count: int = _accumulate_scores(
        term_ptr=term_ptr,
        post_doc_ids=post_doc_ids,
        post_weights=post_weights,
        q_indices=q_indices,
        q_values=q_values,
        scores=scores,
        seen=seen,
        touched=touched,
    )
    if touched_count <= 0:
        empty_docs = np.zeros((0,), dtype=np.int32)
        empty_scores = np.zeros((0,), dtype=np.float32)
        return empty_docs, empty_scores

    touched_docs: np.ndarray = touched[:touched_count]
    touched_scores: np.ndarray = scores[touched_docs]

    if touched_count <= top_k:
        order: np.ndarray = np.argsort(touched_scores)[::-1]
        top_docs: np.ndarray = touched_docs[order]
        top_scores: np.ndarray = touched_scores[order]
    else:
        top_k_int: int = int(top_k)
        top_positions: np.ndarray = np.argpartition(touched_scores, -top_k_int)[
            -top_k_int:
        ]
        top_docs = touched_docs[top_positions]
        top_scores = touched_scores[top_positions]
        order = np.argsort(top_scores)[::-1]
        top_docs = top_docs[order]
        top_scores = top_scores[order]

    # Reset buffers for the next query.
    scores[touched_docs] = 0.0
    seen[touched_docs] = 0
    return top_docs.astype(np.int32, copy=False), top_scores.astype(
        np.float32, copy=False
    )


@numba.njit
def _advance_to(
    post_doc_ids: np.ndarray, pos: int, end: int, target_doc: int
) -> int:
    left: int = pos
    right: int = end
    while left < right:
        mid: int = (left + right) // 2
        if post_doc_ids[mid] < target_doc:
            left = mid + 1
        else:
            right = mid
    return left


@numba.njit
def _min_score(values: np.ndarray, count: int) -> float:
    min_val: float = float(values[0])
    for idx in range(1, count):
        value = float(values[idx])
        if value < min_val:
            min_val = value
    return min_val


@numba.njit
def _find_min_index(values: np.ndarray, count: int) -> int:
    min_idx: int = 0
    min_val: float = float(values[0])
    for idx in range(1, count):
        value = float(values[idx])
        if value < min_val:
            min_val = value
            min_idx = idx
    return min_idx


@numba.njit
def _init_order(current_doc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order: np.ndarray = np.argsort(current_doc)
    order_pos: np.ndarray = np.empty(order.shape[0], dtype=np.int64)
    for idx in range(order.shape[0]):
        order_pos[order[idx]] = idx
    return order, order_pos


@numba.njit
def _sift_forward(
    order: np.ndarray, order_pos: np.ndarray, current_doc: np.ndarray, term_idx: int
) -> None:
    pos: int = int(order_pos[term_idx])
    n_terms: int = int(order.shape[0])
    while pos + 1 < n_terms:
        next_term: int = int(order[pos + 1])
        if current_doc[term_idx] <= current_doc[next_term]:
            break
        order[pos] = next_term
        order_pos[next_term] = pos
        pos += 1
    order[pos] = term_idx
    order_pos[term_idx] = pos


@numba.njit
def _refresh_current_doc(
    post_doc_ids: np.ndarray,
    pos: np.ndarray,
    end: np.ndarray,
    current_doc: np.ndarray,
    term_idx: int,
    sentinel: np.int32,
) -> None:
    if pos[term_idx] < end[term_idx]:
        current_doc[term_idx] = post_doc_ids[pos[term_idx]]
    else:
        current_doc[term_idx] = sentinel


@numba.njit
def _refresh_block_cache(
    term_start: np.ndarray,
    term_end: np.ndarray,
    term_block_base: np.ndarray,
    block_max: np.ndarray,
    post_doc_ids: np.ndarray,
    pos: np.ndarray,
    block_idx: np.ndarray,
    block_end_doc: np.ndarray,
    block_max_val: np.ndarray,
    term_idx: int,
    block_size: int,
    sentinel: np.int32,
) -> None:
    if pos[term_idx] >= term_end[term_idx]:
        block_end_doc[term_idx] = sentinel
        block_max_val[term_idx] = 0.0
        block_idx[term_idx] = 0
        return
    new_block_idx: int = int((pos[term_idx] - term_start[term_idx]) // block_size)
    if new_block_idx != block_idx[term_idx]:
        block_idx[term_idx] = new_block_idx
        block_id: int = int(term_block_base[term_idx] + new_block_idx)
        block_end_pos: int = int(term_start[term_idx]) + (new_block_idx + 1) * block_size
        if block_end_pos > term_end[term_idx]:
            block_end_pos = int(term_end[term_idx])
        block_end_doc[term_idx] = post_doc_ids[block_end_pos - 1]
        block_max_val[term_idx] = float(block_max[block_id])


@numba.njit
def _score_query_postings_wand(
    term_ptr: np.ndarray,
    post_doc_ids: np.ndarray,
    post_weights: np.ndarray,
    term_max: np.ndarray,
    q_indices: np.ndarray,
    q_values: np.ndarray,
    top_k: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    n_terms: int = q_indices.shape[0]
    pos: np.ndarray = np.empty(n_terms, dtype=np.int64)
    end: np.ndarray = np.empty(n_terms, dtype=np.int64)
    q_term_max: np.ndarray = np.empty(n_terms, dtype=np.float64)
    current_doc: np.ndarray = np.empty(n_terms, dtype=np.int32)
    top_docs: np.ndarray = np.empty(top_k, dtype=np.int32)
    top_scores: np.ndarray = np.empty(top_k, dtype=np.float32)
    top_count: int = 0
    threshold: float = -np.inf
    sentinel: np.int32 = np.int32(2147483647)
    pivot_terms: np.ndarray = np.empty(n_terms, dtype=np.int64)
    advance_terms: np.ndarray = np.empty(n_terms, dtype=np.int64)

    for idx in range(n_terms):
        term_id = int(q_indices[idx])
        pos[idx] = int(term_ptr[term_id])
        end[idx] = int(term_ptr[term_id + 1])
        if pos[idx] < end[idx]:
            current_doc[idx] = post_doc_ids[pos[idx]]
        else:
            current_doc[idx] = sentinel
        q_term_max[idx] = float(q_values[idx]) * float(term_max[term_id])

    order, order_pos = _init_order(current_doc)

    while True:
        if current_doc[order[0]] == sentinel:
            break

        ub_sum: float = 0.0
        pivot_doc: int = int(sentinel)
        pivot_found: bool = False
        for ord_idx in range(n_terms):
            term_idx = int(order[ord_idx])
            if current_doc[term_idx] == sentinel:
                break
            ub_sum += q_term_max[term_idx]
            if ub_sum > threshold:
                pivot_doc = int(current_doc[term_idx])
                pivot_found = True
                break

        if not pivot_found:
            break

        if pivot_doc == int(current_doc[order[0]]):
            score: float = 0.0
            pivot_count: int = 0
            for ord_idx in range(n_terms):
                term_idx = int(order[ord_idx])
                if current_doc[term_idx] != pivot_doc:
                    break
                score += float(q_values[term_idx]) * float(post_weights[pos[term_idx]])
                pivot_terms[pivot_count] = term_idx
                pivot_count += 1
            if score > threshold:
                if top_count < top_k:
                    top_docs[top_count] = np.int32(pivot_doc)
                    top_scores[top_count] = np.float32(score)
                    top_count += 1
                    if top_count == top_k:
                        threshold = _min_score(top_scores, top_k)
                else:
                    min_idx = _find_min_index(top_scores, top_k)
                    if score > float(top_scores[min_idx]):
                        top_scores[min_idx] = np.float32(score)
                        top_docs[min_idx] = np.int32(pivot_doc)
                        threshold = _min_score(top_scores, top_k)
            for idx in range(pivot_count):
                term_idx = int(pivot_terms[idx])
                pos[term_idx] += 1
                _refresh_current_doc(
                    post_doc_ids, pos, end, current_doc, term_idx, sentinel
                )
                _sift_forward(order, order_pos, current_doc, term_idx)
        else:
            advance_count: int = 0
            for ord_idx in range(n_terms):
                term_idx = int(order[ord_idx])
                if current_doc[term_idx] >= pivot_doc:
                    break
                advance_terms[advance_count] = term_idx
                advance_count += 1
            for idx in range(advance_count):
                term_idx = int(advance_terms[idx])
                pos[term_idx] = _advance_to(
                    post_doc_ids, int(pos[term_idx]), int(end[term_idx]), pivot_doc
                )
                _refresh_current_doc(
                    post_doc_ids, pos, end, current_doc, term_idx, sentinel
                )
                _sift_forward(order, order_pos, current_doc, term_idx)

    return top_docs, top_scores, top_count


def score_query_postings_wand(
    term_ptr: np.ndarray,
    post_doc_ids: np.ndarray,
    post_weights: np.ndarray,
    term_max: np.ndarray,
    q_indices: np.ndarray,
    q_values: np.ndarray,
    *,
    scores: np.ndarray,
    seen: np.ndarray,
    top_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Score a query using classic WAND and return top-k results."""
    if top_k <= 0:
        empty_docs: np.ndarray = np.zeros((0,), dtype=np.int32)
        empty_scores: np.ndarray = np.zeros((0,), dtype=np.float32)
        return empty_docs, empty_scores
    if q_indices.size == 0:
        empty_docs = np.zeros((0,), dtype=np.int32)
        empty_scores = np.zeros((0,), dtype=np.float32)
        return empty_docs, empty_scores
    if term_max is None:
        raise ValueError("WAND scoring requires term_max.")

    top_docs, top_scores, top_count = _score_query_postings_wand(
        term_ptr=term_ptr,
        post_doc_ids=post_doc_ids,
        post_weights=post_weights,
        term_max=term_max,
        q_indices=q_indices,
        q_values=q_values,
        top_k=int(top_k),
    )
    if top_count <= 0:
        empty_docs = np.zeros((0,), dtype=np.int32)
        empty_scores = np.zeros((0,), dtype=np.float32)
        return empty_docs, empty_scores

    selected_docs: np.ndarray = top_docs[:top_count].copy()
    selected_scores: np.ndarray = top_scores[:top_count].copy()
    order: np.ndarray = np.argsort(selected_scores)[::-1]
    return selected_docs[order].astype(np.int32, copy=False), selected_scores[
        order
    ].astype(np.float32, copy=False)


@numba.njit
def _score_query_postings_bmw(
    term_ptr: np.ndarray,
    post_doc_ids: np.ndarray,
    post_weights: np.ndarray,
    term_max: np.ndarray,
    block_max: np.ndarray,
    block_ptr: np.ndarray,
    q_indices: np.ndarray,
    q_values: np.ndarray,
    top_k: int,
    block_size: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    n_terms: int = q_indices.shape[0]
    pos: np.ndarray = np.empty(n_terms, dtype=np.int64)
    term_start: np.ndarray = np.empty(n_terms, dtype=np.int64)
    term_end: np.ndarray = np.empty(n_terms, dtype=np.int64)
    term_block_base: np.ndarray = np.empty(n_terms, dtype=np.int64)
    term_max_values: np.ndarray = np.empty(n_terms, dtype=np.float64)
    q_term_max: np.ndarray = np.empty(n_terms, dtype=np.float64)
    block_idx: np.ndarray = np.empty(n_terms, dtype=np.int64)
    block_end_doc: np.ndarray = np.empty(n_terms, dtype=np.int32)
    block_max_val: np.ndarray = np.empty(n_terms, dtype=np.float64)
    current_doc: np.ndarray = np.empty(n_terms, dtype=np.int32)
    top_docs: np.ndarray = np.empty(top_k, dtype=np.int32)
    top_scores: np.ndarray = np.empty(top_k, dtype=np.float32)
    top_count: int = 0
    threshold: float = -np.inf
    sentinel: np.int32 = np.int32(2147483647)
    pivot_terms: np.ndarray = np.empty(n_terms, dtype=np.int64)
    advance_terms: np.ndarray = np.empty(n_terms, dtype=np.int64)

    for idx in range(n_terms):
        term_id = int(q_indices[idx])
        start = int(term_ptr[term_id])
        end = int(term_ptr[term_id + 1])
        term_start[idx] = start
        term_end[idx] = end
        term_block_base[idx] = int(block_ptr[term_id])
        pos[idx] = start
        term_max_value: float = float(term_max[term_id])
        term_max_values[idx] = term_max_value
        q_term_max[idx] = float(q_values[idx]) * term_max_value
        if start < end:
            current_doc[idx] = post_doc_ids[start]
            block_idx[idx] = 0
            block_end_pos: int = start + block_size
            if block_end_pos > end:
                block_end_pos = end
            block_end_doc[idx] = post_doc_ids[block_end_pos - 1]
            block_max_val[idx] = float(block_max[term_block_base[idx]])
        else:
            current_doc[idx] = sentinel
            block_idx[idx] = 0
            block_end_doc[idx] = sentinel
            block_max_val[idx] = 0.0

    order, order_pos = _init_order(current_doc)

    while True:
        if current_doc[order[0]] == sentinel:
            break

        ub_sum: float = 0.0
        pivot_doc: int = int(sentinel)
        pivot_found: bool = False
        for ord_idx in range(n_terms):
            term_idx = int(order[ord_idx])
            if current_doc[term_idx] == sentinel:
                break
            ub_sum += q_term_max[term_idx]
            if ub_sum > threshold:
                pivot_doc = int(current_doc[term_idx])
                pivot_found = True
                break

        if not pivot_found:
            break

        if pivot_doc == int(current_doc[order[0]]):
            block_sum: float = 0.0
            pivot_count: int = 0
            for ord_idx in range(n_terms):
                term_idx = int(order[ord_idx])
                if current_doc[term_idx] != pivot_doc:
                    break
                if pivot_doc <= block_end_doc[term_idx]:
                    ub_value = block_max_val[term_idx]
                else:
                    ub_value = term_max_values[term_idx]
                block_sum += float(q_values[term_idx]) * ub_value
                pivot_terms[pivot_count] = term_idx
                pivot_count += 1

            if block_sum <= threshold:
                for idx in range(pivot_count):
                    term_idx = int(pivot_terms[idx])
                    pos[term_idx] += 1
                    _refresh_current_doc(
                        post_doc_ids, pos, term_end, current_doc, term_idx, sentinel
                    )
                    _refresh_block_cache(
                        term_start,
                        term_end,
                        term_block_base,
                        block_max,
                        post_doc_ids,
                        pos,
                        block_idx,
                        block_end_doc,
                        block_max_val,
                        term_idx,
                        block_size,
                        sentinel,
                    )
                    _sift_forward(order, order_pos, current_doc, term_idx)
                continue

            score: float = 0.0
            for idx in range(pivot_count):
                term_idx = int(pivot_terms[idx])
                score += float(q_values[term_idx]) * float(post_weights[pos[term_idx]])
            if score > threshold:
                if top_count < top_k:
                    top_docs[top_count] = np.int32(pivot_doc)
                    top_scores[top_count] = np.float32(score)
                    top_count += 1
                    if top_count == top_k:
                        threshold = _min_score(top_scores, top_k)
                else:
                    min_idx = _find_min_index(top_scores, top_k)
                    if score > float(top_scores[min_idx]):
                        top_scores[min_idx] = np.float32(score)
                        top_docs[min_idx] = np.int32(pivot_doc)
                        threshold = _min_score(top_scores, top_k)
            for idx in range(pivot_count):
                term_idx = int(pivot_terms[idx])
                pos[term_idx] += 1
                _refresh_current_doc(
                    post_doc_ids, pos, term_end, current_doc, term_idx, sentinel
                )
                _refresh_block_cache(
                    term_start,
                    term_end,
                    term_block_base,
                    block_max,
                    post_doc_ids,
                    pos,
                    block_idx,
                    block_end_doc,
                    block_max_val,
                    term_idx,
                    block_size,
                    sentinel,
                )
                _sift_forward(order, order_pos, current_doc, term_idx)
        else:
            advance_count: int = 0
            for ord_idx in range(n_terms):
                term_idx = int(order[ord_idx])
                if current_doc[term_idx] >= pivot_doc:
                    break
                advance_terms[advance_count] = term_idx
                advance_count += 1
            for idx in range(advance_count):
                term_idx = int(advance_terms[idx])
                pos[term_idx] = _advance_to(
                    post_doc_ids, int(pos[term_idx]), int(term_end[term_idx]), pivot_doc
                )
                _refresh_current_doc(
                    post_doc_ids, pos, term_end, current_doc, term_idx, sentinel
                )
                _refresh_block_cache(
                    term_start,
                    term_end,
                    term_block_base,
                    block_max,
                    post_doc_ids,
                    pos,
                    block_idx,
                    block_end_doc,
                    block_max_val,
                    term_idx,
                    block_size,
                    sentinel,
                )
                _sift_forward(order, order_pos, current_doc, term_idx)

    return top_docs, top_scores, top_count


def score_query_postings_bmw(
    term_ptr: np.ndarray,
    post_doc_ids: np.ndarray,
    post_weights: np.ndarray,
    term_max: np.ndarray,
    block_max: np.ndarray,
    block_ptr: np.ndarray,
    q_indices: np.ndarray,
    q_values: np.ndarray,
    *,
    scores: np.ndarray,
    seen: np.ndarray,
    top_k: int,
    block_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Score a query using Block-Max WAND and return top-k results."""
    if top_k <= 0:
        empty_docs: np.ndarray = np.zeros((0,), dtype=np.int32)
        empty_scores: np.ndarray = np.zeros((0,), dtype=np.float32)
        return empty_docs, empty_scores
    if q_indices.size == 0:
        empty_docs = np.zeros((0,), dtype=np.int32)
        empty_scores = np.zeros((0,), dtype=np.float32)
        return empty_docs, empty_scores
    if block_size <= 0:
        raise ValueError("block_size must be a positive integer.")
    if term_max is None or block_max is None or block_ptr is None:
        raise ValueError("BMW scoring requires term_max, block_max, and block_ptr.")

    top_docs, top_scores, top_count = _score_query_postings_bmw(
        term_ptr=term_ptr,
        post_doc_ids=post_doc_ids,
        post_weights=post_weights,
        term_max=term_max,
        block_max=block_max,
        block_ptr=block_ptr,
        q_indices=q_indices,
        q_values=q_values,
        top_k=int(top_k),
        block_size=int(block_size),
    )
    if top_count <= 0:
        empty_docs = np.zeros((0,), dtype=np.int32)
        empty_scores = np.zeros((0,), dtype=np.float32)
        return empty_docs, empty_scores

    selected_docs: np.ndarray = top_docs[:top_count].copy()
    selected_scores: np.ndarray = top_scores[:top_count].copy()
    order: np.ndarray = np.argsort(selected_scores)[::-1]
    return selected_docs[order].astype(np.int32, copy=False), selected_scores[
        order
    ].astype(np.float32, copy=False)


__all__ = [
    "score_query_postings",
    "score_query_postings_wand",
    "score_query_postings_bmw",
]
