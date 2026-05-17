# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Utilities for pruning visual tokens via GeoPrune (residual-L2 scoring).

GeoPrune scores every visual token by the L2 norm of its hidden state in the
residual space obtained after subtracting the top-K principal components of
the per-image feature matrix.  Intuitively, the dominant singular directions
capture the global "DC" component shared by every patch (background statistics,
register-token bias, viewpoint, ...) while the residual captures fine-grained,
token-specific information.  Tokens with a large residual norm are therefore
the most informative and are kept; the rest are discarded.

The implementation is intentionally lightweight (a few power iterations + a
top-k) so that it can run inside the ViT forward pass without measurable
overhead.

Reference:
    - Arora et al., "A Simple but Tough-to-Beat Baseline for Sentence
      Embeddings", ICLR 2017 (the SIF projection trick).
"""

from __future__ import annotations

import torch

__all__ = [
    "power_iteration",
    "compute_residual_l2_scores",
    "prune_visual_tokens_dominant_only",
]


# ---------------------------------------------------------------------------
# Truncated SVD via power iteration
# ---------------------------------------------------------------------------


def power_iteration(
    matrix: torch.Tensor,
    num_iters: int = 10,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Estimate the top singular triple (sigma, v, u) of ``matrix``.

    Args:
        matrix: ``(N, D)`` tensor.
        num_iters: Number of power-iteration sweeps. Five iterations are
            usually enough for ViT hidden states.

    Returns:
        ``(sigma, v, u)`` where ``v`` is the right singular vector of shape
        ``(D,)``, ``u`` is the corresponding left singular vector ``(N,)`` and
        ``sigma`` is a scalar tensor.
    """
    _, dim = matrix.shape
    v = torch.randn(dim, device=matrix.device, dtype=matrix.dtype)
    v = v / v.norm().clamp(min=1e-8)
    sigma = matrix.new_zeros(())
    u = matrix.new_zeros(matrix.shape[0])
    for _ in range(num_iters):
        u = matrix @ v
        u = u / u.norm().clamp(min=1e-8)
        v = matrix.T @ u
        sigma = v.norm()
        v = v / sigma.clamp(min=1e-8)
    return sigma, v, u


def compute_residual_l2_scores(
    features: torch.Tensor,
    num_singular_values: int = 1,
    num_power_iters: int = 10,
) -> torch.Tensor:
    """Compute GeoPrune token importance scores.

    The scoring procedure is:

    1. Estimate the top ``num_singular_values`` singular triples of the
       feature matrix using power iteration with deflation.
    2. Project them out to obtain the residual representation
       ``r_i = h_i - sum_j (h_i . v_j) v_j``.
    3. The score of token ``i`` is ``||r_i||_2``.

    A score of ``num_singular_values <= 0`` short-circuits to the raw L2 norm
    of ``features`` (equivalent to vanilla L2-norm pruning).

    Args:
        features: ``(N, D)`` tensor of per-token features.
        num_singular_values: Number of leading singular components to remove.
        num_power_iters: Number of power-iteration sweeps per component.

    Returns:
        ``(N,)`` tensor of non-negative importance scores. Always returned in
        ``float32`` for numerical stability of the downstream ``topk``.
    """
    if features.ndim != 2:
        raise ValueError(
            f"compute_residual_l2_scores expects a 2D tensor, got shape "
            f"{tuple(features.shape)}"
        )

    work = features.detach().float()
    if num_singular_values <= 0:
        return work.norm(dim=-1)

    residual = work.clone()
    for _ in range(num_singular_values):
        sigma, v, u = power_iteration(residual, num_iters=num_power_iters)
        # Deflation: subtract the rank-1 contribution.
        residual = residual - sigma * torch.outer(u, v)

    return residual.norm(dim=-1)


# ---------------------------------------------------------------------------
# Token selection given precomputed scores
# ---------------------------------------------------------------------------


def prune_visual_tokens_dominant_only(
    embeddings: torch.Tensor,
    scores: torch.Tensor,
    pruning_rate: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep the top-k visual tokens ranked by GeoPrune importance scores.

    Args:
        embeddings: ``(N, hidden_size)`` post-merger embeddings for one image.
        scores: ``(N,)`` importance scores per token. Must be aligned with the
            embeddings (one score per merged token).
        pruning_rate: Fraction of tokens to drop, in ``[0, 1)``. ``0.0`` keeps
            every token, ``0.6`` keeps 40% of tokens.

    Returns:
        ``(pruned_embeddings, keep_indices)`` where ``keep_indices`` is sorted
        ascending so downstream consumers (mRoPE recomputation, deepstack
        features, ...) can index into the original layout deterministically.
    """
    num_tokens = embeddings.shape[0]
    if pruning_rate <= 0.0 or num_tokens <= 1:
        return embeddings, torch.arange(num_tokens, device=embeddings.device)

    if scores.shape[0] != num_tokens:
        raise ValueError(
            f"scores has {scores.shape[0]} elements but embeddings has "
            f"{num_tokens} tokens"
        )

    keep_num = max(1, int(num_tokens * (1.0 - pruning_rate)))
    if keep_num >= num_tokens:
        return embeddings, torch.arange(num_tokens, device=embeddings.device)

    _, topk_indices = torch.topk(scores, keep_num, sorted=False)
    keep_indices = topk_indices.sort().values
    return embeddings[keep_indices], keep_indices
