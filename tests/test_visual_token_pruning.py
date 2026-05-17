# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for GeoPrune scoring and pruning utilities plus config wiring."""

import pytest
import torch

from vllm.model_executor.layers.attention.visual_token_pruning import (
    compute_residual_l2_scores,
    power_iteration,
    prune_visual_tokens_dominant_only,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_embeddings():
    """100 tokens with hidden_size=64 and ascending scores."""
    torch.manual_seed(42)
    num_tokens, dim = 100, 64
    embeddings = torch.randn(num_tokens, dim)
    scores = torch.arange(num_tokens, dtype=torch.float32)
    return embeddings, scores


@pytest.fixture
def small_embeddings():
    """10 tokens with hidden_size=16."""
    torch.manual_seed(0)
    num_tokens, dim = 10, 16
    embeddings = torch.randn(num_tokens, dim)
    scores = torch.arange(num_tokens, dtype=torch.float32)
    return embeddings, scores


# ===================================================================
# Tests for prune_visual_tokens_dominant_only
# ===================================================================


class TestDominantOnly:
    def test_no_pruning_rate_0(self, sample_embeddings):
        emb, scores = sample_embeddings
        pruned, indices = prune_visual_tokens_dominant_only(emb, scores, 0.0)
        assert pruned.shape == emb.shape
        assert torch.equal(pruned, emb)
        assert indices.shape[0] == emb.shape[0]

    def test_prune_half(self, sample_embeddings):
        emb, scores = sample_embeddings
        pruned, indices = prune_visual_tokens_dominant_only(emb, scores, 0.5)
        assert pruned.shape[0] == 50
        assert pruned.shape[1] == emb.shape[1]
        assert indices.shape[0] == 50

    def test_keeps_highest_scoring_tokens(self, sample_embeddings):
        emb, scores = sample_embeddings
        # pruning_rate=0.6 keeps the top-40 scoring tokens (indices 60..99).
        _, indices = prune_visual_tokens_dominant_only(emb, scores, 0.6)
        assert indices.min().item() >= 60

    def test_indices_sorted(self, sample_embeddings):
        emb, scores = sample_embeddings
        _, indices = prune_visual_tokens_dominant_only(emb, scores, 0.7)
        assert torch.all(indices[1:] > indices[:-1])

    def test_pruned_embeds_match_indices(self, sample_embeddings):
        emb, scores = sample_embeddings
        pruned, indices = prune_visual_tokens_dominant_only(emb, scores, 0.5)
        assert torch.allclose(pruned, emb[indices])

    def test_at_least_one_token_kept(self, sample_embeddings):
        emb, scores = sample_embeddings
        pruned, _ = prune_visual_tokens_dominant_only(emb, scores, 0.999)
        assert pruned.shape[0] >= 1

    def test_very_small_input(self):
        emb = torch.randn(3, 8)
        scores = torch.tensor([1.0, 3.0, 2.0])
        pruned, indices = prune_visual_tokens_dominant_only(emb, scores, 0.5)
        # int(3 * (1 - 0.5)) = 1, keeps the highest-scoring token (index 1).
        assert pruned.shape[0] == 1
        assert indices.item() == 1

    def test_single_token_short_circuit(self):
        emb = torch.randn(1, 8)
        scores = torch.tensor([42.0])
        pruned, indices = prune_visual_tokens_dominant_only(emb, scores, 0.5)
        assert pruned.shape[0] == 1
        assert torch.equal(indices, torch.tensor([0]))

    def test_mismatched_scores_raise(self, sample_embeddings):
        emb, _ = sample_embeddings
        wrong_scores = torch.zeros(emb.shape[0] + 1)
        with pytest.raises(ValueError):
            prune_visual_tokens_dominant_only(emb, wrong_scores, 0.5)


# ===================================================================
# Tests for power_iteration / compute_residual_l2_scores
# ===================================================================


class TestPowerIteration:
    def test_recovers_top_singular_triple(self):
        torch.manual_seed(0)
        # Build a deterministic rank-1 matrix with known sigma.
        u_true = torch.randn(32)
        v_true = torch.randn(8)
        u_true = u_true / u_true.norm()
        v_true = v_true / v_true.norm()
        sigma_true = 5.0
        matrix = sigma_true * torch.outer(u_true, v_true)

        sigma, v, u = power_iteration(matrix, num_iters=50)

        assert sigma.item() == pytest.approx(sigma_true, rel=1e-4)
        # Eigenvectors may flip sign; compare absolute cosine similarity.
        assert abs(torch.dot(v, v_true).item()) == pytest.approx(1.0, abs=1e-4)
        assert abs(torch.dot(u, u_true).item()) == pytest.approx(1.0, abs=1e-4)


class TestResidualL2Scores:
    def test_shape_and_dtype(self):
        torch.manual_seed(1)
        feats = torch.randn(20, 16)
        scores = compute_residual_l2_scores(feats, num_singular_values=2)
        assert scores.shape == (20,)
        assert scores.dtype == torch.float32
        assert (scores >= 0).all()

    def test_zero_singular_values_equals_raw_norm(self):
        torch.manual_seed(2)
        feats = torch.randn(15, 8)
        scores = compute_residual_l2_scores(feats, num_singular_values=0)
        assert torch.allclose(scores, feats.float().norm(dim=-1), atol=1e-5)

    def test_rank_one_residual_is_zero(self):
        torch.manual_seed(3)
        # A rank-1 matrix is fully captured by the first singular component;
        # deflation should drive every residual norm essentially to zero.
        u = torch.randn(32)
        v = torch.randn(16)
        feats = torch.outer(u, v)
        scores = compute_residual_l2_scores(
            feats, num_singular_values=1, num_power_iters=50
        )
        assert scores.max().item() < 1e-3

    def test_ranks_match_known_signal(self):
        """Tokens with stronger residual signal must score higher."""
        torch.manual_seed(4)
        # Construct features as `c * v_common + alpha_i * w_i`, where
        # `v_common` is a shared DC direction and `alpha_i` is the
        # per-token residual energy.
        dim = 32
        num_tokens = 16
        v_common = torch.randn(dim)
        w = torch.randn(num_tokens, dim)
        w = w - (w @ v_common.unsqueeze(-1) / v_common.dot(v_common)) * v_common
        alpha = torch.linspace(0.1, 1.6, num_tokens)
        feats = 10.0 * v_common.unsqueeze(0).expand(num_tokens, dim) + alpha.unsqueeze(
            -1
        ) * w

        scores = compute_residual_l2_scores(feats, num_singular_values=1)
        ranking = scores.argsort()
        expected = alpha.argsort()
        # The induced ordering should match the residual energies exactly.
        assert torch.equal(ranking, expected)

    def test_rejects_non_2d(self):
        with pytest.raises(ValueError):
            compute_residual_l2_scores(torch.randn(8))


# ===================================================================
# End-to-end: scoring + topk on the same features
# ===================================================================


def test_geoprune_end_to_end():
    torch.manual_seed(5)
    feats = torch.randn(64, 96)
    scores = compute_residual_l2_scores(feats, num_singular_values=1)
    pruned, indices = prune_visual_tokens_dominant_only(feats, scores, pruning_rate=0.6)
    assert pruned.shape[0] == int(64 * 0.4)
    assert torch.equal(indices, indices.sort().values)
    # The selected tokens should be the ones with the largest residual scores.
    expected = scores.topk(int(64 * 0.4)).indices.sort().values
    assert torch.equal(indices, expected)


# ===================================================================
# Tests for MultiModalConfig auto-enable
# ===================================================================


class TestMultiModalConfigAutoEnable:
    def test_auto_enable_extract_score(self):
        from vllm.config.multimodal import MultiModalConfig

        cfg = MultiModalConfig(image_pruning_rate=0.6)
        assert cfg.extract_vit_attention_score is True

    def test_no_auto_enable_when_rate_none(self):
        from vllm.config.multimodal import MultiModalConfig

        cfg = MultiModalConfig(image_pruning_rate=None)
        assert cfg.extract_vit_attention_score is False

    def test_no_auto_enable_when_rate_0(self):
        from vllm.config.multimodal import MultiModalConfig

        cfg = MultiModalConfig(image_pruning_rate=0.0)
        assert cfg.extract_vit_attention_score is False

    def test_already_enabled_not_overridden(self):
        from vllm.config.multimodal import MultiModalConfig

        cfg = MultiModalConfig(image_pruning_rate=0.6, extract_vit_attention_score=True)
        assert cfg.extract_vit_attention_score is True

    def test_is_multimodal_pruning_enabled_image(self):
        from vllm.config.multimodal import MultiModalConfig

        cfg = MultiModalConfig(image_pruning_rate=0.6)
        assert cfg.is_multimodal_pruning_enabled() is True

    def test_is_multimodal_pruning_enabled_none(self):
        from vllm.config.multimodal import MultiModalConfig

        cfg = MultiModalConfig()
        assert cfg.is_multimodal_pruning_enabled() is False

    def test_is_multimodal_pruning_enabled_video(self):
        from vllm.config.multimodal import MultiModalConfig

        cfg = MultiModalConfig(video_pruning_rate=0.5)
        assert cfg.is_multimodal_pruning_enabled() is True


# ===================================================================
# GPU-based tests (only run when CUDA available)
# ===================================================================


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
class TestPruningOnGPU:
    def test_dominant_only_cuda(self):
        emb = torch.randn(100, 64, device="cuda")
        scores = torch.randn(100, device="cuda")
        pruned, _ = prune_visual_tokens_dominant_only(emb, scores, 0.5)
        assert pruned.device.type == "cuda"
        assert pruned.shape[0] == 50

    def test_residual_scores_cuda(self):
        feats = torch.randn(64, 32, device="cuda")
        scores = compute_residual_l2_scores(feats, num_singular_values=1)
        assert scores.device.type == "cuda"
        assert scores.shape == (64,)

    def test_bfloat16_features(self):
        feats = torch.randn(64, 32, device="cuda", dtype=torch.bfloat16)
        scores = compute_residual_l2_scores(feats, num_singular_values=1)
        # Scores are always returned in float32 for stable ranking.
        assert scores.dtype == torch.float32
        assert scores.shape == (64,)
