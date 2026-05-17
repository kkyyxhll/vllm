# Vision Token Pruning (GeoPrune)

```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-VL-32B-Instruct \
  --image-pruning-rate 0.3
```

This configuration drops the least informative ~30% of image tokens after the
ViT encoder but before they enter the LLM decoder, using GeoPrune scoring.

## Supported models

GeoPrune image token pruning is currently implemented for:

- `Qwen2_5_VLForConditionalGeneration`
- `Qwen3VLMultiModalForConditionalGeneration`
- `Qwen3_5ForConditionalGeneration` and `Qwen3_5MoeForConditionalGeneration`

For other multimodal models the engine ignores `--image-pruning-rate` and falls
back to the default behaviour.

## How it works (high level)

1. The ViT encoder runs as usual to produce per-patch hidden states.
2. At a configurable layer (`vit_attention_score_layer_index`) the model
   snapshots the hidden states *before* the chosen block. This snapshot is
   independent of the attention kernel - GeoPrune does not modify the
   attention computation in any way.
3. The snapshot is reduced from `(seq_len, hidden_size)` to
   `(num_merged_tokens, hidden_size)` by averaging every
   `spatial_merge_unit` adjacent patches, matching the merger output layout.
4. A truncated SVD (power iteration with deflation) removes the top
   `num_singular_values` principal components, which capture global "DC"
   directions shared by every patch. The L2 norm of each residual row is the
   per-token GeoPrune importance score.
5. The lowest-scoring tokens are pruned according to `image_pruning_rate`.
6. mRoPE positions and multimodal metadata are updated to match the pruned
   sequence length, including data-parallel sharded execution.

The pruning happens entirely inside the **vision stack**; the LLM sees a
shorter sequence of image tokens with unchanged text tokens.

### Why "GeoPrune"?

The leading singular components of the per-image feature matrix can be
interpreted as the geometric "centre of mass" of the visual token cloud.
Subtracting them isolates the *direction* in which each token differs from
the bulk. Tokens with large residual norm are therefore the ones that carry
the most token-specific (i.e. discriminative) information, which is exactly
what should be kept for downstream reasoning. The approach is closely
related to the SIF projection trick of Arora et al. (ICLR 2017) and to the
classical "remove the DC term" preprocessing step in spectral analysis.

## Recommended configurations

- **Multi-image tasks** (e.g. 30 images / sample, multi-image QA):
    - `--image-pruning-rate 0.3`-`0.4`
    Preserves or slightly improves integrated accuracy while significantly
    reducing prefill latency.
- **Non-OCR single-image tasks** (perception, reasoning, general VQA):
    - `--image-pruning-rate 0.3`-`0.5`
    Typically <2% absolute accuracy drop across CC-Bench, MMBench, AI2D,
    MMMU-dev.
- **OCR-heavy tasks** (OCRBench, DocVQA):
    - `--image-pruning-rate <= 0.3`
    Higher pruning rates can hurt fine-grained text recognition.

### Performance

GeoPrune adds only the cost of one truncated SVD (≈ a few power iterations on
a `(num_merged_tokens, hidden_size)` matrix) per image. In practice the
overhead is negligible compared with the cost of the ViT blocks it replaces,
and the prefill savings scale linearly with the pruning rate.

- Decode latency is effectively unchanged; savings come from a cheaper
  prefill.
- The ViT encoder itself is not modified, so GeoPrune is compatible with all
  attention backends supported by vLLM (Flash Attention, FlashInfer, XFormers
  fallback, ...).

## Limitations and notes

- This feature currently targets **image tokens** only. Video token pruning
  (e.g. EVS) is handled by a separate path; in Qwen3-VL these mechanisms can
  coexist.
- Very aggressive pruning rates (e.g. ≥0.5 on hard multi-image tasks or OCR)
  can noticeably degrade structured metrics (edit distance, alignment).
