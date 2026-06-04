# Lighthouse Attention
 
Official implementation of **Lighthouse Attention**, from the paper:
 
> **Long Context Pre-Training with Lighthouse Attention**
> Bowen Peng, Subho Ghosh, Jeffrey Quesnelle — Nous Research, May 2026
> [arXiv:2605.06554](https://arxiv.org/abs/2605.06554)
 
---
 
## What is Lighthouse Attention?
 
Training causal transformers on very long sequences is bottlenecked by the quadratic time and memory cost of standard scaled dot-product attention (SDPA). Existing sparse attention methods solve this at inference but introduce a harder problem: they cannot be evaluated against their dense backbone during training, so you never know if the model actually learned to use the sparse structure.
 
Lighthouse Attention solves this differently. It is a **training-only, symmetrical, selection-based hierarchical attention** that:
 
- Wraps around ordinary SDPA — no new kernel required
- Is gradient-free by construction (no straight-through estimator)
- Is removed after training, recovering a full dense attention model
- Achieves sub-quadratic pre- and post-processing while preserving left-to-right causality
The core idea is a **symmetric pyramid** that pools queries, keys, and values simultaneously across a multi-level hierarchy, selects the top-K entries using a fused chunked-bitonic kernel, and attends to them with standard FlashAttention. Because the selection is symmetric and pooling is bidirectional, the model retains full-resolution representations at every level while the expensive attention computation operates on a compressed subsequence.
 
---
 
## How it differs from other sparse attention methods
 
| Property | MoBA | NSA (DeepSeek) | Lighthouse (ours) |
|---|---|---|---|
| Selection granularity | Block-level | Token-level | Multi-level pyramid |
| Symmetry | No | No | Yes (Q, K, V pooled symmetrically) |
| Training-time only | No | No | Yes |
| Gradient-free selection | No | No | Yes |
| Reuses FlashAttention kernel | No | No | Yes |
| Recovers dense model at inference | No | No | Yes |
 
**MoBA** selects contiguous blocks, which fits long-context inference but creates architectural entanglement — the sparse kernel lives inside the attention computation so optimized dense kernels cannot be reused.
 
**NSA (Native Sparse Attention)** scores every past token via a learned indexer and routes the top-K into a sparse operator. This is powerful but the selection is non-differentiable and the sparse kernel is architecture-specific.
 
**Lighthouse** sidesteps both problems. Selection lives outside the attention path, so standard FlashAttention handles the actual computation. The symmetric pyramid compresses the sequence before attention, not inside it. The result is a training-time efficiency method that leaves no architectural trace at inference — the final model is an ordinary dense transformer.
 
---
 
## The Symmetric Pyramid
 
The central mechanism is a two-stage compression and decompression of the input sequence.
 
**Pre-processing (compression):**
The sequence is pooled into a multi-level pyramid. At each level, fixed-size chunks are pooled symmetrically — queries, keys, and values are all pooled together, preserving left-to-right causality. A parameter-free fused chunked-bitonic scorer ranks every pyramid entry bidirectionally. The top-K entries are selected.
 
**Attention:**
The selected K entries form a dense, causally consistent subsequence. Standard FlashAttention runs on this compressed input. Because the entries are pooled symmetrically, outputs back-scatter deterministically — no learned routing, no auxiliary losses.
 
**Post-processing (decompression):**
Outputs are scattered back to full resolution. The back-scatter is deterministic given the pyramid structure, so no additional parameters or losses are needed.
 
The expensive step — the O(N²) attention computation — operates on O(N log N / K) tokens at level L = log_p(N/K), achieving sub-quadratic complexity while preserving the full-resolution residual stream.
 
---
 
## Key Properties
 
**Training correctness.** The central empirical question for any training-time sparse method is: does the resulting model match a dense baseline? Lighthouse models trained to convergence match or beat dense SDPA baselines trained on the same token budget, measured after a short dense recovery phase.
 
**No inference overhead.** All Lighthouse components are training-only. At inference the model is an ordinary dense transformer checkpoint. No sparse kernels, no routing logic, no pyramid — just the fine-tuned weights.
 
**Plug-and-play.** Lighthouse patches into the attention forward pass via a two-line modification. No architectural changes to the surrounding transformer are required.
 
---
 
## Installation
 
Lighthouse Attention is implemented as a patch on [torchtitan](https://github.com/pytorch/torchtitan).
 
```bash
git clone https://github.com/ighoshsubho/lighthouse-attention.git
cd lighthouse-attention
 
# Apply the patch to your torchtitan installation
git apply lighthouse_attention.patch
```
 
### Requirements
 
```
torch >= 2.3
torchtitan
flash-attn >= 2.0
```
 
---
 
## Usage
 
After applying the patch, Lighthouse Attention is controlled via config flags:
 
```bash
# Key hyperparameters
--lighthouse.top_k          # Number of tokens selected per pyramid level (default: 64)
--lighthouse.pool_size      # Pooling chunk size (default: 4)
--lighthouse.levels         # Number of pyramid levels (default: 3)
--lighthouse.scorer_type    # Scoring function: 'bitonic' or 'learned' (default: 'bitonic')
--lighthouse.ctx_parallel   # Enable context parallelism (default: False)
```
 
### Example: Training with Lighthouse Attention
 
```bash
# Train with Lighthouse Attention (top-K=64, pool_size=4, 3 pyramid levels)
torchrun --nproc_per_node=8 train.py \
  --model.name llama3_8b \
  --training.seq_len 32768 \
  --lighthouse.top_k 64 \
  --lighthouse.pool_size 4 \
  --lighthouse.levels 3
 
# Recovery phase: disable Lighthouse, fine-tune with full dense attention
torchrun --nproc_per_node=8 train.py \
  --model.name llama3_8b \
  --training.seq_len 32768 \
  --lighthouse.enabled False \
  --training.load_checkpoint path/to/lighthouse_checkpoint
```
 
---
 
## Two-Stage Training
 
Lighthouse uses a two-stage approach:
 
1. **Pre-training with Lighthouse** (~90% of budget): Sub-quadratic attention via the symmetric pyramid. Fast, memory-efficient, scales to long contexts.
2. **Dense recovery** (~10% of budget): Standard SDPA fine-tuning on top of the Lighthouse checkpoint. Recovers full dense attention quality.
The recovery phase is short because Lighthouse preserves the full-resolution residual stream throughout pre-training — the model has already learned long-range dependencies, it just needs to adapt to the dense attention pattern.
 
---
 
## Source Files
 
```
src/
  lighthouse_selection.py         # Pyramid construction, top-K selection, scatter/gather
  lighthouse_selection_cuda.py    # CUDA-accelerated chunked-bitonic scorer
configs/
  vary_top_k.yaml                 # Sweep top-K hyperparameter
  vary_pool_size.yaml             # Sweep pooling chunk size
  vary_levels.yaml                # Sweep pyramid depth
  vary_scorer_type.yaml           # Compare bitonic vs learned scorer
  vary_ctx_parallel.yaml          # Context parallelism ablation
```
 
---
 
## Citation
 
```bibtex
@article{peng2026lighthouse,
  title={Long Context Pre-Training with Lighthouse Attention},
  author={Peng, Bowen and Ghosh, Subho and Quesnelle, Jeffrey},
  journal={arXiv preprint arXiv:2605.06554},
  year={2026}
}
```
 
---
 
## Related Work
 
- [Lost in the Middle](https://arxiv.org/abs/2307.03172) — Liu et al., 2024. Documents the U-shaped positional attention bias that Lighthouse addresses during pre-training.
- [MoBA](https://arxiv.org/abs/2502.13189) — Block-level sparse attention, inference-time.
- [NSA](https://arxiv.org/abs/2502.11089) — DeepSeek Native Sparse Attention, token-level selection.
- [HISA](https://arxiv.org/abs/2406.16008) — Hierarchical indexer for sparse attention scoring.
- [FlashAttention-2](https://arxiv.org/abs/2307.08691) — The dense kernel Lighthouse delegates actual attention computation to.