# Lighthouse Attention

**Original implementation** of *Lighthouse Attention*: a
selection-based hierarchical attention mechanism for training large language
models at very long context. This is the codebase used to produce all
results in the paper.

This repository ships Lighthouse as a single patch on top of
[pytorch/torchtitan][upstream] plus the two Lighthouse-specific
source files. The patch wires in selection, three scorer variants
(`norm`, `dilated`, `gla`), and an optional context-parallel (CP) path,
with the scorer chosen per-config &mdash; no edits to `model.py` required.

[upstream]: https://github.com/pytorch/torchtitan

## Layout

```
lighthouse-attention/
├── README.md                       this file
├── requirements.txt                pinned versions
├── lighthouse-attention.patch      one patch, applies on torchtitan @ 61c25f8d
├── src/
│   ├── lighthouse_selection.py     drop into torchtitan/models/llama3/model/
│   └── lighthouse_selection_cuda.py
└── configs/
    ├── topk/      vary top-K  (1536, 2048, 3072, 4096, 6144) at p=4, L=3
    ├── pool/      vary pool   (p=2, 4, 8)                    at k=1536, L=3
    ├── levels/    vary levels (L=3, 4, 5)                    at k=1536, p=2
    ├── scorer/    norm | dilated | gla                       at k=2048, p=4, L=3
    └── cp/        CP=2 / DP=4 demo                           at k=1536, p=4, L=3 (norm)
```

## Tested versions

```
torch          2.11.0+cu128
CUDA           12.8
cuDNN          9.19.0
GPU            NVIDIA B200 (sm_100)
upstream sha   61c25f8d   (pytorch/torchtitan @ main)
```

## Apply

1. Clone the upstream torchtitan and check out the tested commit:

   ```bash
   git clone https://github.com/pytorch/torchtitan.git
   cd torchtitan
   git checkout 61c25f8d
   ```

2. Drop in the two Lighthouse source files (the patch does not carry these):

   ```bash
   cp /path/to/lighthouse-attention/src/lighthouse_selection.py      torchtitan/models/llama3/model/
   cp /path/to/lighthouse-attention/src/lighthouse_selection_cuda.py torchtitan/models/llama3/model/
   ```

3. Apply the patch:

   ```bash
   git apply /path/to/lighthouse-attention/lighthouse-attention.patch
   ```

4. Install the requirements (Python 3.13, CUDA 12.8 toolkit on the host):

   ```bash
   python3.13 -m venv .venv && source .venv/bin/activate
   pip install -r /path/to/lighthouse-attention/requirements.txt
   pip install -e . --no-deps
   ```

   `requirements.txt` already pins the PyTorch CUDA-12.8 stable index
   (`https://download.pytorch.org/whl/cu128`) via `--extra-index-url`, so
   `torch==2.11.0+cu128` resolves without any extra flags.

   `flash-linear-attention` is only needed if you select
   `lighthouse_scorer = "gla"`. For `norm` (default) or `dilated`, you can
   leave that line out.

## What the patch changes

| File                                          | Hunk |
|-----------------------------------------------|------|
| `torchtitan/models/llama3/model/args.py`      | Adds `dilation`, `hidden_dim`, `use_selection_lighthouse`, `use_lighthouse_cp`, `lighthouse_num_levels`, `lighthouse_pooling_factor`, `lighthouse_topk`, `lighthouse_scorer`, `lighthouse_full_attn_layers` to `TransformerModelArgs`. |
| `torchtitan/models/llama3/model/model.py`     | `_build_lighthouse_scorer(...)` dispatches on `lighthouse_scorer ∈ {norm, dilated, gla}` and refuses non-`norm` under CP. Wires the gate projection (`wg`) for the GLA path. FFN now honors explicit `hidden_dim` when set. |
| `torchtitan/models/llama3/__init__.py`        | Registers ~26 Lighthouse ablation flavors (`ablation_270m_lighthouse_topk*_*`) covering the (k, p, L) grid in the paper, plus dim-matched dense (`*_sdpa`) flavors for the SDPA-resume stage. |
| `torchtitan/models/llama3/infra/parallelize.py` | `apply_compile` now uses `compile_config.fullgraph` so the Lighthouse path can compile each `TransformerBlock` with graph breaks allowed (`@torch.compiler.disable` on the scorers requires this). |
| `torchtitan/distributed/utils.py`             | `create_context_parallel_ctx(..., enable_load_balance=True)` knob so the CP path can opt out of load-balancing. |
| `torchtitan/hf_datasets/text_datasets.py`     | Registers a `c4_local` dataset entry for an on-disk C4 mirror. |
| `torchtitan/train.py`                         | When CP is enabled and the model has Lighthouse-CP modules, calls `set_cp_info(rank, world_size, cp_group)` once and threads `enable_load_balance=is_lighthouse_cp` through the CP context. |
| `torchtitan/config/job_config.py`             | Adds `fullgraph: bool = False` to the `Compile` dataclass. |

The two new files (`lighthouse_selection.py`, `lighthouse_selection_cuda.py`) live in `src/` and are copied in step 2 above.

## Selecting the scorer

The default scorer is `norm`. To switch, set `lighthouse_scorer` on the
flavor in `torchtitan/models/llama3/__init__.py`:

```python
"my_dilated_run": TransformerModelArgs(
    dim=1024, n_layers=30, hidden_dim=1536, n_heads=8, n_kv_heads=8,
    rope_theta=10000,
    use_selection_lighthouse=True,
    lighthouse_num_levels=3,
    lighthouse_pooling_factor=4,
    lighthouse_topk=2048,
    lighthouse_scorer="dilated",       # <-- override here  (or "norm" / "gla")
    dilation=4,
    lighthouse_full_attn_layers=[0, 1, 28, 29],
),
```

Then point your toml at the new flavor:

```toml
[model]
flavor = "my_dilated_run"
```

The CP path explicitly refuses anything other than `norm` at construction
time:

```
ValueError: lighthouse_scorer='dilated' is not supported under context
parallelism. The CP path was validated only for 'norm'; ...
```

## Running a config

The configs in `configs/` use placeholder paths (`<DUMP_FOLDER>`,
`<HF_ASSETS_PATH>`, `<CHECKPOINT_FOLDER>`). Replace them in place or via
`sed` before launching:

```bash
cd torchtitan
sed -e 's|<DUMP_FOLDER>|/scratch/runs/topk1536|' \
    -e 's|<HF_ASSETS_PATH>|/scratch/tokenizer/bytes|' \
    -e 's|<CHECKPOINT_FOLDER>|/scratch/ckpts/topk1536|' \
    /path/to/lighthouse-attention/configs/topk/topk1536.toml \
    > /tmp/run.toml
torchrun --nproc-per-node 8 ./torchtitan/train.py --job.config_file /tmp/run.toml
```

Each config sets `[training] steps = 10000` to match the Stage-1 Lighthouse
phase from the paper. For the SDPA-resume continuation, point a second toml
at the same `[checkpoint] folder` with `[training] steps = 16000` and the
dim-matched dense flavor (`ablation_270m_topk*_pool*_lvl*_sdpa`) that the
patch registers alongside each lighthouse flavor.

## Context-parallel run

```bash
torchrun --nnodes 1 --nproc-per-node 8 ./torchtitan/train.py \
    --job.config_file /path/to/lighthouse-attention/configs/cp/norm_cp2_dp4.toml
```

The toml sets `context_parallel_degree = 2`. The patch wires
`set_cp_info(...)` automatically the first time `forward_backward_step`
runs under CP, and the train loop uses `enable_load_balance=True` for the
ring-attention path while the Lighthouse selection runs shard-locally.
