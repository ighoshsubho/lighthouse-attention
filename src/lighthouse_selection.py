import math

import triton
import triton.language as tl
import triton.language.core as core
from triton.language.standard import _log2, sum as tl_sum, zeros_like

import torch
import torch.nn as nn
import torch.distributed as dist
from einops import rearrange
from functools import lru_cache

from torchtitan.models.llama3.model.lighthouse_selection_cuda import LighthouseSelectionQkKqChunkedTopkNvrtc

# `chunk_simple_gla` powers the optional `gla` scorer; the rest of the file
# does not depend on fla, so we fail soft and let the model error only if
# the user actually selects `lighthouse_scorer = "gla"`.
try:
    from fla.ops.simple_gla import chunk_simple_gla
except ImportError:
    chunk_simple_gla = None

import os
os.environ["TRITON_PRINT_AUTOTUNING"] = "1"

_AUTOTUNE_CONFIGS = [
    triton.Config({}, num_stages=ns, num_warps=nw)
    for ns in [2, 3, 4, 5, 6, 7]
    for nw in [2, 4, 8, 16]
]
_AUTOTUNE_KEY = ["N", "num_levels", "pooling_factor", "TOPK"]

def maybe_autotune(configs, key, default_config=None):
    if default_config is None:
        default_config = configs[0]
    def decorator(fn):
        if os.environ.get("TRITON_AUTOTUNE", "1").strip().lower() in ("0", "false", "off"):
            return triton.autotune([default_config], key=key)(fn)
        return triton.autotune(configs, key=key)(fn)
    return decorator

@triton.jit
def _compare_and_swap(x, ids, flip, i: core.constexpr, n_dims: core.constexpr):
    n_outer: core.constexpr = x.numel >> n_dims
    shape: core.constexpr = [n_outer * 2**i, 2, 2**(n_dims - i - 1)]
    y = core.reshape(x, shape)

    mask = core.arange(0, 2)[None, :, None]
    left = core.broadcast_to(tl_sum(y * (1 - mask), 1)[:, None, :], shape)
    right = core.broadcast_to(tl_sum(y * mask, 1)[:, None, :], shape)
    left = core.reshape(left, x.shape)
    right = core.reshape(right, x.shape)

    y_idx = core.reshape(ids, shape)
    left_idx = core.broadcast_to(tl_sum(y_idx * (1 - mask), 1)[:, None, :], shape)
    right_idx = core.broadcast_to(tl_sum(y_idx * mask, 1)[:, None, :], shape)
    left_idx = core.reshape(left_idx, x.shape)
    right_idx = core.reshape(right_idx, x.shape)

    # Currently we always do sort on float32 hence we use int32 dtype
    idtype = tl.int32
    ileft = left.to(idtype, bitcast=True)
    iright = right.to(idtype, bitcast=True)
    ix = x.to(idtype, bitcast=True)

    bool_flip = flip != 0
    cond = (left > right) ^ bool_flip

    ret = ix ^ core.where(cond, ileft ^ iright, zeros_like(ix))

    new_ids = ids ^ core.where(cond, left_idx ^ right_idx, zeros_like(ids))

    return ret.to(x.dtype, bitcast=True), new_ids


@triton.jit
def _bitonic_merge(x, ids, stage: core.constexpr, order: core.constexpr,
                   n_dims: core.constexpr):
    '''
    order_type 0 == ascending
    order_type 1 == descending
    order_type 2 == alternating
    '''
    n_outer: core.constexpr = x.numel >> n_dims
    core.static_assert(stage <= n_dims)

    if order == 2:
        shape: core.constexpr = [
            n_outer * 2**(n_dims - 1 - stage), 2, 2**stage
        ]
        flip = core.reshape(
            core.broadcast_to(core.arange(0, 2)[None, :, None], shape),
            x.shape)
    else:
        flip = order

    for i in core.static_range(stage):
        x, ids = _compare_and_swap(x, ids, flip, i + (n_dims - stage), n_dims)
    return x, ids


@triton.jit
def argsort(x,
            ids,
            dim: core.constexpr = None,
            descending: core.constexpr = core.CONSTEXPR_0):
    _dim: core.constexpr = len(x.shape) - 1 if dim is None else dim
    core.static_assert(_dim == len(x.shape) - 1,
                       "only minor dimension is currently supported")
    n_dims: core.constexpr = _log2(x.shape[_dim])

    for i in core.static_range(1, n_dims + 1):
        x, ids = _bitonic_merge(x, ids, i, 2 if i < n_dims else descending,
                                n_dims)
    return x, ids

@triton.jit
def merge_topk_gather(
    prev_topk, # [TOPK]
    prev_idxs, # [TOPK]
    new_scores, # [SIZE]
    new_idxs, # [SIZE]
    k: tl.constexpr
):
    SIZE: tl.constexpr = new_scores.shape[0]
    COMBINED_SIZE: tl.constexpr = k + SIZE

    COMBINED_SIZE_POW2: tl.constexpr = triton.next_power_of_2(COMBINED_SIZE)

    combined_col_idx = tl.arange(0, COMBINED_SIZE_POW2)

    prev_topk_gathered = tl.gather(prev_topk, tl.minimum(combined_col_idx, k - 1), axis=0)
    prev_idxs_gathered = tl.gather(prev_idxs, tl.minimum(combined_col_idx, k - 1), axis=0)

    new_col_idx = tl.maximum(combined_col_idx - k, 0)
    new_scores_gathered = tl.gather(new_scores, tl.minimum(new_col_idx, SIZE - 1), axis=0)
    new_idxs_gathered = tl.gather(new_idxs, tl.minimum(new_col_idx, SIZE - 1), axis=0)

    mask_prev = combined_col_idx < k
    mask_new = (combined_col_idx >= k) & (combined_col_idx < COMBINED_SIZE)

    combined_scores = tl.where(mask_prev, prev_topk_gathered, tl.where(mask_new, new_scores_gathered, -1e6))
    combined_indices = tl.where(mask_prev, prev_idxs_gathered, tl.where(mask_new, new_idxs_gathered, -1))

    sorted_scores, sorted_indices = argsort(combined_scores, combined_indices, dim=0, descending=True)

    gather_topk_indices = tl.arange(0, k)
    gather_rejected_indices = tl.arange(k, k + SIZE)

    topk_vals = tl.gather(sorted_scores, gather_topk_indices, axis=0)
    topk_indices = tl.gather(sorted_indices, gather_topk_indices, axis=0)

    rejected_indices = tl.gather(sorted_indices, gather_rejected_indices, axis=0)

    return topk_vals, topk_indices, rejected_indices

@triton.jit
def _encode_indices(
    actual_indices,
    level_absolute_indices,
    cum_pooling_factor
):
    level_orderd_indices = (cum_pooling_factor * level_absolute_indices).to(tl.float32)

    level_orderd_indices_uint = level_orderd_indices.to(tl.uint32, bitcast=True)
    level_orderd_indices_key = tl.where(level_orderd_indices >= 0, level_orderd_indices_uint ^ 0x80000000, ~level_orderd_indices_uint)

    packed_indices = (level_orderd_indices_key.to(tl.uint64) << 32) | actual_indices.to(tl.uint64)
    return packed_indices

@triton.jit
def _encode_indices_selected(
    actual_indices,
    level_absolute_indices,
    cum_pooling_factor,
    level
):
    level_orderd_indices = cum_pooling_factor * (level_absolute_indices + 1) - tl.exp(tl.log(2.0) * (1.0 - level))

    level_orderd_indices_uint = level_orderd_indices.to(tl.uint32, bitcast=True)
    level_orderd_indices_key = tl.where(level_orderd_indices >= 0, level_orderd_indices_uint ^ 0x80000000, ~level_orderd_indices_uint)

    packed_indices = (level_orderd_indices_key.to(tl.uint64) << 32) | actual_indices.to(tl.uint64)
    return packed_indices

@maybe_autotune(_AUTOTUNE_CONFIGS, _AUTOTUNE_KEY)
@triton.jit
def lighthouse_selection_scores_qk_kq_chunked_topk_kernel(
    scores_ptr_qk,
    scores_ptr_kq,
    indices_ptr,
    stride_scores_qk_B, stride_scores_qk_H, stride_scores_qk_N,
    stride_scores_kq_B, stride_scores_kq_H, stride_scores_kq_N,
    stride_indices_B, stride_indices_H, stride_indices_selected,
    num_levels,
    pooling_factor,
    POOLING_FACTOR_CONST: tl.constexpr,
    TOPK: tl.constexpr,
    KVH: tl.constexpr,
    N: tl.constexpr,
    N_CHUNK_COARSEST: tl.constexpr,
    NUM_SELECTION_INDICES_CHUNK: tl.constexpr,
    num_stages: tl.constexpr,
    warp_specialize: tl.constexpr,
):
    TOPK_HALF: tl.constexpr = TOPK // 2
    
    index_BH = tl.program_id(0).to(tl.int64)
    index_B = index_BH // KVH
    index_KVH = index_BH % KVH

    index_seq_chunk = tl.program_id(1).to(tl.int64)

    selection_indices_start = index_seq_chunk * NUM_SELECTION_INDICES_CHUNK

    scores_ptr_qk += index_B * stride_scores_qk_B + index_KVH * stride_scores_qk_H
    scores_ptr_kq += index_B * stride_scores_kq_B + index_KVH * stride_scores_kq_H
    indices_ptr += index_B * stride_indices_B + index_KVH * stride_indices_H + selection_indices_start * stride_indices_selected

    tile_start = index_seq_chunk * N_CHUNK_COARSEST
    tile_end = tile_start + N_CHUNK_COARSEST

    parent_indices = tl.full([TOPK], value=-1, dtype=tl.int32)

    write_offset = tl.arange(0, TOPK_HALF) # POOLING_FACTOR_CONST
    write_offset_selected = tl.arange(0, TOPK)

    for level in tl.range(num_levels, 0, -1):
        c_l = tl.exp(tl.log(pooling_factor) * (level - 1)).to(tl.int32)

        K_end = N // c_l - 1

        level_offset = 0
        if level == 1:
            level_offset = 0
        else:
            level_offset = (N * (pooling_factor - tl.exp(tl.log(pooling_factor) * (2 - level))) / (pooling_factor - 1)).to(tl.int32)

        if level == num_levels:
            num_K_indices = (tile_end - tile_start).to(tl.int32)
            use_parents = False
            K_indices = tl.full([TOPK], value=-1, dtype=tl.int32)
        else:
            num_K_indices = POOLING_FACTOR_CONST * TOPK
            use_parents = True
            K_indices = parent_indices

        topk_scores_qk = tl.full([TOPK_HALF], value=-1e6, dtype=tl.float32)
        topk_indices_qk = tl.full([TOPK_HALF], value=-1, dtype=tl.int32)

        topk_scores_kq = tl.full([TOPK_HALF], value=-1e6, dtype=tl.float32)
        topk_indices_kq = tl.full([TOPK_HALF], value=-1, dtype=tl.int32)

        total_blocks = tl.cdiv(num_K_indices, TOPK_HALF) # POOLING_FACTOR_CONST

        K_end_absolute = tl.where(K_end >= 0, K_end + level_offset, -1)

        for block_idx in tl.range(0, total_blocks, num_stages=num_stages, warp_specialize=warp_specialize):
            kv_chunk_start = block_idx * TOPK_HALF # POOLING_FACTOR_CONST
            kv_chunk_start = tl.multiple_of(kv_chunk_start, TOPK_HALF) # POOLING_FACTOR_CONST

            chunk_col_idx = tl.arange(0, TOPK_HALF) 
            abs_col_idx = kv_chunk_start + chunk_col_idx

            if use_parents:
                shift = POOLING_FACTOR_CONST.bit_length() - 1
                parent_idx = abs_col_idx >> shift
                child_offset = abs_col_idx & (POOLING_FACTOR_CONST - 1)

                parent_vals = tl.gather(K_indices, parent_idx, axis=0).to(tl.int32)

                K_indices_chunk = parent_vals * POOLING_FACTOR_CONST + child_offset

                K_indices_chunk = tl.where(parent_vals >= 0, K_indices_chunk, -1)

                col_valid_mask = (abs_col_idx < num_K_indices) & (K_indices_chunk <= K_end) & (K_indices_chunk >= 0)
                K_indices_chunk = tl.where(col_valid_mask, K_indices_chunk, -1)

            else:
                actual_token_idx = tile_start + abs_col_idx
                col_valid_mask = abs_col_idx < num_K_indices
                K_indices_chunk = tl.where(col_valid_mask, actual_token_idx.to(tl.int32), -1)

            K_indices_chunk_absolute = tl.where(K_indices_chunk >= 0, K_indices_chunk + level_offset, -1)

            score_offsets_qk = K_indices_chunk_absolute * stride_scores_qk_N
            scores_valid_qk = (K_indices_chunk_absolute <= K_end_absolute) & (K_indices_chunk_absolute >= 0) & (K_end >= 0)  

            if level > 1:
                S_chunk_qk = tl.load(scores_ptr_qk + score_offsets_qk, mask=scores_valid_qk, other=-1e6)

                topk_scores_qk, topk_indices_qk, rejected_indices_qk = merge_topk_gather(
                    topk_scores_qk,
                    topk_indices_qk,
                    S_chunk_qk,
                    K_indices_chunk,
                    TOPK_HALF
                )

                rejected_qk_absolute_indices = tl.where(rejected_indices_qk >= 0, rejected_indices_qk + level_offset, -1)
                score_offsets_kq = rejected_qk_absolute_indices * stride_scores_kq_N

                scores_valid_kq = (rejected_qk_absolute_indices <= K_end_absolute) & (K_end >= 0) & (rejected_qk_absolute_indices >= 0)
                S_chunk_kq = tl.load(scores_ptr_kq + score_offsets_kq, mask=scores_valid_kq, other=-1e6)

                topk_scores_kq, topk_indices_kq, rejected_indices_kq = merge_topk_gather(
                    topk_scores_kq,
                    topk_indices_kq,
                    S_chunk_kq,
                    rejected_indices_qk,
                    TOPK_HALF
                )

                valid_mask = rejected_indices_kq >= 0
                absolute_indices = tl.where(valid_mask, rejected_indices_kq + level_offset, -1)
                causal_mask = (absolute_indices <= K_end_absolute) & (K_end >= 0)

                is_rejected = causal_mask & valid_mask

                if block_idx >= TOPK // TOPK_HALF: # TOPK // POOLING_FACTOR_CONST
                    tl.store(indices_ptr + write_offset * stride_indices_selected, _encode_indices(absolute_indices, rejected_indices_kq, c_l), mask=is_rejected)
                    indices_ptr += TOPK_HALF * stride_indices_selected # POOLING_FACTOR_CONST

                if block_idx == total_blocks - 1:
                    topk_indces = tl.cat(topk_indices_qk, topk_indices_kq, can_reorder=True)
                    topk_absolute_indices = tl.where(topk_indces >= 0, topk_indces + level_offset, -1)
                    tl.store(indices_ptr + write_offset_selected * stride_indices_selected, _encode_indices_selected(topk_absolute_indices, topk_indces, c_l, level), mask=topk_indces >= 0)
                    indices_ptr += TOPK * stride_indices_selected
            else:
                is_selected = scores_valid_qk

                tl.store(indices_ptr + write_offset * stride_indices_selected, _encode_indices(K_indices_chunk_absolute, K_indices_chunk, c_l), mask=is_selected)
                indices_ptr += TOPK_HALF * stride_indices_selected # POOLING_FACTOR_CONST

        parent_indices = tl.cat(topk_indices_qk, topk_indices_kq, can_reorder=True)

@lru_cache(maxsize=100, typed=False)
def compute_output_indices_selection(N: int, num_levels: int, pooling_factor: float, topk: int) -> int:
    selection_indices = 0
    for level in range(num_levels, 0, -1):
        c_l = int(pooling_factor ** (level - 1))
        if level == num_levels:
            max_indices = (N // c_l)
            assert (N // c_l) % pooling_factor == 0, f"Level {level} size must be divisible by pooling factor"
            selection_indices += max_indices
        else:
            max_indices = pooling_factor * topk
            selection_indices += max_indices

    return selection_indices

class LighthouseSelectionQkKqChunkedTopk:
    @torch.no_grad()
    @staticmethod
    @torch.compiler.disable
    def forward(scores_qk, scores_kq, num_levels, pooling_factor, topk, actual_seq_len):
        B, KVH, _ = scores_qk.shape
        device = scores_qk.device

        assert scores_qk.shape == scores_kq.shape, "scores_qk and scores_kq must have the same shape"
        assert topk % 128 == 0, "Topk must be a multiple of TOPK_PER_TILE (128)"
        assert triton.next_power_of_2(pooling_factor) == pooling_factor, "Pooling factor must be a power of 2"
        assert topk // 2 >= pooling_factor, "Topk half must be multiple of pooling factor"

        pooling_factor_int = int(pooling_factor)

        selection_indices = compute_output_indices_selection(actual_seq_len, num_levels, pooling_factor, topk)
        indices_ptr = torch.full((B, KVH, selection_indices), -1, dtype=torch.int64, device=device)

        scores_ptr_qk = scores_qk.to(torch.float32).contiguous()
        scores_ptr_kq = scores_kq.to(torch.float32).contiguous()

        TOPK_PER_TILE = 128
        NUM_TILES = topk // TOPK_PER_TILE
        N_CHUNK_COARSEST = (actual_seq_len // (pooling_factor ** (num_levels - 1))) // NUM_TILES

        assert TOPK_PER_TILE <= N_CHUNK_COARSEST, "TOPK_PER_TILE must be greater than N_CHUNK_COARSEST"

        N_CHUNK = actual_seq_len // NUM_TILES
        NUM_SELECTION_INDICES_CHUNK = compute_output_indices_selection(N_CHUNK, num_levels, pooling_factor, TOPK_PER_TILE)

        grid = lambda args: (B * KVH, NUM_TILES)

        lighthouse_selection_scores_qk_kq_chunked_topk_kernel[grid](
            scores_ptr_qk,
            scores_ptr_kq,
            indices_ptr,
            scores_ptr_qk.stride(0), scores_ptr_qk.stride(1), scores_ptr_qk.stride(2),
            scores_ptr_kq.stride(0), scores_ptr_kq.stride(1), scores_ptr_kq.stride(2),
            indices_ptr.stride(0), indices_ptr.stride(1), indices_ptr.stride(2),
            num_levels,
            pooling_factor=float(pooling_factor),
            POOLING_FACTOR_CONST=int(pooling_factor_int),
            TOPK=TOPK_PER_TILE,
            KVH=KVH,
            N=actual_seq_len,
            N_CHUNK_COARSEST= N_CHUNK_COARSEST,
            NUM_SELECTION_INDICES_CHUNK=NUM_SELECTION_INDICES_CHUNK,
            warp_specialize=False,
        )

        packed_sorted, _ = torch.sort(indices_ptr, descending=False)
        actual_indices = (packed_sorted & 0xFFFFFFFF).to(torch.int32)

        return actual_indices

@triton.jit
def _scatter_fanout_fwd_kernel(
    attn_out_ptr, indices_ptr, rank_ids_ptr, output_ptr,
    N_shard, N_total, S_sel, D,
    num_levels: tl.constexpr,
    pooling_factor: tl.constexpr,
    MAX_PF: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    bh = tl.program_id(0)
    s = tl.program_id(1)
    flat_s = bh * S_sel + s
    valid = s < S_sel

    idx = tl.load(indices_ptr + flat_s, mask=valid, other=-1)
    valid = valid & (idx >= 0)

    rank_id = tl.load(rank_ids_ptr + flat_s, mask=valid, other=0)
    global_off = rank_id * N_shard

    base_pos = 0
    fan_out = 1
    cum_size = 0
    pf_l = 1
    for _l in tl.static_range(num_levels):
        level_size = N_shard // pf_l
        level_start = cum_size
        level_end = cum_size + level_size

        in_level = (idx >= level_start) & (idx < level_end)
        local_pos = idx - level_start
        bp = global_off + local_pos * pf_l

        base_pos = tl.where(in_level, bp, base_pos)
        fan_out = tl.where(in_level, pf_l, fan_out)

        cum_size = level_end
        pf_l = pf_l * pooling_factor

    c_offs = tl.arange(0, MAX_PF)                              # [MAX_PF]
    d_offs = tl.arange(0, BLOCK_D)                              # [BLOCK_D]
    out_positions = base_pos + (fan_out - 1) + c_offs           # [MAX_PF]

    c_mask = valid & (c_offs < fan_out) & (out_positions >= 0) & (out_positions < N_total)  # [MAX_PF]
    mask_2d = c_mask[:, None] & (d_offs[None, :] < D)          # [MAX_PF, BLOCK_D]

    ao_val = tl.load(attn_out_ptr + flat_s * D + d_offs, mask=valid & (d_offs < D), other=0.0)  # [BLOCK_D]
    ao_2d = tl.broadcast_to(ao_val[None, :], [MAX_PF, BLOCK_D])

    out_bh = bh * N_total * D
    ptrs = output_ptr + out_bh + out_positions[:, None] * D + d_offs[None, :]  # [MAX_PF, BLOCK_D]
    tl.atomic_add(ptrs, ao_2d, mask=mask_2d, sem="relaxed", scope="gpu")


@triton.jit
def _scatter_fanout_bwd_kernel(
    grad_output_ptr, indices_ptr, rank_ids_ptr, grad_attn_ptr,
    N_shard, N_total, S_sel, D,
    num_levels: tl.constexpr,
    pooling_factor: tl.constexpr,
    MAX_PF: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    bh = tl.program_id(0)
    s = tl.program_id(1)
    flat_s = bh * S_sel + s
    valid = s < S_sel

    idx = tl.load(indices_ptr + flat_s, mask=valid, other=-1)
    valid = valid & (idx >= 0)

    rank_id = tl.load(rank_ids_ptr + flat_s, mask=valid, other=0)
    global_off = rank_id * N_shard

    base_pos = 0
    fan_out = 1
    cum_size = 0
    pf_l = 1
    for _l in tl.static_range(num_levels):
        level_size = N_shard // pf_l
        level_start = cum_size
        level_end = cum_size + level_size

        in_level = (idx >= level_start) & (idx < level_end)
        local_pos = idx - level_start
        bp = global_off + local_pos * pf_l

        base_pos = tl.where(in_level, bp, base_pos)
        fan_out = tl.where(in_level, pf_l, fan_out)

        cum_size = level_end
        pf_l = pf_l * pooling_factor

    c_offs = tl.arange(0, MAX_PF)                              # [MAX_PF]
    d_offs = tl.arange(0, BLOCK_D)                              # [BLOCK_D]
    out_positions = base_pos + (fan_out - 1) + c_offs           # [MAX_PF]

    c_mask = valid & (c_offs < fan_out) & (out_positions >= 0) & (out_positions < N_total)  # [MAX_PF]
    mask_2d = c_mask[:, None] & (d_offs[None, :] < D)          # [MAX_PF, BLOCK_D]

    go_bh = bh * N_total * D
    ptrs = grad_output_ptr + go_bh + out_positions[:, None] * D + d_offs[None, :]  # [MAX_PF, BLOCK_D]
    g_2d = tl.load(ptrs, mask=mask_2d, other=0.0)              # [MAX_PF, BLOCK_D]
    grad_acc = tl.sum(g_2d, axis=0)                             # [BLOCK_D]

    d_mask = valid & (d_offs < D)
    tl.store(grad_attn_ptr + flat_s * D + d_offs, grad_acc.to(g_2d.dtype), mask=d_mask)


class _ScatterToBaseAutograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, attn_out, indices_hp, rank_ids_hp, N_shard, N_total, num_levels, pooling_factor):
        B, H, S_sel, D = attn_out.shape
        device = attn_out.device
        dtype = attn_out.dtype

        attn_out = attn_out.contiguous()
        indices_hp = indices_hp.contiguous()
        rank_ids_hp = rank_ids_hp.contiguous()

        output = torch.zeros(B, H, N_total, D, device=device, dtype=dtype)

        BH = B * H
        MAX_PF = pooling_factor ** (num_levels - 1)
        BLOCK_D = triton.next_power_of_2(D)

        grid = (BH, S_sel)
        _scatter_fanout_fwd_kernel[grid](
            attn_out, indices_hp, rank_ids_hp, output,
            N_shard, N_total, S_sel, D,
            num_levels, pooling_factor, MAX_PF,
            BLOCK_D=BLOCK_D,
        )

        ctx.save_for_backward(indices_hp, rank_ids_hp)
        ctx.N_shard = N_shard
        ctx.N_total = N_total
        ctx.num_levels = num_levels
        ctx.pooling_factor = pooling_factor
        ctx.S_sel = S_sel
        ctx.D = D

        return output

    @staticmethod
    def backward(ctx, grad_output):
        indices_hp, rank_ids_hp = ctx.saved_tensors
        B, H = indices_hp.shape[:2]
        S_sel = ctx.S_sel
        D = ctx.D
        device = grad_output.device
        dtype = grad_output.dtype

        grad_output = grad_output.contiguous()
        grad_attn = torch.zeros(B, H, S_sel, D, device=device, dtype=dtype)

        BH = B * H
        MAX_PF = ctx.pooling_factor ** (ctx.num_levels - 1)
        BLOCK_D = triton.next_power_of_2(D)

        grid = (BH, S_sel)
        _scatter_fanout_bwd_kernel[grid](
            grad_output, indices_hp, rank_ids_hp, grad_attn,
            ctx.N_shard, ctx.N_total, S_sel, D,
            ctx.num_levels, ctx.pooling_factor, MAX_PF,
            BLOCK_D=BLOCK_D,
        )

        return grad_attn, None, None, None, None, None, None

class DialatedLighthouseNormScorer(torch.nn.Module):
    def __init__(
        self,
        pooling_factor: int = 4,
        levels: int = 3,
    ):
        super().__init__()

        self.pooling_factor = pooling_factor
        self.levels = levels

    def forward(self, xq, xk, xv) -> torch.Tensor:
        b, s, h, d = xq.shape

        scores_qk = xq.norm(dim=-1)
        scores_kq = xk.norm(dim=-1)

        xq_levels = [xq]
        xk_levels = [xk]
        xv_levels = [xv]
        scores_qk_levels = [scores_qk]
        scores_kq_levels = [scores_kq]

        for level in range(1, self.levels):
            pf = self.pooling_factor ** level
            level_len = s // pf
            xq_levels.append(xq[:, :level_len * pf].view(b, level_len, pf, h, d).mean(2))
            xk_levels.append(xk[:, :level_len * pf].view(b, level_len, pf, h, d).mean(2))
            xv_levels.append(xv[:, :level_len * pf].view(b, level_len, pf, h, d).mean(2))
            scores_qk_levels.append(scores_qk[:, :level_len * pf].view(b, level_len, pf, h).max(2).values)
            scores_kq_levels.append(scores_kq[:, :level_len * pf].view(b, level_len, pf, h).max(2).values)

        xq_cat = torch.cat(xq_levels, dim=1).transpose(1, 2)
        xk_cat = torch.cat(xk_levels, dim=1).transpose(1, 2)
        xv_cat = torch.cat(xv_levels, dim=1).transpose(1, 2)
        scores_qk_cat = torch.cat(scores_qk_levels, dim=1).transpose(1, 2)
        scores_kq_cat = torch.cat(scores_kq_levels, dim=1).transpose(1, 2)

        return scores_qk_cat, scores_kq_cat, xq_cat, xk_cat, xv_cat

class DilatedLighthouseScorer(torch.nn.Module):
    def __init__(
        self,
        attention,
        dilation: int = 4,
        pooling_factor: int = 4,
        levels: int = 3,
    ):
        super().__init__()

        self.dilation = dilation

        self.pooling_factor = pooling_factor
        self.levels = levels

        self.attention = attention

    def forward(self, xq, xk, xv) -> torch.Tensor:
        b, s, h, d = xq.shape

        xq = xq.view(b, s//self.dilation, self.dilation * h, d)
        xk = xk.view(b, s//self.dilation, self.dilation * h, d)
        xv = xv.view(b, s//self.dilation, self.dilation * h, d)

        xq_t = xq.transpose(1, 2)
        xk_t = xk.transpose(1, 2)
        xv_t = xv.transpose(1, 2)

        out_qk = self.attention(xq_t, xk_t, xv_t).transpose(1, 2).view(b, s, h, d)
        out_kq = self.attention(xk_t, xq_t, xv_t).transpose(1, 2).view(b, s, h, d)

        scores_qk = out_qk.norm(dim=-1)
        scores_kq = out_kq.norm(dim=-1)

        scores_qk = scores_qk.transpose(1, 2)  # [B, h, S]
        scores_kq = scores_kq.transpose(1, 2)  # [B, h, S]

        xq = rearrange(xq, 'b sdi (di h) d -> b h d (sdi di)', di=self.dilation, h=h)
        xk = rearrange(xk, 'b sdi (di h) d -> b h d (sdi di)', di=self.dilation, h=h)
        xv = rearrange(xv, 'b sdi (di h) d -> b h d (sdi di)', di=self.dilation, h=h)

        S_total = 0
        for l in range(self.levels):
            S_total += s // (self.pooling_factor ** l)

        xq_cat = torch.empty(b, h, d, S_total, device=xq.device, dtype=xq.dtype)
        xk_cat = torch.empty(b, h, d, S_total, device=xk.device, dtype=xk.dtype)
        xv_cat = torch.empty(b, h, d, S_total, device=xv.device, dtype=xv.dtype)
        scores_qk_cat = torch.empty(b, h, S_total, device=scores_qk.device, dtype=scores_qk.dtype)
        scores_kq_cat = torch.empty(b, h, S_total, device=scores_kq.device, dtype=scores_kq.dtype)

        xq_cat[..., :s] = xq
        xk_cat[..., :s] = xk
        xv_cat[..., :s] = xv
        scores_qk_cat[..., :s] = scores_qk
        scores_kq_cat[..., :s] = scores_kq

        offset = s
        for level in range(1, self.levels):
            pf = self.pooling_factor ** level
            level_len = s // pf
            end = offset + level_len

            xq_cat[..., offset:end] = xq[..., :level_len * pf].view(b, h, d, level_len, pf).mean(-1)
            xk_cat[..., offset:end] = xk[..., :level_len * pf].view(b, h, d, level_len, pf).mean(-1)
            xv_cat[..., offset:end] = xv[..., :level_len * pf].view(b, h, d, level_len, pf).mean(-1)
            scores_qk_cat[..., offset:end] = scores_qk[..., :level_len * pf].view(b, h, level_len, pf).max(-1).values
            scores_kq_cat[..., offset:end] = scores_kq[..., :level_len * pf].view(b, h, level_len, pf).max(-1).values

            offset = end

        return scores_qk_cat, scores_kq_cat, xq_cat.transpose(2, 3), xk_cat.transpose(2, 3), xv_cat.transpose(2, 3)


class DilatedLighthouseGLAScorer(torch.nn.Module):
    """Gated-linear-attention scorer (non-CP only). Requires the optional
    `fla` (`flash-linear-attention`) package; selecting this scorer with fla
    not installed will raise at the first forward."""

    def __init__(
        self,
        head_dim: int,
        dilation: int = 4,
        pooling_factor: int = 4,
        levels: int = 3,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.dilation = dilation
        self.pooling_factor = pooling_factor
        self.levels = levels
        self.scaling = 1.0 / math.sqrt(head_dim)

    @torch.compiler.disable
    def forward(self, xq, xk, xv, gk) -> torch.Tensor:
        if chunk_simple_gla is None:
            raise RuntimeError(
                "lighthouse_scorer='gla' requires the `fla` package "
                "(`pip install flash-linear-attention`)."
            )
        b, s, h, d = xq.shape

        xq_d = xq.view(b, s // self.dilation, self.dilation * h, d)
        xk_d = xk.view(b, s // self.dilation, self.dilation * h, d)
        xv_d = xv.view(b, s // self.dilation, self.dilation * h, d)
        gk_d = gk.view(b, s // self.dilation, self.dilation * h)

        output_qk, _ = chunk_simple_gla(xq_d, xk_d, xv_d, gk_d, scale=self.scaling)
        output_kq, _ = chunk_simple_gla(xk_d, xq_d, xv_d, gk_d, scale=self.scaling)

        output_qk = output_qk.view(b, s, h, d)
        output_kq = output_kq.view(b, s, h, d)

        scores_qk = output_qk.norm(dim=-1)  # [B, S, H]
        scores_kq = output_kq.norm(dim=-1)

        xq_levels = [xq]
        xk_levels = [xk]
        xv_levels = [xv]
        scores_qk_levels = [scores_qk]
        scores_kq_levels = [scores_kq]

        for level in range(1, self.levels):
            pf = self.pooling_factor ** level
            level_len = s // pf
            xq_levels.append(xq[:, :level_len * pf].view(b, level_len, pf, h, d).mean(2))
            xk_levels.append(xk[:, :level_len * pf].view(b, level_len, pf, h, d).mean(2))
            xv_levels.append(xv[:, :level_len * pf].view(b, level_len, pf, h, d).mean(2))
            scores_qk_levels.append(scores_qk[:, :level_len * pf].view(b, level_len, pf, h).max(2).values)
            scores_kq_levels.append(scores_kq[:, :level_len * pf].view(b, level_len, pf, h).max(2).values)

        xq_cat = torch.cat(xq_levels, dim=1).transpose(1, 2)
        xk_cat = torch.cat(xk_levels, dim=1).transpose(1, 2)
        xv_cat = torch.cat(xv_levels, dim=1).transpose(1, 2)
        scores_qk_cat = torch.cat(scores_qk_levels, dim=1).transpose(1, 2)
        scores_kq_cat = torch.cat(scores_kq_levels, dim=1).transpose(1, 2)

        return scores_qk_cat, scores_kq_cat, xq_cat, xk_cat, xv_cat


def scatter_to_base_sequence(
    attn_out,
    indices_hp,
    rank_ids_hp,
    N_shard: int,
    N_total: int,
    num_levels: int,
    pooling_factor: int,
):
    return _ScatterToBaseAutograd.apply(
        attn_out, indices_hp, rank_ids_hp,
        N_shard, N_total, num_levels, pooling_factor,
    )

def pyramid_last_positions(seqlen: int, num_levels: int, pf: int, device) -> torch.Tensor:
    parts = []
    for level in range(num_levels):
        cov = pf ** level
        nlv = seqlen // cov
        j = torch.arange(nlv, device=device, dtype=torch.long)
        parts.append((j + 1) * cov - 1)
    return torch.cat(parts)


def apply_rope_at(x: torch.Tensor, pos: torch.Tensor, theta: float) -> torch.Tensor:
    b, h, n, d = x.shape
    inv_freq = 1.0 / (theta ** (torch.arange(0, d, 2, device=x.device, dtype=torch.float32) / d))
    ang = pos.unsqueeze(-1).float() * inv_freq
    fc = torch.polar(torch.ones_like(ang), ang)
    xc = torch.view_as_complex(x.float().reshape(b, h, n, d // 2, 2))
    return torch.view_as_real(xc * fc).reshape(b, h, n, d).type_as(x)


_ASSERT_ROPE_MONOTONIC = os.environ.get("LIGHTHOUSE_ASSERT_ROPE_MONOTONIC", "0") == "1"


def _assert_rope_monotonic(rope_pos: torch.Tensor, tag: str = "") -> None:
    if not _ASSERT_ROPE_MONOTONIC:
        return
    if not bool((rope_pos.diff(dim=-1) >= 0).all()):
        raise AssertionError(f"rope_pos not non-decreasing after gather ({tag})")


class LighthouseLocal(nn.Module):
    def __init__(
        self,
        attention,
        scorer: DilatedLighthouseScorer | DialatedLighthouseNormScorer,
        num_levels: int = 3,
        pooling_factor: int = 4,
        topk: int = 1024,
        head_dim: int = 128,
        rope_theta: float = 10000.0,
    ):
        super().__init__()
        self.scorer = scorer
        self.num_levels = num_levels
        self.pooling_factor = pooling_factor
        self.topk = topk
        self.head_dim = head_dim
        self.rope_theta = rope_theta
        self._pos_table = None
        self._pos_len = None

        self.attention = attention

    def _pos_table_for(self, seqlen, device):
        if self._pos_table is None or self._pos_len != seqlen:
            self._pos_table = pyramid_last_positions(seqlen, self.num_levels, self.pooling_factor, device)
            self._pos_len = seqlen
        return self._pos_table

    def forward(self, xq, xk, xv):
        _, s, _, d = xq.shape

        scores_qk_cat, scores_kq_cat, xq_cat, xk_cat, xv_cat = self.scorer(xq, xk, xv)

        with torch.no_grad():
            indices = LighthouseSelectionQkKqChunkedTopkNvrtc.forward(
                scores_qk_cat.to(torch.float32), scores_kq_cat.to(torch.float32),
                self.num_levels, self.pooling_factor, self.topk, s,
            )

        rope_pos = self._pos_table_for(s, xq.device)[indices]
        _assert_rope_monotonic(rope_pos, "local")

        selected_xq = torch.gather(
            xq_cat, dim=2,
            index=indices.unsqueeze(-1).expand(-1, -1, -1, d),
        )

        selected_xk = torch.gather(
            xk_cat, dim=2,
            index=indices.unsqueeze(-1).expand(-1, -1, -1, d),
        )

        selected_xv = torch.gather(
            xv_cat, dim=2,
            index=indices.unsqueeze(-1).expand(-1, -1, -1, d),
        )

        selected_xq = apply_rope_at(selected_xq, rope_pos, self.rope_theta)
        selected_xk = apply_rope_at(selected_xk, rope_pos, self.rope_theta)

        attn_out = self.attention(selected_xq, selected_xk, selected_xv)

        rank_ids = torch.zeros_like(indices)

        output = scatter_to_base_sequence(
            attn_out, indices, rank_ids,
            s, s,
            self.num_levels, self.pooling_factor,
        )

        return output


def _unstripe_impl(x, world_size, cp_group, seq_dim=2):
    W = world_size
    shape = list(x.shape)
    N_shard = shape[seq_dim]
    chunk = N_shard // W
    assert N_shard % W == 0, f"N_shard={N_shard} must be divisible by world_size={W}"
    new_shape = shape[:seq_dim] + [W, chunk] + shape[seq_dim + 1:]
    x = x.reshape(new_shape).contiguous()
    send_list = [t.contiguous() for t in x.unbind(seq_dim)]
    recv_list = [torch.empty_like(send_list[0]) for _ in range(W)]
    dist.all_to_all(recv_list, send_list, group=cp_group)
    x = torch.stack(recv_list, dim=seq_dim)
    x = x.transpose(seq_dim, seq_dim + 1)
    out_shape = shape[:seq_dim] + [N_shard] + shape[seq_dim + 1:]
    return x.reshape(out_shape).contiguous()


def _stripe_impl(x, world_size, cp_group, seq_dim=2):
    W = world_size
    shape = list(x.shape)
    N_shard = shape[seq_dim]
    chunk = N_shard // W
    assert N_shard % W == 0, f"N_shard={N_shard} must be divisible by world_size={W}"
    new_shape = shape[:seq_dim] + [chunk, W] + shape[seq_dim + 1:]
    x = x.reshape(new_shape)
    x = x.transpose(seq_dim, seq_dim + 1).contiguous()
    send_list = [t.contiguous() for t in x.unbind(seq_dim)]
    recv_list = [torch.empty_like(send_list[0]) for _ in range(W)]
    dist.all_to_all(recv_list, send_list, group=cp_group)
    x = torch.stack(recv_list, dim=seq_dim)
    out_shape = shape[:seq_dim] + [N_shard] + shape[seq_dim + 1:]
    return x.reshape(out_shape).contiguous()


class UnstripeFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, world_size, cp_group, seq_dim):
        ctx.world_size = world_size
        ctx.cp_group = cp_group
        ctx.seq_dim = seq_dim
        return _unstripe_impl(x, world_size, cp_group, seq_dim)

    @staticmethod
    def backward(ctx, grad_output):
        return _stripe_impl(grad_output, ctx.world_size, ctx.cp_group, ctx.seq_dim), None, None, None


def unstripe_tensor(x, world_size, cp_group, seq_dim=2):
    return UnstripeFunction.apply(x, world_size, cp_group, seq_dim)


class SeqToHeadParallel(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, world_size, cp_group):
        ctx.world_size = world_size
        ctx.cp_group = cp_group
        ctx.S_local = x.shape[2]
        B, H, S, D = x.shape
        assert H % world_size == 0, f"H={H} must be divisible by world_size={world_size}"
        H_local = H // world_size
        x_split = x.reshape(B, world_size, H_local, S, D).contiguous()
        send_list = [t.contiguous() for t in x_split.unbind(1)]
        recv_list = [torch.empty_like(send_list[0]) for _ in range(world_size)]
        dist.all_to_all(recv_list, send_list, group=cp_group)
        return torch.cat(recv_list, dim=2)

    @staticmethod
    def backward(ctx, grad_output):
        world_size = ctx.world_size
        cp_group = ctx.cp_group
        S_local = ctx.S_local
        B, H_local, _, D = grad_output.shape
        grad_split = grad_output.reshape(B, H_local, world_size, S_local, D)
        grad_split = grad_split.permute(0, 2, 1, 3, 4).contiguous()
        send_list = [t.contiguous() for t in grad_split.unbind(1)]
        recv_list = [torch.empty_like(send_list[0]) for _ in range(world_size)]
        dist.all_to_all(recv_list, send_list, group=cp_group)
        return torch.cat(recv_list, dim=1), None, None


class HeadToSeqParallel(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, world_size, S_local, cp_group):
        ctx.world_size = world_size
        ctx.cp_group = cp_group
        B, H_local, _, D = x.shape
        x_split = x.reshape(B, H_local, world_size, S_local, D)
        x_split = x_split.permute(0, 2, 1, 3, 4).contiguous()
        send_list = [t.contiguous() for t in x_split.unbind(1)]
        recv_list = [torch.empty_like(send_list[0]) for _ in range(world_size)]
        dist.all_to_all(recv_list, send_list, group=cp_group)
        return torch.cat(recv_list, dim=1)

    @staticmethod
    def backward(ctx, grad_output):
        world_size = ctx.world_size
        cp_group = ctx.cp_group
        B, H, S_local, D = grad_output.shape
        H_local = H // world_size
        grad_split = grad_output.reshape(B, world_size, H_local, S_local, D).contiguous()
        send_list = [t.contiguous() for t in grad_split.unbind(1)]
        recv_list = [torch.empty_like(send_list[0]) for _ in range(world_size)]
        dist.all_to_all(recv_list, send_list, group=cp_group)
        grad_input = torch.stack(recv_list, dim=2)
        return grad_input.reshape(B, H_local, -1, D), None, None, None


def seq_to_head_parallel(x, world_size, cp_group):
    return SeqToHeadParallel.apply(x, world_size, cp_group)


def head_to_seq_parallel(x, world_size, S_local, cp_group):
    return HeadToSeqParallel.apply(x, world_size, S_local, cp_group)


class DialatedLighthouseNormScorerCP(torch.nn.Module):
    def __init__(
        self,
        pooling_factor: int = 4,
        levels: int = 3,
    ):
        super().__init__()
        self.pooling_factor = pooling_factor
        self.levels = levels
        self._checked = False


    def forward(self, xq, xk, xv) -> torch.Tensor:
        b, s, h, d = xq.shape

        scores_qk = xq.norm(dim=-1)
        scores_kq = xk.norm(dim=-1)

        xq_levels = [xq]
        xk_levels = [xk]
        xv_levels = [xv]
        scores_qk_levels = [scores_qk]
        scores_kq_levels = [scores_kq]

        for level in range(1, self.levels):
            pf = self.pooling_factor ** level
            level_len = s // pf
            xq_levels.append(xq[:, :level_len * pf].view(b, level_len, pf, h, d).mean(2))
            xk_levels.append(xk[:, :level_len * pf].view(b, level_len, pf, h, d).mean(2))
            xv_levels.append(xv[:, :level_len * pf].view(b, level_len, pf, h, d).mean(2))
            scores_qk_levels.append(scores_qk[:, :level_len * pf].view(b, level_len, pf, h).max(2).values)
            scores_kq_levels.append(scores_kq[:, :level_len * pf].view(b, level_len, pf, h).max(2).values)

        xq_cat = torch.cat(xq_levels, dim=1).transpose(1, 2)
        xk_cat = torch.cat(xk_levels, dim=1).transpose(1, 2)
        xv_cat = torch.cat(xv_levels, dim=1).transpose(1, 2)
        scores_qk_cat = torch.cat(scores_qk_levels, dim=1).transpose(1, 2)
        scores_kq_cat = torch.cat(scores_kq_levels, dim=1).transpose(1, 2)

        return scores_qk_cat, scores_kq_cat, xq_cat, xk_cat, xv_cat


class LighthouseCP(nn.Module):
    def __init__(
        self,
        attention,
        scorer: DialatedLighthouseNormScorerCP,
        num_levels: int = 3,
        pooling_factor: int = 4,
        topk: int = 1024,
        head_dim: int = 128,
        rope_theta: float = 10000.0,
    ):
        super().__init__()
        self.scorer = scorer
        self.num_levels = num_levels
        self.pooling_factor = pooling_factor
        self.topk = topk
        self.head_dim = head_dim
        self.rope_theta = rope_theta
        self.attention = attention
        self._rank = 0
        self._world_size = 1
        self._cp_group = None
        self._pos_table = None
        self._pos_len = None

    def set_cp_info(self, rank: int, world_size: int, cp_group):
        self._rank = rank
        self._world_size = world_size
        self._cp_group = cp_group

    def _pos_table_for(self, seqlen, device):
        if self._pos_table is None or self._pos_len != seqlen:
            self._pos_table = pyramid_last_positions(seqlen, self.num_levels, self.pooling_factor, device)
            self._pos_len = seqlen
        return self._pos_table

    def forward(self, xq, xk, xv):
        _, s, _, d = xq.shape
        half = s // 2
        topk_per_load = self.topk // (self._world_size * 2) # For a global topk we need to get topk in this shard and '*2' coz of head tail load balance

        scores_qk_h, scores_kq_h, xq_h, xk_h, xv_h = self.scorer(
          xq[:, :half], xk[:, :half], xv[:, :half]
        )
        scores_qk_t, scores_kq_t, xq_t, xk_t, xv_t = self.scorer(
            xq[:, half:], xk[:, half:], xv[:, half:]
        )

        with torch.no_grad():
          indices_h = LighthouseSelectionQkKqChunkedTopkNvrtc.forward(
              scores_qk_h.to(torch.float32), scores_kq_h.to(torch.float32),
              self.num_levels, self.pooling_factor, topk_per_load, half,
          )
          indices_t = LighthouseSelectionQkKqChunkedTopkNvrtc.forward(
              scores_qk_t.to(torch.float32), scores_kq_t.to(torch.float32),
              self.num_levels, self.pooling_factor, topk_per_load, half,
          )

        pos_table = self._pos_table_for(half, xq.device)
        W, r = self._world_size, self._rank
        rope_h = pos_table[indices_h] + r * half
        rope_t = pos_table[indices_t] + (2 * W - 1 - r) * half
        _assert_rope_monotonic(rope_h, "cp-head")
        _assert_rope_monotonic(rope_t, "cp-tail")

        sel_xq = torch.cat([
            apply_rope_at(torch.gather(xq_h, 2, indices_h.unsqueeze(-1).expand(-1,-1,-1,d)), rope_h, self.rope_theta),
            apply_rope_at(torch.gather(xq_t, 2, indices_t.unsqueeze(-1).expand(-1,-1,-1,d)), rope_t, self.rope_theta),
        ], dim=2)
        sel_xk = torch.cat([
            apply_rope_at(torch.gather(xk_h, 2, indices_h.unsqueeze(-1).expand(-1,-1,-1,d)), rope_h, self.rope_theta),
            apply_rope_at(torch.gather(xk_t, 2, indices_t.unsqueeze(-1).expand(-1,-1,-1,d)), rope_t, self.rope_theta),
        ], dim=2)
        sel_xv = torch.cat([
            torch.gather(xv_h, 2, indices_h.unsqueeze(-1).expand(-1,-1,-1,d)),
            torch.gather(xv_t, 2, indices_t.unsqueeze(-1).expand(-1,-1,-1,d)),
        ], dim=2)

        attn_out = self.attention(sel_xq, sel_xk, sel_xv)

        attn_h, attn_t = attn_out.chunk(2, dim=2)
        rank_ids_h = torch.zeros_like(indices_h)
        rank_ids_t = torch.zeros_like(indices_t)

        out_h = scatter_to_base_sequence(attn_h, indices_h, rank_ids_h, half, half,
                                            self.num_levels, self.pooling_factor)
        out_t = scatter_to_base_sequence(attn_t, indices_t, rank_ids_t, half, half,
                                            self.num_levels, self.pooling_factor)
        return torch.cat([out_h, out_t], dim=2)
