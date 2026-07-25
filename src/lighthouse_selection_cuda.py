import torch
import math
from functools import lru_cache

_HEADER_CODE = ""


def _make_kernel_source(pooling_factor: int, topk_per_tile: int = 128) -> str:
    return f"""
#define WARP_SIZE 32

__device__ __forceinline__ unsigned int float_to_ordered_uint(float f) {{
  unsigned int u = __float_as_uint(f);
  return (f >= 0.0f) ? (u ^ 0x80000000u) : (~u);
}}

__device__ __forceinline__ unsigned int encode_key(int level_local_idx, int c_l) {{
  return float_to_ordered_uint(__int2float_rn(c_l * level_local_idx));
}}

__device__ __forceinline__ unsigned int encode_key_selected(
    int level_local_idx, int c_l, int level) {{
  float ordered = __int2float_rn(c_l * (level_local_idx + 1))
                  - exp2f(1.0f - (float)level);
  return float_to_ordered_uint(ordered);
}}

__device__ __forceinline__ void write_encoded(
    unsigned int* keys, int* values, int offset,
    unsigned int key, int actual_idx) {{
  keys[offset]   = key;
  values[offset] = actual_idx;
}}

constexpr int POOLING_FACTOR  = {pooling_factor};
constexpr int TOPK             = {topk_per_tile};
constexpr int TOPK_HALF        = TOPK / 2;
constexpr int BATCH            = TOPK_HALF;
constexpr int SORT_SIZE        = TOPK;
constexpr int WARMUP_BATCHES   = TOPK / BATCH;
// Compile-time log2 of pooling_factor (power-of-2 guaranteed by host assert)
constexpr int SHIFT            = {int(math.log2(pooling_factor))};

__device__ __forceinline__ void bitonic_sort_descending(
    float* s_scores, int* s_indices) {{
  const int tid = threadIdx.x;

  // ----- warp-local phase (k=2..32): pure register shuffles, no smem traffic -----
  {{
    float my_score = s_scores[tid];
    int   my_index = s_indices[tid];

#pragma unroll
    for (int k = 2; k <= 32; k <<= 1) {{
#pragma unroll
      for (int j = k >> 1; j > 0; j >>= 1) {{
        int   ixj   = tid ^ j;
        bool  desc  = ((tid & k) == 0);
        float o_score = __shfl_xor_sync(0xffffffff, my_score, j);
        int   o_index = __shfl_xor_sync(0xffffffff, my_index, j);
        bool swap = (tid < ixj)
            ? (desc ? (my_score < o_score) : (my_score > o_score))
            : (desc ? (my_score > o_score) : (my_score < o_score));
        if (swap) {{ my_score = o_score; my_index = o_index; }}
      }}
    }}

    s_scores[tid]  = my_score;
    s_indices[tid] = my_index;
    __syncthreads();
  }}

  // ----- cross-warp phase (k=64..SORT_SIZE) -----
#pragma unroll
  for (int k = 64; k <= SORT_SIZE; k <<= 1) {{
#pragma unroll
    for (int j = k >> 1; j > 16; j >>= 1) {{
      int ixj = tid ^ j;
      if (ixj > tid) {{
        bool  desc = ((tid & k) == 0);
        float a = s_scores[tid],  b = s_scores[ixj];
        bool  swap = desc ? (a < b) : (a > b);
        if (swap) {{
          s_scores[tid]    = b;  s_scores[ixj]    = a;
          int ti           = s_indices[tid];
          s_indices[tid]   = s_indices[ixj];
          s_indices[ixj]   = ti;
        }}
      }}
      __syncthreads();
    }}

    // last 16 steps stay in registers via shuffles
    {{
      float my_score = s_scores[tid];
      int   my_index = s_indices[tid];

#pragma unroll
      for (int j = 16; j > 0; j >>= 1) {{
        int   ixj   = tid ^ j;
        bool  desc  = ((tid & k) == 0);
        float o_score = __shfl_xor_sync(0xffffffff, my_score, j);
        int   o_index = __shfl_xor_sync(0xffffffff, my_index, j);
        bool swap = (tid < ixj)
            ? (desc ? (my_score < o_score) : (my_score > o_score))
            : (desc ? (my_score > o_score) : (my_score < o_score));
        if (swap) {{ my_score = o_score; my_index = o_index; }}
      }}

      s_scores[tid]  = my_score;
      s_indices[tid] = my_index;
      __syncthreads();
    }}
  }}
}}

extern "C"
__global__ void lighthouse_selection_chunked_topk_kernel(
    const float* __restrict__ scores_qk,
    const float* __restrict__ scores_kq,
    unsigned int* __restrict__ keys_out,
    int*          __restrict__ values_out,
    const int stride_sqk_B, const int stride_sqk_H, const int stride_sqk_N,
    const int stride_skq_B, const int stride_skq_H, const int stride_skq_N,
    const int stride_idx_B, const int stride_idx_H, const int stride_idx_S,
    const int num_levels,
    const int pooling_factor,
    const int KVH,
    const int N,
    const int N_CHUNK_COARSEST,
    const int NUM_SEL_IDX_CHUNK)
{{
  const int tid = threadIdx.x;

  const long long index_BH  = (long long)blockIdx.x;
  const long long index_B   = index_BH / KVH;
  const long long index_KVH = index_BH % KVH;
  const long long index_seq = (long long)blockIdx.y;

  const float* sq = scores_qk + index_B * stride_sqk_B + index_KVH * stride_sqk_H;
  const float* sk = scores_kq + index_B * stride_skq_B + index_KVH * stride_skq_H;

  int bh_off = (int)(index_B * stride_idx_B + index_KVH * stride_idx_H);
  unsigned int* key_base = keys_out   + bh_off;
  int*          val_base = values_out + bh_off;
  int out_off   = (int)(index_seq * NUM_SEL_IDX_CHUNK);

  int tile_start = (int)(index_seq * N_CHUNK_COARSEST);
  int tile_end   = tile_start + N_CHUNK_COARSEST;

  // Shared memory layout (all 4-byte elements, no padding needed):
  //   [s_topk_scores_qk | s_topk_indices_qk | s_topk_scores_kq | s_topk_indices_kq]
  //   [s_parent]
  //   [s_sort_scores | s_sort_indices]
  extern __shared__ char smem_raw[];
  float* s_topk_scores_qk  = (float*)smem_raw;
  int*   s_topk_indices_qk = (int*  )(s_topk_scores_qk  + TOPK_HALF);
  float* s_topk_scores_kq  = (float*)(s_topk_indices_qk + TOPK_HALF);
  int*   s_topk_indices_kq = (int*  )(s_topk_scores_kq  + TOPK_HALF);
  int*   s_parent          = (int*  )(s_topk_indices_kq + TOPK_HALF);
  float* s_sort_scores     = (float*)(s_parent          + TOPK);
  int*   s_sort_indices    = (int*  )(s_sort_scores     + SORT_SIZE);

  if (tid < TOPK) s_parent[tid] = -1;
  __syncthreads();

  for (int level = num_levels; level >= 1; level--) {{
    // c_l = pooling_factor^(level-1), computed with integer shifts (pf is power-of-2)
    const int c_l = 1 << (SHIFT * (level - 1));

    const int K_end = N / c_l - 1;

    // level_offset: number of pooled tokens contributed by coarser levels
    int level_offset;
    if (level <= 1) {{
      level_offset = 0;
    }} else {{
      // pf_pow = pf^(2-level);  use integer shift when exponent >= 0, expf otherwise
      float pf_pow = expf(logf((float)pooling_factor) * (2.0f - (float)level));
      level_offset = (int)((float)N * ((float)pooling_factor - pf_pow)
                           / ((float)pooling_factor - 1.0f));
    }}

    int  num_K_indices;
    bool use_parents;
    if (level == num_levels) {{
      num_K_indices = tile_end - tile_start;
      use_parents   = false;
    }} else {{
      num_K_indices = POOLING_FACTOR * TOPK;
      use_parents   = true;
    }}

    const int K_end_abs = (K_end >= 0) ? (K_end + level_offset) : -1;

    // ------------------------------------------------------------------ level > 1
    if (level > 1) {{
      if (tid < TOPK_HALF) {{
        s_topk_scores_qk[tid]  = -1e6f;
        s_topk_indices_qk[tid] = -1;
        s_topk_scores_kq[tid]  = -1e6f;
        s_topk_indices_kq[tid] = -1;
      }}
      __syncthreads();

      const int total_batches = (num_K_indices + BATCH - 1) / BATCH;

      for (int batch = 0; batch < total_batches; batch++) {{
        const int kv_start = batch * BATCH;

        // --- Load: upper half brings new candidates, lower half holds current topk ---
        if (tid < TOPK_HALF) {{
          s_sort_scores[tid]  = s_topk_scores_qk[tid];
          s_sort_indices[tid] = s_topk_indices_qk[tid];
        }} else {{
          const int bi      = tid - TOPK_HALF;
          const int abs_col = kv_start + bi;
          int K_idx = -1;

          if (abs_col < num_K_indices) {{
            if (use_parents) {{
              const int pidx = abs_col >> SHIFT;
              const int coff = abs_col &  (POOLING_FACTOR - 1);
              const int pval = (pidx < TOPK) ? s_parent[pidx] : -1;
              if (pval >= 0) {{
                K_idx = pval * POOLING_FACTOR + coff;
                if (K_idx > K_end || K_idx < 0) K_idx = -1;
              }}
            }} else {{
              K_idx = tile_start + abs_col;
            }}
          }}

          const int  K_abs  = (K_idx >= 0) ? (K_idx + level_offset) : -1;
          const bool valid  = (K_abs >= 0) && (K_abs <= K_end_abs) && (K_end >= 0);
          s_sort_scores[tid]  = valid ? __ldg(&sq[K_abs * stride_sqk_N]) : -1e6f;
          s_sort_indices[tid] = K_idx;
        }}
        __syncthreads(); // all writes to s_sort_* visible before sort reads them

        bitonic_sort_descending(s_sort_scores, s_sort_indices);

        if (tid < TOPK_HALF) {{
          s_topk_scores_qk[tid]  = s_sort_scores[tid];
          s_topk_indices_qk[tid] = s_sort_indices[tid];
        }}

        // Save rejected QK indices (upper half) before overwriting s_sort_*
        const int rej_qk_idx = (tid >= TOPK_HALF) ? s_sort_indices[tid] : -1;
        __syncthreads();

        // --- Second sort: re-rank rejected QK candidates by KQ score ---
        if (tid < TOPK_HALF) {{
          s_sort_scores[tid]  = s_topk_scores_kq[tid];
          s_sort_indices[tid] = s_topk_indices_kq[tid];
        }} else {{
          const int  rej_abs  = (rej_qk_idx >= 0) ? (rej_qk_idx + level_offset) : -1;
          const bool kq_valid = (rej_abs >= 0) && (rej_abs <= K_end_abs) && (K_end >= 0);
          s_sort_scores[tid]  = kq_valid ? __ldg(&sk[rej_abs * stride_skq_N]) : -1e6f;
          s_sort_indices[tid] = rej_qk_idx;
        }}
        __syncthreads(); // same reason as above

        bitonic_sort_descending(s_sort_scores, s_sort_indices);

        if (tid < TOPK_HALF) {{
          s_topk_scores_kq[tid]  = s_sort_scores[tid];
          s_topk_indices_kq[tid] = s_sort_indices[tid];
        }}

        // --- Drain: emit definitely-rejected indices once pipeline is warm ---
        if (batch >= WARMUP_BATCHES) {{
          if (tid >= TOPK_HALF) {{
            const int  dr_idx   = s_sort_indices[tid];
            const int  dr_abs   = (dr_idx >= 0) ? (dr_idx + level_offset) : -1;
            const bool dr_valid = (dr_abs >= 0) && (dr_abs <= K_end_abs) && (K_end >= 0);
            if (dr_valid) {{
              write_encoded(key_base, val_base,
                  (out_off + (tid - TOPK_HALF)) * stride_idx_S,
                  encode_key(dr_idx, c_l), dr_abs);
            }}
          }}
          out_off += BATCH;
        }}

        // --- On the last batch, flush the surviving topk ---
        if (batch == total_batches - 1) {{
          __syncthreads();
          const int topk_idx = (tid < TOPK_HALF)
              ? s_topk_indices_qk[tid]
              : s_topk_indices_kq[tid - TOPK_HALF];

          if (topk_idx >= 0) {{
            const int topk_abs = topk_idx + level_offset;
            write_encoded(key_base, val_base,
                (out_off + tid) * stride_idx_S,
                encode_key_selected(topk_idx, c_l, level), topk_abs);
          }}
          out_off += TOPK;
        }}
        __syncthreads();
      }}

    // ------------------------------------------------------------------ level == 1
    }} else {{
      const int total_batches_l1 = (num_K_indices + TOPK - 1) / TOPK;
      for (int batch = 0; batch < total_batches_l1; batch++) {{
        const int kv_start = batch * TOPK;
        const int abs_col  = kv_start + tid;
        int K_idx = -1;

        if (abs_col < num_K_indices) {{
          const int pidx = abs_col >> SHIFT;
          const int coff = abs_col &  (POOLING_FACTOR - 1);
          const int pval = (pidx < TOPK) ? s_parent[pidx] : -1;
          if (pval >= 0) {{
            K_idx = pval * POOLING_FACTOR + coff;
            if (K_idx > K_end || K_idx < 0) K_idx = -1;
          }}
        }}

        const int  K_abs  = (K_idx >= 0) ? (K_idx + level_offset) : -1;
        const bool valid  = (K_abs >= 0) && (K_abs <= K_end_abs) && (K_end >= 0);
        if (valid) {{
          write_encoded(key_base, val_base,
              (out_off + tid) * stride_idx_S,
              encode_key(K_idx, c_l), K_abs);
        }}

        const int batch_count = min(TOPK, num_K_indices - kv_start);
        out_off += batch_count;
        __syncthreads();
      }}
    }}

    // Update parent pointers for the next (finer) level
    if (tid < TOPK_HALF)
      s_parent[tid] = s_topk_indices_qk[tid];
    else if (tid < TOPK)
      s_parent[tid] = s_topk_indices_kq[tid - TOPK_HALF];
    __syncthreads();
  }}
}}
"""


_compiled_kernels: dict = {}


def _get_kernel(pooling_factor: int, topk_per_tile: int = 128):
    key = (pooling_factor, topk_per_tile)
    if key not in _compiled_kernels:
        kernel_source = _HEADER_CODE + "\n" + _make_kernel_source(pooling_factor, topk_per_tile)
        _compiled_kernels[key] = torch.cuda._compile_kernel(
            kernel_source,
            "lighthouse_selection_chunked_topk_kernel",
            # --use_fast_math removed: approximate float ops (fma, rcp) can silently
            # change which indices are selected.  -O3 gives codegen wins without
            # altering floating-point semantics.
            nvcc_options=["--std=c++17", "-O3"],
        )
    return _compiled_kernels[key]


@lru_cache(maxsize=100, typed=False)
def compute_output_indices_selection(
    N: int, num_levels: int, pooling_factor: int, topk: int
) -> int:
    total = 0
    for level in range(num_levels, 0, -1):
        c_l = pooling_factor ** (level - 1)
        if level == num_levels:
            total += N // c_l
        else:
            total += pooling_factor * topk
    return total


class LighthouseSelectionQkKqChunkedTopkNvrtc:
    @torch.no_grad()
    @staticmethod
    def forward(
        scores_qk: torch.Tensor,
        scores_kq: torch.Tensor,
        num_levels: int,
        pooling_factor: int,
        topk: int,
        actual_seq_len: int,
    ) -> torch.Tensor:
        assert scores_qk.shape == scores_kq.shape
        assert scores_qk.is_cuda and scores_kq.is_cuda
        assert scores_qk.dtype == torch.float32
        assert topk % 128 == 0, "topk must be a multiple of 128"
        assert (pooling_factor & (pooling_factor - 1)) == 0, "pooling_factor must be a power of 2"
        assert topk // 2 >= pooling_factor

        scores_qk = scores_qk.contiguous()
        scores_kq = scores_kq.contiguous()

        B, KVH, _ = scores_qk.shape
        pf = pooling_factor

        TOPK_PER_TILE = 128
        NUM_TILES = topk // TOPK_PER_TILE
        assert NUM_TILES >= 1, "topk must be >= 128"

        coarsest_len = actual_seq_len
        for _ in range(num_levels - 1):
            coarsest_len //= pf
        N_CHUNK_COARSEST = coarsest_len // NUM_TILES
        assert N_CHUNK_COARSEST >= TOPK_PER_TILE, (
            f"N_CHUNK_COARSEST={N_CHUNK_COARSEST} must be >= TOPK_PER_TILE={TOPK_PER_TILE}"
        )

        N_CHUNK = actual_seq_len // NUM_TILES
        sel_idx       = compute_output_indices_selection(actual_seq_len, num_levels, pf, topk)
        sel_idx_chunk = compute_output_indices_selection(N_CHUNK,        num_levels, pf, TOPK_PER_TILE)

        sort_keys = torch.full((B, KVH, sel_idx), -1, dtype=torch.int32, device=scores_qk.device)
        values    = torch.full((B, KVH, sel_idx), -1, dtype=torch.int32, device=scores_qk.device)

        kernel = _get_kernel(pf, TOPK_PER_TILE)

        smem_bytes = (
            TOPK_PER_TILE // 2 * 4 * 4  # 4 × TOPK_HALF float/int arrays (qk + kq scores & indices)
            + TOPK_PER_TILE * 4          # parent indices
            + TOPK_PER_TILE * 4 * 2      # sort workspace (scores + indices)
        )

        kernel(
            grid=(B * KVH, NUM_TILES, 1),
            block=(TOPK_PER_TILE, 1, 1),
            args=[
                scores_qk, scores_kq,
                sort_keys, values,
                scores_qk.stride(0), scores_qk.stride(1), scores_qk.stride(2),
                scores_kq.stride(0), scores_kq.stride(1), scores_kq.stride(2),
                sort_keys.stride(0), sort_keys.stride(1), sort_keys.stride(2),
                num_levels, pf, KVH, actual_seq_len,
                N_CHUNK_COARSEST, sel_idx_chunk,
            ],
            shared_mem=smem_bytes,
        )

        _, sort_order = torch.sort(sort_keys, dim=-1, stable=True)
        result = torch.gather(values, dim=-1, index=sort_order)
        return result