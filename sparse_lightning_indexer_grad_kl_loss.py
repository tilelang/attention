import torch
from dataclasses import dataclass
from typing import Optional, Tuple

import tilelang
import tilelang.language as T

tilelang.disable_cache()

# 固定 tile 与流水参数。
BLOCK_N = 128
BLOCK_K = 128
VECTOR_BASEN = 256
GATHER_CHUNK_SIZE = 16
V2_SCATTER_BATCH = 16
V2_TARGET_BATCHES_PER_AIV = 16
# 一个物理核负责一个完整的（B，S1 分块）任务。
# V0 发布已 gather 的 Key 和 KeyIndex 数据流，C1/V1/C2/V2 消费 task ID % 2 对应的工作区槽位。
S2_WORKSPACE = 1024
N_SPLIT = 128
K_SPLIT = 128
TASK_PIPELINE_DEPTH = 2
V0_TO_C1_P_FLAGS = (0, 1)
V0_TO_C1_SY_FLAGS = (2, 3)
C1_TO_V1_P_FLAGS = (4, 5)
C1_TO_V1_SY_FLAGS = (6, 7)
V1_TO_C2_FLAGS = (8, 9)
# TileLang-Ascend 无法生成官方 A3 kernel 使用的模式 1 AIV 对等待。
# 模式 2 在 AIC 处汇聚两个 AIV 的同一组 flag，实现等价等待。
V1_PSY_TO_AIC_FLAGS = (10, 11)
V1_PSY_RELAY_FLAGS = (12, 13)
DQUERY = 512
DQUERY_INDEX = 128

# JIT kernel、workspace 和辅助张量均按 specialization 缓存。
_kernel_cache = {}
_workspace_cache = {}
_full_valid_aux_cache = {}
_tail_aux_cache = {}
_v0_zero_row_cache = {}


def _clear_caches():
    global _kernel_cache, _workspace_cache, _full_valid_aux_cache, _tail_aux_cache
    global _v0_zero_row_cache
    _kernel_cache.clear()
    _workspace_cache.clear()
    _full_valid_aux_cache.clear()
    _tail_aux_cache.clear()
    _v0_zero_row_cache.clear()

pass_configs = {
    # 关闭自动同步后，所有 SetFlag/WaitFlag/PipeBarrier 均由 kernel 显式管理。
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: False,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _expected_rank(layout_query: str) -> int:
    return 3 if layout_query == "TND" else 4


def _add_batch_dim(*tensors: torch.Tensor) -> tuple[torch.Tensor, ...]:
    return tuple(tensor.unsqueeze(0) for tensor in tensors)


def _length_tensor(
    lengths: Optional[torch.Tensor], default_length: int,
) -> torch.Tensor:
    if lengths is None:
        return torch.tensor([default_length], dtype=torch.int32)
    if isinstance(lengths, (list, tuple)):
        return torch.tensor(lengths, dtype=torch.int32)
    return lengths.to(dtype=torch.int32)


def _get_aicore_count() -> int:
    """读取当前 NPU 的物理 Cube core 数量。"""
    try:
        if torch.npu.is_available() and torch.npu.device_count() > 0:
            props = torch.npu.get_device_properties(torch.npu.current_device())
            if hasattr(props, "cube_core_num"):
                core_count = int(props.cube_core_num)
                if core_count > 0:
                    return core_count
    except Exception as exc:
        raise RuntimeError("Failed to query the NPU cube-core count") from exc
    raise RuntimeError(
        "NPU device properties do not expose a valid cube_core_num"
    )

@dataclass
class SparseLightningIndexerGradKLLossTilingData:
    B: int
    S1: int
    S2: int
    N1: int
    N2: int
    G: int
    Nidx1: int
    Nidx2: int
    DQuery: int
    DQueryIndex: int
    K: int
    sparse_mode: int = 3


@dataclass(frozen=True)
class TaskSchedule:
    """完整（B，S1-tile）任务所有权的运行时划分。"""
    task_count: int
    launch_core_num: int
    tasks_per_core: int


def _make_task_schedule(
    B: int, S1: int, S2: int, K: int, hardware_core_num: int,
) -> TaskSchedule:
    task_count = B * S1
    hardware_core_num = max(hardware_core_num, 1)
    scatter_batches = ceil_div(K, V2_SCATTER_BATCH)
    finalize_batches = ceil_div(B * S2, V2_SCATTER_BATCH)
    v2_batches = max(scatter_batches, finalize_batches)
    v2_core_need = ceil_div(
        v2_batches, 2 * V2_TARGET_BATCHES_PER_AIV,
    )
    active_cores = min(
        max(task_count, v2_core_need, 1), hardware_core_num,
    )
    return TaskSchedule(
        task_count=task_count,
        launch_core_num=active_cores,
        tasks_per_core=ceil_div(task_count, active_cores),
    )


def check_param_valid(
    query: torch.Tensor, key: torch.Tensor, query_index: torch.Tensor, key_index: torch.Tensor,
    weights: torch.Tensor, sparse_indices: torch.Tensor, softmax_max: torch.Tensor, softmax_sum: torch.Tensor, *,
    sparse_mode: int = 3, layout_query: str = "BSND",
) -> SparseLightningIndexerGradKLLossTilingData:
    expected_rank = _expected_rank(layout_query)
    layout_name = "TND" if expected_rank == 3 else "BSND"
    for name, tensor in (
        ("query", query), ("key", key),
        ("queryIndex", query_index), ("keyIndex", key_index),
    ):
        assert tensor.dim() == expected_rank, (
            f"{name} dim must be {expected_rank} ({layout_name}), got {tensor.dim()}"
        )
    assert query.dtype in (torch.float16, torch.bfloat16), f"query dtype must be float16/bfloat16, got {query.dtype}"
    assert key.dtype == query.dtype, f"key dtype must match query, got {key.dtype}"
    assert query_index.dtype == query.dtype, f"queryIndex dtype must match query, got {query_index.dtype}"
    assert key_index.dtype == query.dtype, f"keyIndex dtype must match query, got {key_index.dtype}"
    assert weights.dtype in (torch.float16, torch.bfloat16, torch.float32), f"weights dtype must be float16/bfloat16/float32, got {weights.dtype}"
    if weights.dtype != torch.float32:
        assert weights.dtype == query.dtype, f"weights dtype must match query when not float32, got {weights.dtype}"
    assert sparse_mode == 3, f"sparse_mode must be 3, got {sparse_mode}"
    if layout_query == "TND":
        S1, N1, DQuery = query.shape
        S2, N2, _ = key.shape
        _, Nidx1, DQueryIndex = query_index.shape
        _, Nidx2, _ = key_index.shape
        B = 1
    else:
        B, S1, N1, DQuery = query.shape
        _, S2, N2, _ = key.shape
        _, _, Nidx1, DQueryIndex = query_index.shape
        _, _, Nidx2, _ = key_index.shape
    K = sparse_indices.shape[-1]
    assert N2 == 1, f"N2 must be 1 (MQA), got {N2}"
    G = N1 // N2
    assert N1 == N2 * G, f"N1({N1}) must be divisible by N2({N2})"
    assert DQuery == DQUERY, f"DQuery must be {DQUERY}, got {DQuery}"
    assert DQueryIndex == DQUERY_INDEX, f"DQueryIndex must be {DQUERY_INDEX}, got {DQueryIndex}"
    assert N1 in (32, 64, 128), f"N1 must be 32/64/128, got {N1}"
    assert Nidx1 in (8, 16, 32, 64), f"Nidx1 must be 8/16/32/64, got {Nidx1}"
    assert K % 1024 == 0 and 1024 <= K <= 8192, f"K must be in [1024,8192] and multiple of 1024, got {K}"
    return SparseLightningIndexerGradKLLossTilingData(
        B=B, S1=S1, S2=S2, N1=N1, N2=N2, G=G, Nidx1=Nidx1, Nidx2=Nidx2, DQuery=DQuery, DQueryIndex=DQueryIndex,
        K=K, sparse_mode=sparse_mode,
    )

# ==============================================================================
# 单内核的 CV 分离流水：V0 gather，C1 两次 matmul，V1 backward，C2 求梯度，
# V2 转换 dQuery 并 scatter dKey。每个物理核负责一个完整的（B，S1）任务。
# ==============================================================================
@tilelang.jit(
    out_idx=[-4, -3, -2, -1],
    # 工作区由主机封装层缓存并显式传入，不能登记为 workspace_idx。
    workspace_idx=[], pass_configs=pass_configs,
)
def tilelang_grad_kl_loss_single_kernel(
    B, S1, S2, N1, N2, G, Nidx1, Nidx2, DQuery, DQueryIndex, K, launch_core_num,
    d_rope, input_dtype, calc_dtype, scale_value, block_n=BLOCK_N, full_valid=False, k_scatter_exec=None,
    source_s2=None, sparse_k_stride=None,
):
    source_s2 = S2 if source_s2 is None else source_s2
    sparse_k_stride = K if sparse_k_stride is None else sparse_k_stride
    k_scatter_exec = K if k_scatter_exec is None else k_scatter_exec
    assert 0 < source_s2 <= S2
    assert sparse_k_stride >= K
    assert 0 < k_scatter_exec <= K
    assert k_scatter_exec % V2_SCATTER_BATCH == 0
    n_block = block_n
    k_block = BLOCK_K
    p_rows_per_chunk = min(N1, 64)
    p_row_chunk_count = ceil_div(N1, p_rows_per_chunk)
    p_cols_per_chunk = min(n_block, 64)
    p_col_chunk_count = ceil_div(n_block, p_cols_per_chunk)
    # V1 反向计算保持在已经验证精度正确的 32x64 物理分块内。
    bw_col_block = min(n_block, 64)
    assert n_block % bw_col_block == 0
    bw_col_count = n_block // bw_col_block
    n_num = K // n_block
    d_num_query = DQuery // k_block
    d_num_rope = ceil_div(d_rope, k_block)
    d_num_query_index = DQueryIndex // k_block
    c2_dk_k = max(Nidx1, 16)
    rows_per_aiv = Nidx1 // 2
    task_count = B * S1
    launch_core_num = max(launch_core_num, 1)
    tasks_per_core = ceil_div(task_count, launch_core_num)
    task_pipeline_lag = TASK_PIPELINE_DEPTH - 1
    total_aiv_num = launch_core_num * 2
    # 16 KiB scatter UB 在 ping/pong 间均分，每个槽位为 16 行 x 128 个 FP32。
    v2_scatter_batch = V2_SCATTER_BATCH
    # V2 只搬运主机侧静态确定的有效 scatter 宽度。
    v2_total_batches = k_scatter_exec // v2_scatter_batch
    v2_base_batches_per_core = v2_total_batches // launch_core_num
    v2_tail_batch_cores = v2_total_batches % launch_core_num
    v2_k_padded = K
    # 宏收集会校验条件两侧的 BufferRegion，因此保持可合法索引的二维形状。
    k_real_size_shape = (B, S1)
    valid_k_mask_shape = (B, S1, K)
    dkey_output_rows = B * S2
    assert dkey_output_rows % 32 == 0, (
        "The padded dKey output must preserve 32-row DMA alignment"
    )
    dkey_clear_batch = 32
    dkey_clear_total_batches = dkey_output_rows // dkey_clear_batch
    dkey_finalize_batch = 16
    dkey_finalize_total_batches = dkey_output_rows // dkey_finalize_batch
    loss_reduce_chunk = 128
    loss_partial_stride = 8
    loss_partial_span = task_count * 2 * loss_partial_stride
    loss_partial_size = ceil_div(
        loss_partial_span, loss_reduce_chunk
    ) * loss_reduce_chunk
    loss_full_chunk_count = loss_partial_size // loss_reduce_chunk
    @T.macro
    def vector0_gather_task(
        SparseIndices, V0ZeroRow, Key, KeyRope, KeyIndex, GatherKey_ws, GatherKeyRope_ws,
        GatherKeyIndex_ws, bz_task, s1_begin_task, s1_end_task, task_slot, vector_id,
    ):
        """按照与 MergeKv 相同的 subblock 所有权 gather 一个任务。
        两个 AIV 都 gather 三条数据流。每个 AIV 负责当前 1024 行 K tile
        的一个连续半区，随后共同发布由 C1 消费的 mode-2 readiness flag。
        """
        with T.Scope("V"):
            v0_key_ub = T.alloc_ub((2, GATHER_CHUNK_SIZE, DQuery), input_dtype)
            v0_ki_ub = T.alloc_ub((2, GATHER_CHUNK_SIZE, DQueryIndex), input_dtype)
            v0_rope_ub = T.alloc_ub((2, GATHER_CHUNK_SIZE, max(d_rope, 1)), input_dtype)
            T.set_flag("mte3", "mte2", 0)
            T.set_flag("mte3", "mte2", 1)
            for s1_id in T.serial(s1_begin_task, s1_end_task):
                for k_outer in T.serial(ceil_div(K, S2_WORKSPACE)):
                    outer_start = k_outer * S2_WORKSPACE
                    rows_per_aiv = S2_WORKSPACE // 2
                    aiv_start = outer_start + vector_id * rows_per_aiv
                    gather_chunks = rows_per_aiv // GATHER_CHUNK_SIZE
                    T.wait_flag("mte3", "mte2", 0)
                    for lane in T.serial(GATHER_CHUNK_SIZE):
                        k_id = aiv_start + lane
                        sparse_idx = SparseIndices[bz_task, s1_id, 0, k_id]
                        if sparse_idx >= 0:
                            T.copy(Key[bz_task, sparse_idx, 0, 0:DQuery], v0_key_ub[0, lane, 0:DQuery])
                            T.copy(KeyIndex[bz_task, sparse_idx, 0, 0:DQueryIndex], v0_ki_ub[0, lane, 0:DQueryIndex])
                            if d_rope > 0:
                                T.copy(KeyRope[bz_task, sparse_idx, 0, 0:d_rope], v0_rope_ub[0, lane, 0:d_rope])
                        if sparse_idx < 0:
                            T.copy(V0ZeroRow[0:DQuery], v0_key_ub[0, lane, 0:DQuery])
                            T.copy(V0ZeroRow[0:DQueryIndex], v0_ki_ub[0, lane, 0:DQueryIndex])
                            if d_rope > 0:
                                T.copy(V0ZeroRow[0:d_rope], v0_rope_ub[0, lane, 0:d_rope])
                    T.set_flag("mte2", "mte3", 0)
                    for chunk_id in T.serial(gather_chunks):
                        cur = chunk_id % 2
                        nxt = (chunk_id + 1) % 2
                        if chunk_id + 1 < gather_chunks:
                            next_start = aiv_start + (chunk_id + 1) * GATHER_CHUNK_SIZE
                            T.wait_flag("mte3", "mte2", nxt)
                            for lane in T.serial(GATHER_CHUNK_SIZE):
                                k_id = next_start + lane
                                sparse_idx = SparseIndices[bz_task, s1_id, 0, k_id]
                                if sparse_idx >= 0:
                                    T.copy(Key[bz_task, sparse_idx, 0, 0:DQuery], v0_key_ub[nxt, lane, 0:DQuery])
                                    T.copy(KeyIndex[bz_task, sparse_idx, 0, 0:DQueryIndex], v0_ki_ub[nxt, lane, 0:DQueryIndex])
                                    if d_rope > 0:
                                        T.copy(KeyRope[bz_task, sparse_idx, 0, 0:d_rope], v0_rope_ub[nxt, lane, 0:d_rope])
                                if sparse_idx < 0:
                                    T.copy(V0ZeroRow[0:DQuery], v0_key_ub[nxt, lane, 0:DQuery])
                                    T.copy(V0ZeroRow[0:DQueryIndex], v0_ki_ub[nxt, lane, 0:DQueryIndex])
                                    if d_rope > 0:
                                        T.copy(V0ZeroRow[0:d_rope], v0_rope_ub[nxt, lane, 0:d_rope])
                            T.set_flag("mte2", "mte3", nxt)
                        T.wait_flag("mte2", "mte3", cur)
                        cur_start = aiv_start + chunk_id * GATHER_CHUNK_SIZE
                        T.copy(v0_key_ub[cur, 0:GATHER_CHUNK_SIZE, 0:DQuery], GatherKey_ws[bz_task, s1_id, cur_start:cur_start + GATHER_CHUNK_SIZE, 0:DQuery])
                        T.copy(v0_ki_ub[cur, 0:GATHER_CHUNK_SIZE, 0:DQueryIndex], GatherKeyIndex_ws[bz_task, s1_id, cur_start:cur_start + GATHER_CHUNK_SIZE, 0:DQueryIndex])
                        if d_rope > 0:
                            T.copy(v0_rope_ub[cur, 0:GATHER_CHUNK_SIZE, 0:d_rope], GatherKeyRope_ws[bz_task, s1_id, cur_start:cur_start + GATHER_CHUNK_SIZE, 0:d_rope])
                        T.set_flag("mte3", "mte2", cur)
                    # 模式 2 的 AIC wait 汇聚两个 AIV 的同一 flag，避免 C1 提前消费。
                    flag_slot = (task_slot + k_outer) % 2
                    if flag_slot == 0:
                        T.set_cross_flag("MTE3", V0_TO_C1_P_FLAGS[0])
                        T.set_cross_flag("MTE3", V0_TO_C1_SY_FLAGS[0])
                    else:
                        T.set_cross_flag("MTE3", V0_TO_C1_P_FLAGS[1])
                        T.set_cross_flag("MTE3", V0_TO_C1_SY_FLAGS[1])
            T.wait_flag("mte3", "mte2", 0)
            T.wait_flag("mte3", "mte2", 1)
    @T.macro
    def cube1_task(
        Query, QueryRope, QueryIndex, GatherKey_ws, GatherKeyRope_ws, GatherKeyIndex_ws,
        bmm1Res, bmm2Res, bz_task, s1_begin_task, s1_end_task, task_slot,
    ):
        """已经验证精度正确的单槽 C1 基线。"""
        with T.Scope("C"):
            c1_query_chunk_0_l1 = T.alloc_L1((N1, K_SPLIT), input_dtype)
            c1_query_chunk_1_l1 = T.alloc_L1((N1, K_SPLIT), input_dtype)
            c1_query_chunk_2_l1 = T.alloc_L1((N1, K_SPLIT), input_dtype)
            c1_query_chunk_3_l1 = T.alloc_L1((N1, K_SPLIT), input_dtype)
            c1_query_rope_cache_l1 = T.alloc_L1((N1, max(d_rope, 1)), input_dtype)
            c1_key_l1 = T.alloc_L1((N_SPLIT, K_SPLIT), input_dtype)
            c1_q_l0a = T.alloc_L0A((N1, K_SPLIT), input_dtype)
            c1_key_l0b = T.alloc_L0B((K_SPLIT, N_SPLIT), input_dtype)
            c1_p_l0c = T.alloc_L0C((N1, N_SPLIT), calc_dtype)
            c1_qi_l1 = T.alloc_L1((Nidx1, K_SPLIT), input_dtype)
            c1_ki_l1 = T.alloc_L1((N_SPLIT, K_SPLIT), input_dtype)
            c1_qi_l0a = T.alloc_L0A((Nidx1, K_SPLIT), input_dtype)
            c1_ki_l0b = T.alloc_L0B((K_SPLIT, N_SPLIT), input_dtype)
            c1_sy_l0c = T.alloc_L0C((Nidx1, N_SPLIT), calc_dtype)
            T.set_flag("mte1", "mte2", 0)
            T.set_flag("m", "mte1", 0)
            T.set_flag("fix", "m", 0)
            T.set_flag("mte1", "mte2", 1)
            T.set_flag("m", "mte1", 1)
            T.set_flag("fix", "m", 1)
            for s1_id in T.serial(s1_begin_task, s1_end_task):
                T.copy(Query[bz_task, s1_id, 0:N1, 0:K_SPLIT], c1_query_chunk_0_l1)
                T.copy(Query[bz_task, s1_id, 0:N1, K_SPLIT:2 * K_SPLIT], c1_query_chunk_1_l1)
                T.copy(Query[bz_task, s1_id, 0:N1, 2 * K_SPLIT:3 * K_SPLIT], c1_query_chunk_2_l1)
                T.copy(Query[bz_task, s1_id, 0:N1, 3 * K_SPLIT:4 * K_SPLIT], c1_query_chunk_3_l1)
                if d_rope > 0:
                    T.copy(QueryRope[bz_task, s1_id, 0:N1, 0:d_rope], c1_query_rope_cache_l1[0:N1, 0:d_rope])
                T.set_flag("mte2", "mte1", 4)
                T.wait_flag("mte2", "mte1", 4)
                T.copy(QueryIndex[bz_task, s1_id, 0:Nidx1, 0:DQueryIndex], c1_qi_l1)
                T.set_flag("mte2", "mte1", 5)
                for k_outer in T.serial(ceil_div(K, S2_WORKSPACE)):
                    flag_slot = (task_slot + k_outer) % 2
                    if flag_slot == 0:
                        T.wait_cross_flag(V0_TO_C1_P_FLAGS[0])
                    else:
                        T.wait_cross_flag(V0_TO_C1_P_FLAGS[1])
                    for k_inner in T.serial(S2_WORKSPACE // N_SPLIT):
                        k_start = k_outer * S2_WORKSPACE + k_inner * N_SPLIT
                        T.wait_flag("mte1", "mte2", 0)
                        T.copy(GatherKey_ws[bz_task, s1_id, k_start:k_start + N_SPLIT, 0:K_SPLIT], c1_key_l1)
                        T.set_flag("mte2", "mte1", 0)
                        for d_idx in T.serial(d_num_query):
                            T.wait_flag("m", "mte1", 0)
                            T.wait_flag("mte2", "mte1", 0)
                            if d_idx == 0:
                                T.copy(c1_query_chunk_0_l1, c1_q_l0a)
                            if d_idx == 1:
                                T.copy(c1_query_chunk_1_l1, c1_q_l0a)
                            if d_idx == 2:
                                T.copy(c1_query_chunk_2_l1, c1_q_l0a)
                            if d_idx == 3:
                                T.copy(c1_query_chunk_3_l1, c1_q_l0a)
                            T.copy(c1_key_l1, c1_key_l0b, transpose=True)
                            T.set_flag("mte1", "mte2", 0)
                            T.set_flag("mte1", "m", 0)
                            if d_idx + 1 < d_num_query:
                                T.wait_flag("mte1", "mte2", 0)
                                T.copy(GatherKey_ws[bz_task, s1_id, k_start:k_start + N_SPLIT, (d_idx + 1) * K_SPLIT:(d_idx + 2) * K_SPLIT], c1_key_l1)
                                T.set_flag("mte2", "mte1", 0)
                            if d_idx + 1 >= d_num_query and d_num_rope > 0:
                                T.wait_flag("mte1", "mte2", 0)
                                T.copy(GatherKeyRope_ws[bz_task, s1_id, k_start:k_start + N_SPLIT, 0:K_SPLIT], c1_key_l1)
                                T.set_flag("mte2", "mte1", 0)
                            T.wait_flag("mte1", "m", 0)
                            T.wait_flag("fix", "m", 0)
                            T.mma(c1_q_l0a, c1_key_l0b, c1_p_l0c, init=(d_idx == 0))
                            T.set_flag("m", "mte1", 0)
                            if d_idx + 1 < d_num_query or d_num_rope > 0:
                                T.set_flag("fix", "m", 0)
                        for rope_idx in T.serial(d_num_rope):
                            T.wait_flag("m", "mte1", 0)
                            T.wait_flag("mte2", "mte1", 0)
                            T.copy(c1_query_rope_cache_l1[0:N1, rope_idx * K_SPLIT:(rope_idx + 1) * K_SPLIT], c1_q_l0a)
                            T.copy(c1_key_l1, c1_key_l0b, transpose=True)
                            T.set_flag("mte1", "mte2", 0)
                            T.set_flag("mte1", "m", 0)
                            if rope_idx + 1 < d_num_rope:
                                T.wait_flag("mte1", "mte2", 0)
                                T.copy(GatherKeyRope_ws[bz_task, s1_id, k_start:k_start + N_SPLIT, (rope_idx + 1) * K_SPLIT:(rope_idx + 2) * K_SPLIT], c1_key_l1)
                                T.set_flag("mte2", "mte1", 0)
                            T.wait_flag("mte1", "m", 0)
                            T.wait_flag("fix", "m", 0)
                            T.mma(c1_q_l0a, c1_key_l0b, c1_p_l0c, init=False)
                            T.set_flag("m", "mte1", 0)
                            if rope_idx + 1 < d_num_rope:
                                T.set_flag("fix", "m", 0)
                        T.set_flag("m", "fix", 0)
                        T.wait_flag("m", "fix", 0)
                        T.copy(c1_p_l0c, bmm1Res[bz_task, s1_id, 0:N1, k_start:k_start + N_SPLIT])
                        T.set_flag("fix", "m", 0)
                T.wait_flag("mte1", "mte2", 0)
                T.wait_flag("m", "mte1", 0)
                T.wait_flag("fix", "m", 0)
                if task_slot == 0:
                    T.set_cross_flag("FIX", C1_TO_V1_P_FLAGS[0])
                else:
                    T.set_cross_flag("FIX", C1_TO_V1_P_FLAGS[1])
                T.wait_flag("mte2", "mte1", 5)
                if d_num_query_index == 1:
                    T.wait_flag("m", "mte1", 1)
                    T.copy(c1_qi_l1, c1_qi_l0a)
                    T.set_flag("m", "mte1", 1)
                for k_outer in T.serial(ceil_div(K, S2_WORKSPACE)):
                    flag_slot = (task_slot + k_outer) % 2
                    if flag_slot == 0:
                        T.wait_cross_flag(V0_TO_C1_SY_FLAGS[0])
                    else:
                        T.wait_cross_flag(V0_TO_C1_SY_FLAGS[1])
                    for k_inner in T.serial(S2_WORKSPACE // N_SPLIT):
                        k_start = k_outer * S2_WORKSPACE + k_inner * N_SPLIT
                        for d_idx in T.serial(d_num_query_index):
                            T.wait_flag("mte1", "mte2", 1)
                            T.copy(GatherKeyIndex_ws[bz_task, s1_id, k_start:k_start + N_SPLIT, d_idx * K_SPLIT:(d_idx + 1) * K_SPLIT], c1_ki_l1)
                            T.set_flag("mte2", "mte1", 1)
                            T.wait_flag("m", "mte1", 1)
                            T.wait_flag("mte2", "mte1", 1)
                            if d_num_query_index != 1:
                                T.copy(c1_qi_l1[0:Nidx1, d_idx * K_SPLIT:(d_idx + 1) * K_SPLIT], c1_qi_l0a)
                            T.copy(c1_ki_l1, c1_ki_l0b, transpose=True)
                            T.set_flag("mte1", "mte2", 1)
                            T.set_flag("mte1", "m", 1)
                            T.wait_flag("mte1", "m", 1)
                            T.wait_flag("fix", "m", 1)
                            T.mma(c1_qi_l0a, c1_ki_l0b, c1_sy_l0c, init=(d_idx == 0))
                            T.set_flag("m", "mte1", 1)
                            if d_idx + 1 < d_num_query_index:
                                T.set_flag("fix", "m", 1)
                        T.set_flag("m", "fix", 1)
                        T.wait_flag("m", "fix", 1)
                        T.copy(c1_sy_l0c, bmm2Res[bz_task, s1_id, 0:Nidx1, k_start:k_start + N_SPLIT], enable_relu=True)
                        T.set_flag("fix", "m", 1)
                T.wait_flag("mte1", "mte2", 1)
                T.wait_flag("m", "mte1", 1)
                T.wait_flag("fix", "m", 1)
                if task_slot == 0:
                    T.set_cross_flag("FIX", C1_TO_V1_SY_FLAGS[0])
                else:
                    T.set_cross_flag("FIX", C1_TO_V1_SY_FLAGS[1])
    @T.macro
    def vector1_psy_task(
        KRealSize, ValidKMask, SoftmaxMax, SoftmaxSum, Weights, bmm1Res, bmm2Res, PReduced,
        SoftmaxI, bz_task, s1_begin_task, s1_end_task, task_slot, vector_id,
    ):
        """翻译 ProcessVector1 的前半部分。
        AIV0 负责 VectorP 并生成 PReduced，AIV1 负责 VectorSy 并生成
        SoftmaxI。两者遍历同一任务的全部 K chunk，而不是在每次 K launch
        中重复完整 K reduction。
        """
        with T.Scope("V"):
            if task_slot == 0:
                if vector_id == 0:
                    T.wait_cross_flag(C1_TO_V1_P_FLAGS[0])
                else:
                    T.wait_cross_flag(C1_TO_V1_SY_FLAGS[0])
            else:
                if vector_id == 0:
                    T.wait_cross_flag(C1_TO_V1_P_FLAGS[1])
                else:
                    T.wait_cross_flag(C1_TO_V1_SY_FLAGS[1])
            for s1_id in T.serial(s1_begin_task, s1_end_task):
                valid_k_size = K if full_valid else KRealSize[bz_task, s1_id]
                if vector_id == 0:
                    v1_p_bmm1 = T.alloc_ub((N1, n_block), calc_dtype)
                    v1_p_max = T.alloc_ub(N1, calc_dtype)
                    v1_p_sum = T.alloc_ub(N1, calc_dtype)
                    v1_p_partial = T.alloc_ub(max(N1, n_block), calc_dtype)
                    v1_p_out = T.alloc_ub(n_block, calc_dtype)
                    v1_p_tmp = T.alloc_ub((N1, 8), calc_dtype)
                    v1_p_valid = T.alloc_ub(n_block, calc_dtype)
                    v1_p_valid_mask = T.alloc_ub(n_block // 8, "uint8")
                    T.copy(SoftmaxMax[bz_task, 0, s1_id, 0:N1], v1_p_max)
                    T.copy(SoftmaxSum[bz_task, 0, s1_id, 0:N1], v1_p_sum)
                    T.set_flag("mte2", "v", 0)
                    T.set_flag("mte3", "v", 0)
                    T.set_flag("v", "mte2", 1)
                    T.wait_flag("mte2", "v", 0)
                    T.tile.abs(v1_p_sum, v1_p_sum)
                    T.tile.fill(v1_p_partial[0:N1], 1e-8)
                    T.pipe_barrier("v")
                    T.tile.max(v1_p_sum, v1_p_sum, v1_p_partial[0:N1])
                    for n_idx in T.serial(n_num):
                        n_start = n_idx * n_block
                        T.wait_flag("v", "mte2", 1)
                        T.wait_flag("mte3", "v", 0)
                        if n_start < valid_k_size:
                            T.copy(bmm1Res[bz_task, s1_id, 0:N1, n_start:n_start + n_block], v1_p_bmm1,)
                            if not full_valid and n_start + n_block > valid_k_size:
                                T.copy(ValidKMask[ bz_task, s1_id, n_idx * n_block:(n_idx + 1) * n_block, ], v1_p_valid,)
                            T.set_flag("mte2", "v", 1)
                            T.wait_flag("mte2", "v", 1)
                            T.tile.mul(v1_p_bmm1, v1_p_bmm1, scale_value)
                            T.pipe_barrier("v")
                            if not full_valid and n_start + n_block > valid_k_size:
                                T.tile.compare(
                                    v1_p_valid_mask, v1_p_valid, 0.0, "GT",
                                )
                                T.pipe_barrier("v")
                                for p_mask_row in T.serial(N1):
                                    T.tile.select(
                                        v1_p_bmm1[p_mask_row, 0:n_block], v1_p_valid_mask,
                                        v1_p_bmm1[p_mask_row, 0:n_block], -T.infinity(calc_dtype),
                                        "VSEL_TENSOR_SCALAR_MODE",
                                    )
                                T.pipe_barrier("v")
                            for p_row_chunk in T.serial(p_row_chunk_count):
                                p_row_start = p_row_chunk * p_rows_per_chunk
                                for p_col_chunk in T.serial(p_col_chunk_count):
                                    p_col_start = p_col_chunk * p_cols_per_chunk
                                    T.tile.row_expand_sub_experiment(
                                        v1_p_bmm1[
                                            p_row_start:p_row_start + p_rows_per_chunk,
                                            p_col_start:p_col_start + p_cols_per_chunk,
                                        ],
                                        v1_p_bmm1[
                                            p_row_start:p_row_start + p_rows_per_chunk,
                                            p_col_start:p_col_start + p_cols_per_chunk,
                                        ],
                                        v1_p_max[
                                            p_row_start:p_row_start + p_rows_per_chunk
                                        ],
                                        v1_p_tmp[
                                            p_row_start:p_row_start + p_rows_per_chunk, 0:8,
                                        ],
                                    )
                            T.pipe_barrier("v")
                            T.tile.exp(v1_p_bmm1, v1_p_bmm1)
                            T.pipe_barrier("v")
                            for p_row_chunk in T.serial(p_row_chunk_count):
                                p_row_start = p_row_chunk * p_rows_per_chunk
                                for p_col_chunk in T.serial(p_col_chunk_count):
                                    p_col_start = p_col_chunk * p_cols_per_chunk
                                    T.tile.row_expand_div_experiment(
                                        v1_p_bmm1[
                                            p_row_start:p_row_start + p_rows_per_chunk,
                                            p_col_start:p_col_start + p_cols_per_chunk,
                                        ],
                                        v1_p_bmm1[
                                            p_row_start:p_row_start + p_rows_per_chunk,
                                            p_col_start:p_col_start + p_cols_per_chunk,
                                        ],
                                        v1_p_sum[
                                            p_row_start:p_row_start + p_rows_per_chunk
                                        ],
                                        v1_p_tmp[
                                            p_row_start:p_row_start + p_rows_per_chunk, 0:8,
                                        ],
                                    )
                            T.pipe_barrier("v")
                            if not full_valid and n_start + n_block > valid_k_size:
                                for p_mask_row in T.serial(N1):
                                    T.tile.mul(
                                        v1_p_bmm1[p_mask_row, 0:n_block], v1_p_bmm1[p_mask_row, 0:n_block], v1_p_valid,
                                    )
                                T.pipe_barrier("v")
                            T.tile.fill(v1_p_out, 0.0)
                            T.pipe_barrier("v")
                            for p_reduce_chunk in T.serial(p_row_chunk_count):
                                p_reduce_start = p_reduce_chunk * p_rows_per_chunk
                                T.reduce_sum(
                                    v1_p_bmm1[
                                        p_reduce_start:p_reduce_start + p_rows_per_chunk, 0:n_block,
                                    ], v1_p_partial[0:n_block], 0,
                                )
                                T.pipe_barrier("v")
                                T.tile.add(
                                    v1_p_out, v1_p_out, v1_p_partial[0:n_block],
                                )
                                T.pipe_barrier("v")
                            T.tile.mul(v1_p_out, v1_p_out, 1.0 / N1)
                        else:
                            T.tile.fill(v1_p_out, 0.0)
                        T.set_flag("v", "mte2", 1)
                        T.set_flag("v", "mte3", 0)
                        T.wait_flag("v", "mte3", 0)
                        T.copy(v1_p_out, PReduced[bz_task, s1_id, n_start:n_start + n_block],)
                        T.set_flag("mte3", "v", 0)
                    T.wait_flag("v", "mte2", 1)
                    T.wait_flag("mte3", "v", 0)
                if vector_id == 1:
                    v1_sy_storage = T.alloc_ub(
                        (Nidx1 + n_num, n_block), calc_dtype
                    )
                    v1_sy_weight_half = T.alloc_ub(Nidx1, input_dtype)
                    v1_sy_weight = T.alloc_ub(Nidx1, calc_dtype)
                    v1_sy_weight_2d = T.alloc_ub((Nidx1, n_block), calc_dtype)
                    v1_sy_reduce = T.alloc_ub(n_block, calc_dtype)
                    v1_sy_chunk = T.alloc_ub(n_block, calc_dtype)
                    v1_sy_valid_mask = T.alloc_ub(n_block // 8, "uint8")
                    v1_sy_broadcast = T.alloc_ub((1, n_block), calc_dtype)
                    v1_sy_max = T.alloc_ub((1,), calc_dtype)
                    v1_sy_sum = T.alloc_ub((1,), calc_dtype)
                    v1_sy_tmp_max = T.alloc_ub((1,), calc_dtype)
                    v1_sy_tmp_sum = T.alloc_ub((1,), calc_dtype)
                    v1_sy_eps = T.alloc_ub((1,), calc_dtype)
                    v1_sy_floor = T.alloc_ub((1,), calc_dtype)
                    v1_sy_tmp = T.alloc_ub((Nidx1, 8), calc_dtype)
                    T.copy(Weights[bz_task, s1_id, 0:Nidx1], v1_sy_weight_half)
                    T.set_flag("mte2", "v", 2)
                    T.set_flag("mte3", "v", 1)
                    T.wait_flag("mte2", "v", 2)
                    T.tile.cast(v1_sy_weight, v1_sy_weight_half, "CAST_NONE", Nidx1,)
                    T.tile.fill(v1_sy_weight_2d, 1.0)
                    T.pipe_barrier("v")
                    for sy_col_chunk in T.serial(p_col_chunk_count):
                        sy_col_start = sy_col_chunk * p_cols_per_chunk
                        T.tile.row_expand_mul_experiment(
                            v1_sy_weight_2d[
                                0:Nidx1, sy_col_start:sy_col_start + p_cols_per_chunk,
                            ],
                            v1_sy_weight_2d[
                                0:Nidx1, sy_col_start:sy_col_start + p_cols_per_chunk,
                            ], v1_sy_weight, v1_sy_tmp,
                        )
                    T.tile.fill(v1_sy_max, -T.infinity(calc_dtype))
                    T.set_flag("v", "mte2", 3)
                    for n_idx in T.serial(n_num):
                        n_start = n_idx * n_block
                        if n_start < valid_k_size:
                            T.wait_flag("v", "mte2", 3)
                            T.copy(bmm2Res[bz_task, s1_id, 0:Nidx1, n_start:n_start + n_block], v1_sy_storage[0:Nidx1, 0:n_block],)
                            if not full_valid and n_start + n_block > valid_k_size:
                                T.copy(ValidKMask[ bz_task, s1_id, n_idx * n_block:(n_idx + 1) * n_block, ], v1_sy_chunk,)
                            T.set_flag("mte2", "v", 3)
                            T.wait_flag("mte2", "v", 3)
                            T.tile.mul(
                                v1_sy_storage[0:Nidx1, 0:n_block], v1_sy_storage[0:Nidx1, 0:n_block], v1_sy_weight_2d,
                            )
                            T.pipe_barrier("v")
                            T.reduce_sum(
                                v1_sy_storage[0:Nidx1, 0:n_block], v1_sy_reduce, 0,
                            )
                            if not full_valid and n_start + n_block > valid_k_size:
                                T.tile.compare(
                                    v1_sy_valid_mask, v1_sy_chunk, 0.0, "GT",
                                )
                                T.pipe_barrier("v")
                                T.tile.select(
                                    v1_sy_reduce, v1_sy_valid_mask, v1_sy_reduce, -T.infinity(calc_dtype),
                                    "VSEL_TENSOR_SCALAR_MODE",
                                )
                                T.pipe_barrier("v")
                            T.copy(v1_sy_reduce, v1_sy_storage[Nidx1 + n_idx, 0:n_block],)
                            T.pipe_barrier("v")
                            T.tile.mul(v1_sy_broadcast[0, :], v1_sy_reduce, 1.0)
                            T.pipe_barrier("v")
                            T.reduce_max(v1_sy_broadcast, v1_sy_tmp_max, 1)
                            T.pipe_barrier("v")
                            T.tile.max(v1_sy_max, v1_sy_max, v1_sy_tmp_max)
                            T.set_flag("v", "mte2", 3)
                        else:
                            T.tile.fill(
                                v1_sy_storage[Nidx1 + n_idx, 0:n_block], -T.infinity(calc_dtype),
                            )
                    T.tile.fill(v1_sy_floor, -1e20)
                    T.pipe_barrier("v")
                    T.tile.max(v1_sy_max, v1_sy_max, v1_sy_floor)
                    T.wait_flag("v", "mte2", 3)
                    T.tile.fill(v1_sy_sum, 0.0)
                    for n_idx in T.serial(n_num):
                        n_start = n_idx * n_block
                        if n_start < valid_k_size:
                            T.tile.broadcast(v1_sy_broadcast, v1_sy_max)
                            T.pipe_barrier("v")
                            T.tile.sub(
                                v1_sy_storage[Nidx1 + n_idx, 0:n_block], v1_sy_storage[Nidx1 + n_idx, 0:n_block],
                                v1_sy_broadcast[0, :],
                            )
                            T.pipe_barrier("v")
                            T.tile.exp(
                                v1_sy_storage[Nidx1 + n_idx, 0:n_block], v1_sy_storage[Nidx1 + n_idx, 0:n_block],
                            )
                            T.pipe_barrier("v")
                            if not full_valid and n_start + n_block > valid_k_size:
                                # 明确将部分尾块中的无效列置零，不依赖极小 exp 近似。
                                T.tile.compare(
                                    v1_sy_valid_mask, v1_sy_chunk, 0.0, "GT",
                                )
                                T.pipe_barrier("v")
                                T.tile.select(
                                    v1_sy_storage[Nidx1 + n_idx, 0:n_block], v1_sy_valid_mask,
                                    v1_sy_storage[Nidx1 + n_idx, 0:n_block], 0.0, "VSEL_TENSOR_SCALAR_MODE",
                                )
                                T.pipe_barrier("v")
                        else:
                            T.tile.fill(
                                v1_sy_storage[Nidx1 + n_idx, 0:n_block], 0.0,
                            )
                            T.pipe_barrier("v")
                        T.tile.mul(
                            v1_sy_broadcast[0, :], v1_sy_storage[Nidx1 + n_idx, 0:n_block], 1.0,
                        )
                        T.pipe_barrier("v")
                        T.reduce_sum(v1_sy_broadcast, v1_sy_tmp_sum, 1)
                        T.pipe_barrier("v")
                        T.tile.add(v1_sy_sum, v1_sy_sum, v1_sy_tmp_sum)
                    T.tile.fill(v1_sy_eps, 1e-8)
                    T.pipe_barrier("v")
                    T.tile.max(v1_sy_sum, v1_sy_sum, v1_sy_eps)
                    for n_idx in T.serial(n_num):
                        T.wait_flag("mte3", "v", 1)
                        T.tile.broadcast(v1_sy_broadcast, v1_sy_sum)
                        T.pipe_barrier("v")
                        T.tile.div(
                            v1_sy_storage[Nidx1 + n_idx, 0:n_block], v1_sy_storage[Nidx1 + n_idx, 0:n_block],
                            v1_sy_broadcast[0, :],
                        )
                        T.set_flag("v", "mte3", 1)
                        T.wait_flag("v", "mte3", 1)
                        T.copy(v1_sy_storage[Nidx1 + n_idx, 0:n_block], SoftmaxI[bz_task, s1_id, n_idx * n_block:(n_idx + 1) * n_block],)
                        T.set_flag("mte3", "v", 1)
                    T.wait_flag("mte3", "v", 1)
            if task_slot == 0:
                T.set_cross_flag("MTE3", V1_PSY_TO_AIC_FLAGS[0])
                T.wait_cross_flag(V1_PSY_RELAY_FLAGS[0])
            else:
                T.set_cross_flag("MTE3", V1_PSY_TO_AIC_FLAGS[1])
                T.wait_cross_flag(V1_PSY_RELAY_FLAGS[1])
    @T.macro
    def relay_vector1_psy(task_slot):
        with T.Scope("C"):
            if task_slot == 0:
                T.wait_cross_flag(V1_PSY_TO_AIC_FLAGS[0])
                T.set_cross_flag("FIX", V1_PSY_RELAY_FLAGS[0])
            else:
                T.wait_cross_flag(V1_PSY_TO_AIC_FLAGS[1])
                T.set_cross_flag("FIX", V1_PSY_RELAY_FLAGS[1])
    @T.macro
    def vector1_backward_task(
        KRealSize, ValidKMask, Weights, bmm2Res, PReduced, SoftmaxI, dWeightOut, reluGradRes,
        LossPartial, bz_task, s1_begin_task, s1_end_task, task_slot, vector_id,
    ):
        """为一个完整任务翻译 VectorDwDqDk + VectorLoss。
        配对的 AIV 分别负责互不重叠的 Nidx1 半区。只有 Cube relay 观察到
        两个 producer 都完成后，双方才消费 P/SY workspace。ReLU 导数使用
        1-exp(-10000*x) 算术门控，规避 Compare packed mask 与 Select 的布局问题。
        """
        with T.Scope("V"):
            v1_bw_weight_half = T.alloc_ub(rows_per_aiv, input_dtype)
            v1_bw_weight = T.alloc_ub(rows_per_aiv, calc_dtype)
            v1_bw_weight_2d = T.alloc_ub(
                (rows_per_aiv, bw_col_block), calc_dtype
            )
            v1_bw_p = T.alloc_ub(bw_col_block, calc_dtype)
            v1_bw_sy = T.alloc_ub(bw_col_block, calc_dtype)
            v1_bw_diff = T.alloc_ub(bw_col_block, calc_dtype)
            v1_bw_valid = T.alloc_ub(bw_col_block, calc_dtype)
            v1_bw_diff_2d = T.alloc_ub(
                (rows_per_aiv, bw_col_block), calc_dtype
            )
            v1_bw_bmm2 = T.alloc_ub(
                (rows_per_aiv, bw_col_block), calc_dtype
            )
            v1_bw_rg = T.alloc_ub(
                (rows_per_aiv, bw_col_block), calc_dtype
            )
            v1_bw_rg_half = T.alloc_ub(
                (rows_per_aiv, bw_col_block), input_dtype
            )
            v1_bw_dw = T.alloc_ub(rows_per_aiv, calc_dtype)
            v1_bw_dw_accum = T.alloc_ub(rows_per_aiv, calc_dtype)
            v1_bw_dw_half = T.alloc_ub(rows_per_aiv, input_dtype)
            v1_bw_eps = T.alloc_ub(bw_col_block, calc_dtype)
            v1_bw_log_p = T.alloc_ub(bw_col_block, calc_dtype)
            v1_bw_log_sy = T.alloc_ub(bw_col_block, calc_dtype)
            v1_bw_loss_2d = T.alloc_ub((1, bw_col_block), calc_dtype)
            v1_bw_loss_sum = T.alloc_ub((1,), calc_dtype)
            v1_bw_loss_accum = T.alloc_ub((1,), calc_dtype)
            v1_bw_row_tmp = T.alloc_ub((rows_per_aiv, 8), calc_dtype)
            T.tile.fill(v1_bw_eps, 1e-8)
            # reluGrad store 与下一轮 diff 复用 UB，由 MTE3 将其归还给 Vector。
            T.set_flag("mte3", "v", 2)
            T.set_flag("mte3", "v", 3)
            for s1_id in T.serial(s1_begin_task, s1_end_task):
                g_start = vector_id * rows_per_aiv
                T.tile.fill(v1_bw_dw_accum, 0.0)
                T.tile.fill(v1_bw_loss_accum, 0.0)
                T.copy(Weights[bz_task, s1_id, g_start:g_start + rows_per_aiv], v1_bw_weight_half,)
                T.set_flag("mte2", "v", 2)
                T.wait_flag("mte2", "v", 2)
                T.tile.cast(v1_bw_weight, v1_bw_weight_half, "CAST_NONE", rows_per_aiv,)
                if rows_per_aiv == 4:
                    for bw_row in T.serial(rows_per_aiv):
                        T.tile.broadcast(
                            v1_bw_weight_2d[
                                bw_row, 0:bw_col_block
                            ], v1_bw_weight[bw_row:bw_row + 1],
                        )
                else:
                    T.tile.fill(v1_bw_weight_2d, 1.0)
                    T.pipe_barrier("v")
                    T.tile.row_expand_mul_experiment(
                        v1_bw_weight_2d, v1_bw_weight_2d, v1_bw_weight, v1_bw_row_tmp,
                    )
                T.pipe_barrier("v")
                T.set_flag("v", "mte2", 3)
                for n_idx in T.serial(n_num):
                    for bw_col_id in T.serial(bw_col_count):
                        bw_col_start = (
                            n_idx * n_block + bw_col_id * bw_col_block
                        )
                        bw_col_end = bw_col_start + bw_col_block
                        T.wait_flag("v", "mte2", 3)
                        T.copy(PReduced[bz_task, s1_id, bw_col_start:bw_col_end], v1_bw_p,)
                        T.copy(SoftmaxI[bz_task, s1_id, bw_col_start:bw_col_end], v1_bw_sy,)
                        T.copy(bmm2Res[ bz_task, s1_id, g_start:g_start + rows_per_aiv, bw_col_start:bw_col_end, ], v1_bw_bmm2,)
                        if not full_valid:
                            T.copy(ValidKMask[bz_task, s1_id, bw_col_start:bw_col_end], v1_bw_valid,)
                        T.set_flag("mte2", "v", 3)
                        T.wait_flag("mte2", "v", 3)
                        # diff 与上一块仍在被 MTE3 读取的 reluGrad 复用 UB，覆写前等待归还。
                        T.wait_flag("mte3", "v", 2)
                        T.tile.sub(
                            v1_bw_diff, v1_bw_sy, v1_bw_p,
                        )
                        T.pipe_barrier("v")
                        T.tile.broadcast(
                            v1_bw_diff_2d, v1_bw_diff,
                        )
                        # bmm2Res 已经过 ReLU；算术门控避免 packed mask 的布局歧义。
                        T.tile.mul(v1_bw_rg, v1_bw_bmm2, -10000.0)
                        T.pipe_barrier("v")
                        T.tile.exp(v1_bw_rg, v1_bw_rg)
                        T.pipe_barrier("v")
                        T.tile.mul(v1_bw_rg, v1_bw_rg, -1.0)
                        T.pipe_barrier("v")
                        T.tile.add(v1_bw_rg, v1_bw_rg, 1.0)
                        T.pipe_barrier("v")
                        T.tile.mul(
                            v1_bw_bmm2, v1_bw_bmm2, v1_bw_diff_2d,
                        )
                        T.pipe_barrier("v")
                        T.reduce_sum(
                            v1_bw_bmm2, v1_bw_dw, 1,
                        )
                        T.pipe_barrier("v")
                        T.tile.add(v1_bw_dw_accum, v1_bw_dw_accum, v1_bw_dw)
                        T.tile.mul(
                            v1_bw_bmm2, v1_bw_weight_2d, v1_bw_diff_2d,
                        )
                        T.pipe_barrier("v")
                        T.tile.mul(v1_bw_bmm2, v1_bw_bmm2, v1_bw_rg)
                        T.pipe_barrier("v")
                        T.tile.cast(v1_bw_rg_half, v1_bw_bmm2, "CAST_RINT", rows_per_aiv * bw_col_block,)
                        T.pipe_barrier("v")
                        T.set_flag("v", "mte3", 2)
                        T.wait_flag("v", "mte3", 2)
                        T.pipe_barrier("mte3")
                        T.copy(v1_bw_rg_half, reluGradRes[bz_task, s1_id, g_start:g_start + rows_per_aiv, bw_col_start:bw_col_end],)
                        T.set_flag("mte3", "v", 2)
                        # 在 AIV 对间拆分外层 K 分块，由所属 AIV 消费两个物理子 tile。
                        if n_idx % 2 == vector_id:
                            T.tile.max(v1_bw_p, v1_bw_p, v1_bw_eps,)
                            T.tile.max(v1_bw_sy, v1_bw_sy, v1_bw_eps,)
                            T.pipe_barrier("v")
                            T.tile.ln(
                                v1_bw_log_p, v1_bw_p,
                            )
                            T.tile.ln(
                                v1_bw_log_sy, v1_bw_sy,
                            )
                            T.pipe_barrier("v")
                            T.tile.sub(
                                v1_bw_loss_2d[0, :], v1_bw_log_p, v1_bw_log_sy,
                            )
                            T.pipe_barrier("v")
                            T.tile.mul(
                                v1_bw_loss_2d[0, :], v1_bw_loss_2d[0, :], v1_bw_p,
                            )
                            T.pipe_barrier("v")
                            if not full_valid:
                                T.tile.mul(
                                    v1_bw_loss_2d[0, :], v1_bw_loss_2d[0, :], v1_bw_valid,
                                )
                                T.pipe_barrier("v")
                            T.reduce_sum(
                                v1_bw_loss_2d, v1_bw_loss_sum, 1
                            )
                            T.pipe_barrier("v")
                            T.tile.add(
                                v1_bw_loss_accum, v1_bw_loss_accum, v1_bw_loss_sum,
                            )
                        T.set_flag("v", "mte2", 3)
                T.wait_flag("v", "mte2", 3)
                T.pipe_barrier("v")
                T.wait_flag("mte3", "v", 3)
                T.tile.cast(v1_bw_dw_half, v1_bw_dw_accum, "CAST_RINT", rows_per_aiv,)
                T.set_flag("v", "mte3", 3)
                T.wait_flag("v", "mte3", 3)
                T.copy(v1_bw_dw_half, dWeightOut[bz_task, s1_id, g_start:g_start + rows_per_aiv],)
                T.copy(v1_bw_loss_accum, LossPartial[ ((bz_task * S1 + s1_id) * 2 + vector_id) * loss_partial_stride: ((bz_task * S1 + s1_id) * 2 + vector_id) * loss_partial_stride + 1 ],)
                T.set_flag("mte3", "v", 3)
            T.wait_flag("mte3", "v", 2)
            T.wait_flag("mte3", "v", 3)
            if task_slot == 0:
                T.set_cross_flag("MTE3", V1_TO_C2_FLAGS[0])
            else:
                T.set_cross_flag("MTE3", V1_TO_C2_FLAGS[1])
    @T.macro
    def cube2_task(
        QueryIndex, GatherKeyIndex_ws, reluGradRes, C2ZeroPad, dQueryIndexAccum,
        dk_matmul_partial, bz_task, s1_begin_task, s1_end_task, task_slot,
    ):
        """使用显式 MMA 流水线计算 dKey partial 和 dQuery。"""
        with T.Scope("C"):
            c2_rg_l1 = T.alloc_L1((c2_dk_k, n_block), input_dtype)
            c2_qi_l1 = T.alloc_L1((c2_dk_k, DQueryIndex), input_dtype)
            c2_gki_l1 = T.alloc_L1((n_block, DQueryIndex), input_dtype)
            c2_dk_rg_l0a = T.alloc_L0A((n_block, c2_dk_k), input_dtype)
            c2_dk_qi_l0b = T.alloc_L0B((c2_dk_k, DQueryIndex), input_dtype)
            c2_dk_l0c = T.alloc_L0C((n_block, DQueryIndex), calc_dtype)
            c2_dq_rg_l0a = T.alloc_L0A((Nidx1, n_block), input_dtype)
            c2_dq_gki_l0b = T.alloc_L0B((n_block, DQueryIndex), input_dtype)
            c2_dq_l0c = T.alloc_L0C((Nidx1, DQueryIndex), calc_dtype)
            T.copy(QueryIndex[bz_task, s1_begin_task, 0:Nidx1, 0:DQueryIndex], c2_qi_l1[0:Nidx1, 0:DQueryIndex],)
            if Nidx1 == 8:
                # BF16 Cube 的 reduction 维度需为 16，逻辑 8 行之外的物理尾部置零。
                T.copy(C2ZeroPad, c2_qi_l1[8:16, 0:DQueryIndex])
                T.copy(C2ZeroPad, c2_rg_l1[8:16, 0:n_block])
            T.set_flag("mte2", "mte1", 6)
            if task_slot == 0:
                T.wait_cross_flag(V1_TO_C2_FLAGS[0])
            else:
                T.wait_cross_flag(V1_TO_C2_FLAGS[1])
            T.set_flag("mte1", "mte2", 2)
            T.set_flag("m", "mte1", 2)
            T.set_flag("fix", "m", 2)
            T.set_flag("m", "mte1", 3)
            T.set_flag("fix", "m", 3)
            T.wait_flag("mte2", "mte1", 6)
            for s1_id in T.serial(s1_begin_task, s1_end_task):
                for n_idx in T.serial(n_num):
                    T.wait_flag("mte1", "mte2", 2)
                    T.copy(reluGradRes[bz_task, s1_id, 0:Nidx1, n_idx * n_block:(n_idx + 1) * n_block], c2_rg_l1[0:Nidx1, 0:n_block],)
                    T.copy(GatherKeyIndex_ws[bz_task, s1_id, n_idx * n_block:(n_idx + 1) * n_block, 0:DQueryIndex], c2_gki_l1,)
                    T.set_flag("mte2", "mte1", 2)
                    T.wait_flag("m", "mte1", 2)
                    T.wait_flag("m", "mte1", 3)
                    T.wait_flag("mte2", "mte1", 2)
                    T.copy(c2_rg_l1, c2_dk_rg_l0a, transpose=True)
                    T.copy(c2_qi_l1, c2_dk_qi_l0b)
                    T.copy(c2_rg_l1[0:Nidx1, 0:n_block], c2_dq_rg_l0a,)
                    T.copy(c2_gki_l1, c2_dq_gki_l0b)
                    T.set_flag("mte1", "mte2", 2)
                    T.set_flag("mte1", "m", 2)
                    T.set_flag("mte1", "m", 3)
                    T.wait_flag("mte1", "m", 2)
                    T.wait_flag("fix", "m", 2)
                    T.mma(c2_dk_rg_l0a, c2_dk_qi_l0b, c2_dk_l0c, init=True)
                    T.set_flag("m", "mte1", 2)
                    T.set_flag("m", "fix", 2)
                    T.wait_flag("m", "fix", 2)
                    T.copy(c2_dk_l0c, dk_matmul_partial[bz_task, s1_id, n_idx * n_block:(n_idx + 1) * n_block, 0:DQueryIndex],)
                    T.set_flag("fix", "m", 2)
                    T.wait_flag("mte1", "m", 3)
                    T.wait_flag("fix", "m", 3)
                    T.mma(c2_dq_rg_l0a, c2_dq_gki_l0b, c2_dq_l0c, init=(n_idx == 0))
                    T.set_flag("m", "mte1", 3)
                    if n_idx + 1 < n_num:
                        T.set_flag("fix", "m", 3)
                T.set_flag("m", "fix", 3)
                T.wait_flag("m", "fix", 3)
                T.copy(c2_dq_l0c, dQueryIndexAccum[bz_task, s1_id, 0:Nidx1, 0:DQueryIndex],)
                T.set_flag("fix", "m", 3)
            T.wait_flag("mte1", "mte2", 2)
            T.wait_flag("m", "mte1", 2)
            T.wait_flag("fix", "m", 2)
            T.wait_flag("m", "mte1", 3)
            T.wait_flag("fix", "m", 3)
    @T.macro
    def clear_dkey_accumulator(dKeyIndexAccum, core_id, vector_id):
        """在 kernel 内由所有 AIV 协作清零 FP32 scatter 目标。"""
        with T.Scope("V"):
            v2_zero_batch = T.alloc_ub(
                (dkey_clear_batch, DQueryIndex), calc_dtype
            )
            aiv_id = core_id * 2 + vector_id
            clear_base_batches = dkey_clear_total_batches // total_aiv_num
            clear_tail_aivs = dkey_clear_total_batches % total_aiv_num
            clear_batch_count = clear_base_batches + T.if_then_else(
                aiv_id < clear_tail_aivs, 1, 0
            )
            clear_batch_start = (
                aiv_id * clear_base_batches + T.min(aiv_id, clear_tail_aivs)
            )
            T.tile.fill(v2_zero_batch, 0.0)
            T.set_flag("v", "mte3", 5)
            T.wait_flag("v", "mte3", 5)
            for clear_iter in T.serial(clear_batch_count):
                clear_row = (
                    clear_batch_start + clear_iter
                ) * dkey_clear_batch
                T.copy(v2_zero_batch, dKeyIndexAccum[ clear_row:clear_row + dkey_clear_batch, 0:DQueryIndex, ],)
            T.set_flag("mte3", "v", 5)
            T.wait_flag("mte3", "v", 5)
    @T.macro
    def vector2_scatter_all_tasks(
        SparseIndices, KRealSize, dk_matmul_partial, dKeyIndexAccum, core_id, vector_id,
    ):
        """使用所有 core，对每个任务执行批量 GM→UB 和 ping-pong scatter。"""
        with T.Scope("V"):
            v2_dk_batch_ping = T.alloc_ub(
                (v2_scatter_batch, DQueryIndex), calc_dtype
            )
            v2_dk_batch_pong = T.alloc_ub(
                (v2_scatter_batch, DQueryIndex), calc_dtype
            )
            core_batch_count = v2_base_batches_per_core + T.if_then_else(
                core_id < v2_tail_batch_cores, 1, 0
            )
            core_batch_start = (
                core_id * v2_base_batches_per_core
                + T.min(core_id, v2_tail_batch_cores)
            )
            first_vector_batches = (core_batch_count + 1) // 2
            vector_batch_count = (
                first_vector_batches
                - vector_id * (2 * first_vector_batches - core_batch_count)
            )
            vector_batch_start = (
                core_batch_start + vector_id * first_vector_batches
            )
            aiv_k_start = vector_batch_start * v2_scatter_batch
            T.set_flag("mte3", "mte2", 6)
            T.set_flag("mte3", "mte2", 7)
            for task_id in T.serial(task_count):
                bz_task = task_id // S1
                s1_task = task_id % S1
                # 每个 task 仅读取一次真实 K 长度；DMA 范围由 k_scatter_exec 静态决定。
                valid_k_size = K if full_valid else KRealSize[bz_task, s1_task]
                if vector_batch_count > 0:
                    T.wait_flag("mte3", "mte2", 6)
                    T.copy(dk_matmul_partial[ bz_task, s1_task, aiv_k_start:aiv_k_start + v2_scatter_batch, 0:DQueryIndex, ], v2_dk_batch_ping,)
                    T.set_flag("mte2", "mte3", 6)
                    for batch_id in T.serial(vector_batch_count):
                        current_slot = batch_id % 2
                        next_slot = (batch_id + 1) % 2
                        if batch_id + 1 < vector_batch_count:
                            next_k_start = (
                                aiv_k_start
                                + (batch_id + 1) * v2_scatter_batch
                            )
                            T.wait_flag("mte3", "mte2", next_slot + 6)
                            if next_slot == 0:
                                T.copy(dk_matmul_partial[ bz_task, s1_task, next_k_start:next_k_start + v2_scatter_batch, 0:DQueryIndex, ], v2_dk_batch_ping,)
                            else:
                                T.copy(dk_matmul_partial[bz_task, s1_task, next_k_start:next_k_start + v2_scatter_batch, 0:DQueryIndex],
                                       v2_dk_batch_pong)
                            T.set_flag("mte2", "mte3", next_slot + 6)
                        T.wait_flag("mte2", "mte3", current_slot + 6)
                        for pair_idx in T.serial(v2_scatter_batch // 2):
                            row_idx = pair_idx * 2
                            k_idx = (
                                aiv_k_start
                                + batch_id * v2_scatter_batch
                                + row_idx
                            )
                            sparse_idx0 = SparseIndices[
                                bz_task, s1_task, 0, k_idx
                            ]
                            sparse_idx1 = SparseIndices[
                                bz_task, s1_task, 0, k_idx + 1
                            ]
                            if k_idx < valid_k_size:
                                if sparse_idx0 >= 0:
                                    if current_slot == 0:
                                        T.tile.atomic_add(dKeyIndexAccum[ bz_task * S2 + sparse_idx0, 0 ], v2_dk_batch_ping[ row_idx:row_idx + 1, 0:DQueryIndex ],)
                                    else:
                                        T.tile.atomic_add(dKeyIndexAccum[ bz_task * S2 + sparse_idx0, 0 ], v2_dk_batch_pong[ row_idx:row_idx + 1, 0:DQueryIndex ],)
                            if k_idx + 1 < valid_k_size:
                                if sparse_idx1 >= 0:
                                    if current_slot == 0:
                                        T.tile.atomic_add(dKeyIndexAccum[ bz_task * S2 + sparse_idx1, 0 ], v2_dk_batch_ping[ row_idx + 1:row_idx + 2, 0:DQueryIndex, ],)
                                    else:
                                        T.tile.atomic_add(dKeyIndexAccum[ bz_task * S2 + sparse_idx1, 0 ], v2_dk_batch_pong[ row_idx + 1:row_idx + 2, 0:DQueryIndex, ],)
                        T.set_flag("mte3", "mte2", current_slot + 6)
            T.wait_flag("mte3", "mte2", 6)
            T.wait_flag("mte3", "mte2", 7)
    @T.macro
    def finalize_vector_outputs(
        dQueryIndexAccum, dKeyIndexAccum, dQueryIndexOut, dKeyIndexOut, LossPartial, LossOut, core_id, vector_id,
    ):
        """分配输出 cast，并归约所有 AIV 的 loss partial。"""
        with T.Scope("V"):
            v2_finalize_dq_fp32 = T.alloc_ub((1, DQueryIndex), calc_dtype)
            v2_finalize_dq_half = T.alloc_ub((1, DQueryIndex), input_dtype)
            v2_finalize_dk_fp32 = T.alloc_ub(
                (2, dkey_finalize_batch, DQueryIndex), calc_dtype
            )
            v2_finalize_dk_half = T.alloc_ub(
                (2, dkey_finalize_batch, DQueryIndex), input_dtype
            )
            T.set_flag("v", "mte2", 4)
            T.set_flag("mte3", "v", 7)
            # 仅任务所有者转换 dQuery，避免 scatter 核承担额外标量工作和事件开销。
            for dq_task_iter in T.serial(tasks_per_core):
                dq_task = dq_task_iter * launch_core_num + core_id
                if dq_task < task_count:
                    dq_bz = dq_task // S1
                    dq_s1 = dq_task % S1
                    for dq_group_local in T.serial(Nidx1 // 2):
                        dq_group = dq_group_local * 2 + vector_id
                        T.wait_flag("v", "mte2", 4)
                        T.copy(dQueryIndexAccum[ dq_bz, dq_s1, dq_group:dq_group + 1, 0:DQueryIndex, ], v2_finalize_dq_fp32,)
                        T.set_flag("mte2", "v", 5)
                        T.wait_flag("mte2", "v", 5)
                        T.wait_flag("mte3", "v", 7)
                        T.tile.cast(v2_finalize_dq_half, v2_finalize_dq_fp32, "CAST_RINT", DQueryIndex,)
                        T.set_flag("v", "mte2", 4)
                        T.set_flag("v", "mte3", 5)
                        T.wait_flag("v", "mte3", 5)
                        T.copy(v2_finalize_dq_half, dQueryIndexOut[ dq_bz, dq_s1, dq_group:dq_group + 1, 0:DQueryIndex, ],)
                        T.set_flag("mte3", "v", 7)
            T.wait_flag("v", "mte2", 4)
            T.wait_flag("mte3", "v", 7)
            aiv_id = core_id * 2 + vector_id
            finalize_base_batches = dkey_finalize_total_batches // total_aiv_num
            finalize_tail_aivs = dkey_finalize_total_batches % total_aiv_num
            finalize_batch_count = finalize_base_batches + T.if_then_else(
                aiv_id < finalize_tail_aivs, 1, 0
            )
            finalize_batch_start = (
                aiv_id * finalize_base_batches
                + T.min(aiv_id, finalize_tail_aivs)
            )
            T.set_flag("v", "mte2", 5)
            T.set_flag("v", "mte2", 6)
            T.set_flag("mte3", "v", 5)
            T.set_flag("mte3", "v", 6)
            for finalize_iter in T.serial(finalize_batch_count):
                finalize_slot = finalize_iter % 2
                finalize_row = (
                    finalize_batch_start + finalize_iter
                ) * dkey_finalize_batch
                T.wait_flag("v", "mte2", finalize_slot + 5)
                T.copy(dKeyIndexAccum[ finalize_row:finalize_row + dkey_finalize_batch, 0:DQueryIndex, ], v2_finalize_dk_fp32[finalize_slot, :, :],)
                T.set_flag("mte2", "v", finalize_slot + 5)
                T.wait_flag("mte2", "v", finalize_slot + 5)
                T.wait_flag("mte3", "v", finalize_slot + 5)
                T.tile.cast(v2_finalize_dk_half[finalize_slot, :, :], v2_finalize_dk_fp32[finalize_slot, :, :], "CAST_RINT", dkey_finalize_batch * DQueryIndex,)
                T.set_flag("v", "mte2", finalize_slot + 5)
                T.set_flag("v", "mte3", finalize_slot + 5)
                T.wait_flag("v", "mte3", finalize_slot + 5)
                T.copy(v2_finalize_dk_half[finalize_slot, :, :], dKeyIndexOut[ finalize_row:finalize_row + dkey_finalize_batch, 0:DQueryIndex, ],)
                T.set_flag("mte3", "v", finalize_slot + 5)
            T.wait_flag("v", "mte2", 5)
            T.wait_flag("v", "mte2", 6)
            T.wait_flag("mte3", "v", 5)
            T.wait_flag("mte3", "v", 6)
            if core_id == 0:
                if vector_id == 0:
                    v2_loss_chunk = T.alloc_ub(
                        (1, loss_reduce_chunk), calc_dtype
                    )
                    v2_loss_chunk_sum = T.alloc_ub((1,), calc_dtype)
                    v2_loss_total = T.alloc_ub((1,), calc_dtype)
                    T.tile.fill(v2_loss_total, 0.0)
                    for loss_chunk_id in T.serial(loss_full_chunk_count):
                        loss_chunk_start = loss_chunk_id * loss_reduce_chunk
                        T.set_flag("v", "mte2", 4)
                        T.wait_flag("v", "mte2", 4)
                        T.copy(LossPartial[ loss_chunk_start:loss_chunk_start + loss_reduce_chunk ], v2_loss_chunk[0, :],)
                        T.set_flag("mte2", "v", 4)
                        T.wait_flag("mte2", "v", 4)
                        T.reduce_sum(
                            v2_loss_chunk, v2_loss_chunk_sum, 1,
                        )
                        T.pipe_barrier("v")
                        T.tile.add(
                            v2_loss_total, v2_loss_total, v2_loss_chunk_sum,
                        )
                    T.set_flag("v", "mte3", 4)
                    T.wait_flag("v", "mte3", 4)
                    T.copy(v2_loss_total, LossOut[0:1])
                    T.set_flag("mte3", "v", 4)
                    T.wait_flag("mte3", "v", 4)
    @T.prim_func
    def main(
        SparseIndices: T.Tensor((B, S1, Nidx2, sparse_k_stride), "int32"),
        KRealSize: T.Tensor(k_real_size_shape, "int32"), ValidKMask: T.Tensor(valid_k_mask_shape, calc_dtype),
        V0ZeroRow: T.Tensor((DQuery,), input_dtype), Query: T.Tensor((B, S1, N1, DQuery), input_dtype),
        QueryRope: T.Tensor((B, S1, N1, d_rope), input_dtype), Key: T.Tensor((B, source_s2, N2, DQuery), input_dtype),
        KeyRope: T.Tensor((B, source_s2, N2, d_rope), input_dtype),
        QueryIndex: T.Tensor((B, S1, Nidx1, DQueryIndex), input_dtype),
        KeyIndex: T.Tensor((B, source_s2, Nidx2, DQueryIndex), input_dtype),
        Weights: T.Tensor((B, S1, Nidx1), input_dtype), SoftmaxMax: T.Tensor((B, N2, S1, G), calc_dtype),
        SoftmaxSum: T.Tensor((B, N2, S1, G), calc_dtype),
        dQueryIndexAccum: T.Tensor((B, S1, Nidx1, DQueryIndex), calc_dtype),
        dKeyIndexAccum: T.Tensor((dkey_output_rows, DQueryIndex), calc_dtype),
        GatherKey_ws: T.Tensor((B, S1, K, DQuery), input_dtype),
        GatherKeyRope_ws: T.Tensor((B, S1, K, d_rope), input_dtype),
        GatherKeyIndex_ws: T.Tensor((B, S1, K, DQueryIndex), input_dtype),
        bmm1Res: T.Tensor((B, S1, N1, K), calc_dtype), bmm2Res: T.Tensor((B, S1, Nidx1, K), calc_dtype),
        PReduced: T.Tensor((B, S1, K), calc_dtype), SoftmaxI: T.Tensor((B, S1, K), calc_dtype),
        dk_matmul_partial: T.Tensor((B, S1, v2_k_padded, DQueryIndex), calc_dtype),
        reluGradRes: T.Tensor((B, S1, Nidx1, K), input_dtype), LossPartial: T.Tensor((loss_partial_size,), calc_dtype),
        C2ZeroPad: T.Tensor((8, DQueryIndex), input_dtype),
        dQueryIndex: T.Tensor((B, S1, Nidx1, DQueryIndex), input_dtype),
        dKeyIndexOut: T.Tensor((dkey_output_rows, DQueryIndex), input_dtype),
        dWeightOut: T.Tensor((B, S1, Nidx1), input_dtype), LossOut: T.Tensor((1,), calc_dtype),
    ):
        with T.Kernel(launch_core_num, is_npu=True) as (cid, vid):
            for pipe_iter in T.serial(tasks_per_core + task_pipeline_lag):
                if pipe_iter < tasks_per_core:
                    task_current = pipe_iter * launch_core_num + cid
                    if task_current < task_count:
                        bz_current = task_current // S1
                        s1_current = task_current % S1
                        vector0_gather_task(
                            SparseIndices, V0ZeroRow, Key, KeyRope, KeyIndex, GatherKey_ws,
                            GatherKeyRope_ws, GatherKeyIndex_ws, bz_current, s1_current, s1_current + 1, pipe_iter % 2,
                            vid,
                        )
                if pipe_iter >= task_pipeline_lag:
                    task_prev_iter = pipe_iter - task_pipeline_lag
                    if task_prev_iter < tasks_per_core:
                        task_prev = task_prev_iter * launch_core_num + cid
                        if task_prev < task_count:
                            bz_prev = task_prev // S1
                            s1_prev = task_prev % S1
                            task_prev_slot = task_prev_iter % 2
                            cube1_task(
                                Query, QueryRope, QueryIndex, GatherKey_ws, GatherKeyRope_ws,
                                GatherKeyIndex_ws, bmm1Res, bmm2Res, bz_prev, s1_prev, s1_prev + 1, task_prev_slot,
                            )
                            vector1_psy_task(
                                KRealSize, ValidKMask, SoftmaxMax, SoftmaxSum, Weights,
                                bmm1Res, bmm2Res, PReduced, SoftmaxI, bz_prev, s1_prev, s1_prev + 1,
                                task_prev_slot, vid,
                            )
                            relay_vector1_psy(task_prev_slot)
                            vector1_backward_task(
                                KRealSize, ValidKMask, Weights, bmm2Res, PReduced, SoftmaxI,
                                dWeightOut, reluGradRes, LossPartial, bz_prev, s1_prev, s1_prev + 1,
                                task_prev_slot, vid,
                            )
                            cube2_task(
                                QueryIndex, GatherKeyIndex_ws, reluGradRes, C2ZeroPad,
                                dQueryIndexAccum, dk_matmul_partial, bz_prev, s1_prev, s1_prev + 1, task_prev_slot,
                            )
            # 所有任务完成 C2 后再开始全核确定性 scatter，空闲核也可参与。
            T.sync_all()
            clear_dkey_accumulator(dKeyIndexAccum, cid, vid)
            T.sync_all()
            vector2_scatter_all_tasks(
                SparseIndices, KRealSize, dk_matmul_partial, dKeyIndexAccum, cid, vid,
            )
            T.sync_all()
            finalize_vector_outputs(
                dQueryIndexAccum, dKeyIndexAccum, dQueryIndex, dKeyIndexOut, LossPartial, LossOut, cid, vid,
            )
    return main


# ==============================================================================
# Host runtime：参数归一化、specialization 选择、workspace 缓存和布局包装。
# ==============================================================================
def _host_sequence_lengths(
    sequence_lengths: Optional[torch.Tensor], default_length: int, batch_size: int,
) -> Optional[list[int]]:
    if sequence_lengths is None:
        return [default_length] * batch_size
    if isinstance(sequence_lengths, (list, tuple)):
        return [int(length) for length in sequence_lengths]
    if isinstance(sequence_lengths, torch.Tensor) and sequence_lengths.device.type == "cpu":
        return [int(length) for length in sequence_lengths.tolist()]
    return None

def _derive_execution_widths(
    actual_query_lengths: Optional[torch.Tensor], actual_key_lengths: Optional[torch.Tensor],
    batch_size: int, query_length: int, key_length: int, top_k: int,
) -> Tuple[int, int]:
    """返回 Cube 执行宽度与 V2 scatter 执行宽度。
    两个宽度都只依赖主机可见的长度。若长度仅在 NPU 上可见，不能为了
    取得 maxRealK 引入同步，保守地让二者均退化为原始 K。
    """
    host_query_lengths = _host_sequence_lengths(
        actual_query_lengths, query_length, batch_size
    )
    host_key_lengths = _host_sequence_lengths(
        actual_key_lengths, key_length, batch_size
    )
    if host_query_lengths is None or host_key_lengths is None:
        return top_k, top_k
    if len(host_query_lengths) != batch_size or len(host_key_lengths) != batch_size:
        return top_k, top_k
    max_real_k = max(
        min(top_k, max(0, key_len - query_len + query_length))
        for query_len, key_len in zip(host_query_lengths, host_key_lengths)
    )
    k_exec = max(
        S2_WORKSPACE, ceil_div(max_real_k, S2_WORKSPACE) * S2_WORKSPACE,
    )
    k_scatter_exec = max(
        V2_SCATTER_BATCH, ceil_div(max_real_k, V2_SCATTER_BATCH) * V2_SCATTER_BATCH,
    )
    return k_exec, min(k_scatter_exec, k_exec)

def _is_static_full_valid(
    actual_query_lengths: Optional[torch.Tensor], actual_key_lengths: Optional[torch.Tensor],
    batch_size: int, query_length: int, key_length: int, top_k: int,
) -> bool:
    """仅在主机侧长度可见且每个查询位置均具有完整 K 时启用快路径。"""
    host_query_lengths = _host_sequence_lengths(
        actual_query_lengths, query_length, batch_size
    )
    host_key_lengths = _host_sequence_lengths(
        actual_key_lengths, key_length, batch_size
    )
    if host_query_lengths is None or host_key_lengths is None:
        return False
    if len(host_query_lengths) != batch_size or len(host_key_lengths) != batch_size:
        return False
    return all(
        key_len - query_len + 1 >= top_k
        for query_len, key_len in zip(host_query_lengths, host_key_lengths)
    )

def _get_full_valid_aux_inputs(
    device: torch.device, calc_dtype: torch.dtype, batch_size: int, query_length: int, top_k: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """返回 full-valid specialization 的安全占位输入，并跨调用复用。
    TileLang 会校验条件表达式的两个 BufferRegion；即使 full-valid 代码生成
    最终不会读取 KRealSize，也必须给该形状提供合法的二维占位张量。
    """
    cache_key = (device, calc_dtype, batch_size, query_length, top_k)
    cached = _full_valid_aux_cache.get(cache_key)
    if cached is None:
        cached = (torch.full((batch_size, query_length), top_k, dtype=torch.int32, device=device),
                  torch.empty((batch_size, query_length, top_k), dtype=calc_dtype, device=device))
        _full_valid_aux_cache[cache_key] = cached
    return cached

def _get_tail_aux_inputs(
    device: torch.device, calc_dtype: torch.dtype, host_query_lengths: list[int],
    host_key_lengths: list[int], batch_size: int, query_length: int, top_k: int, execution_k: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """缓存 host 可见长度对应的尾块元数据，避免每次调用启动 NPU 预处理算子。"""
    cache_key = (
        device, calc_dtype, tuple(host_query_lengths), tuple(host_key_lengths),
        batch_size, query_length, top_k, execution_k,
    )
    cached = _tail_aux_cache.get(cache_key)
    if cached is None:
        real_sizes = [
            [
                min(top_k, max(0, key_len - query_len + query_pos + 1))
                for query_pos in range(query_length)
            ]
            for query_len, key_len in zip(host_query_lengths, host_key_lengths)
        ]
        k_real_size = torch.tensor(
            real_sizes, dtype=torch.int32, device=device,
        )
        valid_k_mask = (
            torch.arange(execution_k, dtype=torch.int32, device=device).view(
                1, 1, execution_k
            )
            < k_real_size.unsqueeze(-1)
        ).to(calc_dtype).contiguous()
        cached = (k_real_size, valid_k_mask)
        _tail_aux_cache[cache_key] = cached
    return cached

def _get_v0_zero_row(
    device: torch.device, input_dtype: torch.dtype,
) -> torch.Tensor:
    """返回 V0 三路无效索引共用的 GM 零哨兵行。"""
    cache_key = (device, input_dtype)
    cached = _v0_zero_row_cache.get(cache_key)
    if cached is None:
        cached = torch.zeros(DQUERY, dtype=input_dtype, device=device)
        _v0_zero_row_cache[cache_key] = cached
    return cached

def sparse_lightning_indexer_grad_kl_loss(
    query: torch.Tensor, key: torch.Tensor, query_index: torch.Tensor, key_index: torch.Tensor,
    weights: torch.Tensor, sparse_indices: torch.Tensor, softmax_max: torch.Tensor,
    softmax_sum: torch.Tensor, scale_value: float, *,
    query_rope: Optional[torch.Tensor] = None, key_rope: Optional[torch.Tensor] = None,
    actual_seq_lengths_query: Optional[torch.Tensor] = None,
    actual_seq_lengths_key: Optional[torch.Tensor] = None, layout_query: str = "BSND", sparse_mode: int = 3,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    tiling = check_param_valid(
        query, key, query_index, key_index, weights, sparse_indices, softmax_max, softmax_sum,
        sparse_mode=sparse_mode, layout_query=layout_query,
    )
    if layout_query == "TND":
        return sparse_lightning_indexer_grad_kl_loss_tnd(
            query, key, query_index, key_index, weights, sparse_indices, softmax_max, softmax_sum, scale_value,
            query_rope=query_rope, key_rope=key_rope, actual_seq_lengths_query=actual_seq_lengths_query,
            actual_seq_lengths_key=actual_seq_lengths_key, sparse_mode=sparse_mode,
        )
    B, S1, S2, N1, N2, G = (
        tiling.B, tiling.S1, tiling.S2, tiling.N1, tiling.N2, tiling.G,
    )
    Nidx1, Nidx2, DQuery, DQueryIndex, K = (
        tiling.Nidx1, tiling.Nidx2, tiling.DQuery, tiling.DQueryIndex, tiling.K,
    )
    source_s2 = S2
    sparse_k_stride = K
    input_dtype = query.dtype
    calc_dtype_str = "float"
    input_dtype_str = "float16" if input_dtype == torch.float16 else "bfloat16"
    weights_orig_dtype = weights.dtype
    if layout_query == "BSND" and sparse_mode in (3,):
        full_valid = _is_static_full_valid(
            actual_seq_lengths_query, actual_seq_lengths_key, B, S1, S2, K,
        )
        k_scatter_exec = K
        if full_valid:
            k_real_size, valid_k_mask = _get_full_valid_aux_inputs(
                query.device, torch.float32, B, S1, K
            )
        else:
            k_exec, k_scatter_exec = _derive_execution_widths(
                actual_seq_lengths_query, actual_seq_lengths_key, B, S1, S2, K,
            )
            top_k = K
            K = k_exec
            host_query_lengths = _host_sequence_lengths(
                actual_seq_lengths_query, S1, B
            )
            host_key_lengths = _host_sequence_lengths(
                actual_seq_lengths_key, S2, B
            )
            if host_query_lengths is not None and host_key_lengths is not None:
                k_real_size, valid_k_mask = _get_tail_aux_inputs(
                    query.device, torch.float32, host_query_lengths, host_key_lengths, B, S1, top_k, K,
                )
            else:
                if actual_seq_lengths_query is not None:
                    actual_q_len = actual_seq_lengths_query.to(
                        device=query.device, dtype=torch.int32
                    )
                else:
                    actual_q_len = torch.full(
                        (B,), S1, dtype=torch.int32, device=query.device
                    )
                if actual_seq_lengths_key is not None:
                    actual_k_len = actual_seq_lengths_key.to(
                        device=query.device, dtype=torch.int32
                    )
                else:
                    actual_k_len = torch.full(
                        (B,), S2, dtype=torch.int32, device=query.device
                    )
                query_positions = torch.arange(
                    S1, dtype=torch.int32, device=query.device
                )
                k_real_size = (
                    actual_k_len.unsqueeze(1)
                    - actual_q_len.unsqueeze(1)
                    + query_positions
                    + 1
                ).clamp_(min=0, max=top_k).contiguous()
                valid_k_mask = (
                    torch.arange(K, dtype=torch.int32, device=query.device).view(
                        1, 1, K
                    )
                    < k_real_size.unsqueeze(-1)
                ).to(torch.float32).contiguous()
        S2_pad = (S2 + VECTOR_BASEN - 1) // VECTOR_BASEN * VECTOR_BASEN
        if weights.dtype == torch.float32:
            weights = weights.to(input_dtype)
        d_rope_orig = query_rope.shape[-1] if query_rope is not None else 0
        # Gather 已移入内核的 V0 阶段，不再需要主机侧预处理。
        v0_zero_row = _get_v0_zero_row(query.device, input_dtype)
        d_rope = ((d_rope_orig + BLOCK_K - 1) // BLOCK_K) * BLOCK_K if d_rope_orig > 0 else 0
        if query_rope is None:
            query_rope = torch.zeros(B, S1, N1, 0, dtype=input_dtype, device=query.device)
        if key_rope is None:
            key_rope = torch.zeros(B, source_s2, N2, 0, dtype=input_dtype, device=query.device)
        if d_rope > d_rope_orig and d_rope_orig > 0:
            pad_len = d_rope - d_rope_orig
            query_rope = torch.nn.functional.pad(query_rope, (0, pad_len))
            key_rope = torch.nn.functional.pad(key_rope, (0, pad_len))
        hardware_core_count = _get_aicore_count()
        schedule = _make_task_schedule(
            B, S1, S2_pad, K, hardware_core_count,
        )
        core_count = schedule.launch_core_num
        block_n = N_SPLIT
        n_num_ws = K // block_n
        v2_k_padded = K
        loss_partial_stride = 8
        loss_partial_size = ceil_div(
            B * S1 * 2 * loss_partial_stride, 128
        ) * 128
        cache_key = (
            B, S1, S2_pad, source_s2, N1, N2, G, Nidx1, Nidx2,
            DQuery, DQueryIndex, K, sparse_k_stride, k_scatter_exec, core_count, d_rope,
            input_dtype_str, calc_dtype_str, scale_value, block_n, full_valid,
        )
        func = _kernel_cache.get(cache_key)
        if func is None:
            func = tilelang_grad_kl_loss_single_kernel(
                B, S1, S2_pad, N1, N2, G, Nidx1, Nidx2, DQuery, DQueryIndex, K,
                core_count, d_rope, input_dtype_str, calc_dtype_str, scale_value, block_n,
                full_valid, k_scatter_exec, source_s2, sparse_k_stride,
            )
            _kernel_cache[cache_key] = func
        ws_key = (
            B, S1, S2_pad, N1, Nidx1, K, DQueryIndex, d_rope, input_dtype, query.device, n_num_ws, v2_k_padded,
            loss_partial_size,
        )
        workspace = _workspace_cache.get(ws_key)
        if workspace is not None:
            d_query_index_accum, d_key_index_accum, \
                gather_key_ws, gather_key_rope_ws, gather_key_index_ws, \
                bmm1_res_ws, bmm2_res_ws, p_reduced_ws, softmax_i_ws, \
                dk_matmul_partial, relu_grad_out, loss_partial_ws, \
                c2_zero_pad = workspace
        else:
            gather_key_ws = torch.zeros(B, S1, K, DQuery, dtype=input_dtype, device=query.device)
            gather_key_rope_ws = torch.zeros(B, S1, K, d_rope, dtype=input_dtype, device=query.device)
            gather_key_index_ws = torch.zeros(B, S1, K, DQueryIndex, dtype=input_dtype, device=query.device)
            bmm1_res_ws = torch.zeros(B, S1, N1, K, dtype=torch.float32, device=query.device)
            bmm2_res_ws = torch.zeros(B, S1, Nidx1, K, dtype=torch.float32, device=query.device)
            p_reduced_ws = torch.zeros(B, S1, K, dtype=torch.float32, device=query.device)
            softmax_i_ws = torch.zeros(B, S1, K, dtype=torch.float32, device=query.device)
            d_query_index_accum = torch.zeros(B, S1, Nidx1, DQueryIndex, dtype=torch.float32, device=query.device)
            dk_matmul_partial = torch.zeros(
                B, S1, v2_k_padded, DQueryIndex, dtype=torch.float32, device=query.device,
            )
            d_key_index_accum = torch.empty(
                B * S2_pad, DQueryIndex, dtype=torch.float32, device=query.device,
            )
            relu_grad_out = torch.zeros(B, S1, Nidx1, K, dtype=input_dtype, device=query.device)
            loss_partial_ws = torch.zeros(
                loss_partial_size, dtype=torch.float32, device=query.device
            )
            c2_zero_pad = torch.zeros(
                8, DQueryIndex, dtype=input_dtype, device=query.device
            )
            _workspace_cache[ws_key] = (
                d_query_index_accum, d_key_index_accum, gather_key_ws, gather_key_rope_ws, gather_key_index_ws,
                bmm1_res_ws, bmm2_res_ws, p_reduced_ws, softmax_i_ws,
                dk_matmul_partial, relu_grad_out, loss_partial_ws, c2_zero_pad,
            )
        d_query_index, d_key_index_flat, d_weight, loss = func(
            sparse_indices,                             # 0
            k_real_size, valid_k_mask,                  # 1-2
            v0_zero_row,                                # 3
            query, query_rope, key, key_rope,           # 4-7
            query_index, key_index,                     # 8-9
            weights, softmax_max, softmax_sum,          # 10-12
            d_query_index_accum,                        # 13，工作区
            d_key_index_accum,                          # 14，工作区
            gather_key_ws, gather_key_rope_ws, gather_key_index_ws,  # 15-17
            bmm1_res_ws, bmm2_res_ws,                   # 18-19
            p_reduced_ws, softmax_i_ws,                 # 20-21
            dk_matmul_partial, relu_grad_out,           # 22-23
            loss_partial_ws, c2_zero_pad,               # 24-25
        )
        if weights_orig_dtype == torch.float32:
            d_weight = d_weight.float()
        d_key_index = d_key_index_flat.reshape(
            B, S2_pad, Nidx2, DQueryIndex
        )[:, :S2]
        return d_query_index, d_key_index, d_weight, loss
    raise ValueError(f"Unsupported layout_query={layout_query} with sparse_mode={sparse_mode}")

def sparse_lightning_indexer_grad_kl_loss_tnd(
    query: torch.Tensor, key: torch.Tensor, query_index: torch.Tensor, key_index: torch.Tensor,
    weights: torch.Tensor, sparse_indices: torch.Tensor, softmax_max: torch.Tensor,
    softmax_sum: torch.Tensor, scale_value: float, *,
    query_rope: Optional[torch.Tensor] = None, key_rope: Optional[torch.Tensor] = None,
    actual_seq_lengths_query: Optional[torch.Tensor] = None,
    actual_seq_lengths_key: Optional[torch.Tensor] = None, sparse_mode: int = 3,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    (
        query_b, key_b, query_index_b, key_index_b, weights_b,
        sparse_indices_b, softmax_max_b, softmax_sum_b,
    ) = _add_batch_dim(
        query, key, query_index, key_index, weights,
        sparse_indices, softmax_max, softmax_sum,
    )
    actual_q_len = _length_tensor(actual_seq_lengths_query, query.shape[0])
    actual_k_len = _length_tensor(actual_seq_lengths_key, key.shape[0])
    d_qi, d_ki, dw, loss = sparse_lightning_indexer_grad_kl_loss(
        query_b, key_b, query_index_b, key_index_b, weights_b, sparse_indices_b,
        softmax_max_b, softmax_sum_b, scale_value,
        query_rope=query_rope.unsqueeze(0) if query_rope is not None else None,
        key_rope=key_rope.unsqueeze(0) if key_rope is not None else None, actual_seq_lengths_query=actual_q_len,
        actual_seq_lengths_key=actual_k_len, layout_query="BSND", sparse_mode=sparse_mode,
    )
    return d_qi.squeeze(0), d_ki.squeeze(0), dw.squeeze(0), loss
