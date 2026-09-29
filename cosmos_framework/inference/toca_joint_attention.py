"""Joint GEN attention and ToCa incoming score: QK is evaluated exactly once.

Inference-only, noncausal GEN queries with complete UND+GEN keys and real GQA.
The Flash-style online softmax produces AV while retaining FP32 logits. A
column reduction reuses those logits and row normalizers, never recomputing QK.
This deliberately trades bounded scratch memory for the official shared-QK
semantics; native Dense and cached-step attention kernels remain untouched.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _joint_forward(
    Q,
    K,
    V,
    O,
    Logits,
    Lse,
    NQ: tl.constexpr,
    NK: tl.constexpr,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    D: tl.constexpr,
    SQ0: tl.constexpr,
    SQ1: tl.constexpr,
    SQ2: tl.constexpr,
    SK0: tl.constexpr,
    SK1: tl.constexpr,
    SK2: tl.constexpr,
    SV0: tl.constexpr,
    SV1: tl.constexpr,
    SV2: tl.constexpr,
    SCALE: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BD: tl.constexpr,
):
    qi = tl.program_id(0) * BM + tl.arange(0, BM)
    h = tl.program_id(1)
    kh = h // (HQ // HK)
    di = tl.arange(0, BD)
    q = tl.load(Q + qi[:, None] * SQ0 + h * SQ1 + di[None, :] * SQ2, (qi[:, None] < NQ) & (di[None, :] < D), other=0)
    m = tl.full((BM,), -float("inf"), tl.float32)
    z = tl.full((BM,), 0, tl.float32)
    acc = tl.full((BM, BD), 0, tl.float32)
    for start in range(tl.cdiv(NK, BN)):
        ki = start * BN + tl.arange(0, BN)
        k = tl.load(
            K + ki[None, :] * SK0 + kh * SK1 + di[:, None] * SK2, (ki[None, :] < NK) & (di[:, None] < D), other=0
        )
        logits = tl.dot(q, k).to(tl.float32) * (SCALE * 1.4426950408889634)
        logits = tl.where(ki[None, :] < NK, logits, -float("inf"))
        tl.store(Logits + (h * NQ + qi[:, None]) * NK + ki[None, :], logits, (qi[:, None] < NQ) & (ki[None, :] < NK))
        new_m = tl.maximum(m, tl.max(logits, 1))
        alpha = tl.exp2(m - new_m)
        p = tl.exp2(logits - new_m[:, None])
        z = z * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]
        v = tl.load(
            V + ki[:, None] * SV0 + kh * SV1 + di[None, :] * SV2, (ki[:, None] < NK) & (di[None, :] < D), other=0
        )
        acc = tl.dot(p.to(v.dtype), v, acc)
        m = new_m
    acc = acc / z[:, None]
    tl.store(O + (qi[:, None] * HQ + h) * D + di[None, :], acc, (qi[:, None] < NQ) & (di[None, :] < D))
    tl.store(Lse + h * NQ + qi, m + tl.log2(z), qi < NQ)


@triton.jit
def _column_partials(
    Logits,
    Lse,
    Future,
    Partial,
    NQ: tl.constexpr,
    NK: tl.constexpr,
    NF: tl.constexpr,
    NU: tl.constexpr,
    NPART: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
):
    fi = tl.program_id(0) * BN + tl.arange(0, BN)
    part = tl.program_id(1)
    h = tl.program_id(2)
    qi = part * BM + tl.arange(0, BM)
    ki = tl.load(Future + fi, fi < NF, other=0) + NU
    lse = tl.load(Lse + h * NQ + qi, qi < NQ, other=0)
    logits = tl.load(
        Logits + (h * NQ + qi[:, None]) * NK + ki[None, :], (qi[:, None] < NQ) & (fi[None, :] < NF), other=-float("inf")
    )
    mass = tl.sum(tl.exp2(logits - lse[:, None]), 0)
    tl.store(Partial + (h * NPART + part) * NF + fi, mass, fi < NF)


@triton.jit
def _finish_columns(
    Partial,
    Score,
    NF: tl.constexpr,
    NPART: tl.constexpr,
    HQ: tl.constexpr,
    NQ: tl.constexpr,
    BR: tl.constexpr,
    BN: tl.constexpr,
):
    fi = tl.program_id(0) * BN + tl.arange(0, BN)
    ri = tl.arange(0, BR)
    values = tl.load(Partial + ri[:, None] * NF + fi[None, :], (ri[:, None] < HQ * NPART) & (fi[None, :] < NF), other=0)
    score = tl.sum(values, 0) / (HQ * NQ)
    tl.store(Score + fi, score, fi < NF)


def joint_attention_score(q, k_und, k_gen, v_und, v_gen, future_positions, scale, *, block_m=32, block_n=64):
    """Return [Ngen,Hq,D] native-dtype AV and [Nfuture] FP32 score.

    The denominator includes all keys; only the requested key columns are
    reduced. No query sampling, token relabeling, dropout or head averaging
    before softmax. Logits/row sums and score accumulation remain FP32.
    """
    if not q.is_cuda or q.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("Joint ToCa kernel requires CUDA BF16/FP16 inputs")
    nq, hq, d = q.shape
    nu, hk, kd = k_und.shape
    if (
        k_gen.shape != (nq, hk, d)
        or kd != d
        or hq % hk
        or d not in (32, 64, 128)
        or v_und.shape != k_und.shape
        or v_gen.shape != k_gen.shape
        or any(x.dtype != q.dtype or x.device != q.device for x in (k_und, k_gen, v_und, v_gen))
    ):
        raise ValueError("Unsupported Q/K/V shape, dtype or GQA mapping")
    if future_positions.ndim != 1 or future_positions.device != q.device or future_positions.dtype != torch.long:
        raise ValueError("Future indices must be a CUDA int64 vector")
    k, v = torch.cat((k_und, k_gen)), torch.cat((v_und, v_gen))
    nk, nf = nq + nu, future_positions.numel()
    logits = torch.empty((hq, nq, nk), dtype=torch.float32, device=q.device)
    lse = torch.empty((hq, nq), dtype=torch.float32, device=q.device)
    out = torch.empty((nq, hq, d), dtype=q.dtype, device=q.device)
    _joint_forward[(triton.cdiv(nq, block_m), hq)](
        q,
        k,
        v,
        out,
        logits,
        lse,
        nq,
        nk,
        hq,
        hk,
        d,
        *q.stride(),
        *k.stride(),
        *v.stride(),
        float(scale),
        block_m,
        block_n,
        triton.next_power_of_2(d),
        num_warps=4,
        num_stages=2,
    )
    npart = triton.cdiv(nq, 128)
    partial = torch.empty((hq * npart, nf), dtype=torch.float32, device=q.device)
    score = torch.empty(nf, dtype=torch.float32, device=q.device)
    _column_partials[(triton.cdiv(nf, 32), npart, hq)](
        logits,
        lse,
        future_positions,
        partial,
        nq,
        nk,
        nf,
        nu,
        npart,
        128,
        32,
        num_warps=4,
    )
    _finish_columns[(triton.cdiv(nf, 16),)](
        partial,
        score,
        nf,
        npart,
        hq,
        nq,
        triton.next_power_of_2(hq * npart),
        16,
        num_warps=4,
    )
    return out, score
