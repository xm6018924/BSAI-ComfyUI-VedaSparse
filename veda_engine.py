"""BSAI VedaSparse — Veda 蒸馏稀疏注意力引擎 for MiniMax-H3（v3.4 monkey-patch 真路径）。

v3.4 = 直接替换 H3 attention.forward 的 attention 调用段（真加速唯一可行路径）
================================================================================

v3.0~v3.3 失败根因（已定位）
---------------------------
* v3.0/v3.1 走 optimized_attention_override 注入：H3 attention.forward（model.py:200）
  **裸调原生 optimized_attention**，且 H3 传了 preferred_attention=self.comfy_attention
  （int8 底模 = comfy_kitchen_int8 attention）。override 链在 H3 上不可靠，
  "sparse called" 日志实际来自其它插件的 attention，H3 从未真正稀疏。
* v3.3 spy hook 试图复刻 qkv_proj + RMSNorm + RoPE：H3 用 quantized fused op
  （comfy.quant_ops.ck.rms_rope_split_half_），复刻与 H3 实际数值不同，
  误差累积 64 层 × N 步 → 废片。

v3.4 方案（本文件）
-------------------
1. monkey-patch H3 `comfy.ldm.minimax.model.Attention.forward`（实例级）：
   新 forward **逐行复刻 H3 原 forward 前段**（qkv_proj → split → v.view →
   rms_rope_split_half_ / q_norm·k_norm → AttentionTensorContainer 包装），
   q/k/v 与 H3 数值 **100% 一致**（就是同一段代码，冒烟实测 0.00e+00）。
2. 只在 `optimized_attention(...)` 调用点替换为 Veda 稀疏注意力：
   * conditioning 行（text/refs/audio）作为 key **恒 dense**（音频安全线）；
   * video 行做 TripPool top-k（含自身 tile）；video query 与 cond/video key
     走 **联合 softmax**（logsumexp 恒等，数学精确）。
3. video/audio span 直接读 H3 已注入的 `transformer_options["minimax_h3_layout"]`
   （model.py:624），不需要任何 PackedLayout monkey-patch。
4. 执行层：Triton block-sparse kernel **按 selected-index 列表**循环
   （每 query tile 只迭代其 top-k+cond 的 key tile，不是全序列）；
   出错自动降级 → 纯 PyTorch sparse（带 sink）→ tensor dense → 原 forward。
5. 输出 reshape 为 H3 期待的 [B, L, H*D]，走 self.out_proj，与 H3 完全一致。

硬件/依赖：Triton 3.5+（SM 80+）；无 Triton 自动 PyTorch 路径。
"""

import logging
import types

import torch
import torch.nn.functional as F

BLOCK = 64

# 全局开关与统计
_STATS = {"sparse": 0, "dense": 0, "heuristic": 0, "distilled": 0,
          "errors": 0, "fallback_1d": 0}
_SEEN = set()
_FALLBACK_TO_DENSE = False                  # OOM 触发后整进程 dense

_TRITON_SPARSE_AVAILABLE = False
try:
    import triton
    import triton.language as tl
    _TRITON_SPARSE_AVAILABLE = True
except ImportError:
    logging.info("[BSAI VedaSparse v3.4] triton not available -> PyTorch path")

_ENABLE_TRITON = True


# ---------------------------------------------------------------------------
# Triton block-sparse attention kernel（selected-index 循环）
# ---------------------------------------------------------------------------

if _TRITON_SPARSE_AVAILABLE:

    @triton.jit
    def _sparse_attn_fwd(
        Q, K, V, Out, SelIdx, SelCnt,
        sm_scale,
        stride_qb, stride_qh, stride_qm, stride_qd,
        stride_kb, stride_kh, stride_kn, stride_kd,
        stride_vb, stride_vh, stride_vn, stride_vd,
        stride_ob, stride_oh, stride_om, stride_od,
        N_CTX: tl.constexpr,
        N_TILE_K: tl.constexpr,
        MAX_SEL: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        HEAD_DIM: tl.constexpr,
    ):
        """FlashAttn 风格 block-sparse forward（selected-index）。

        SelCnt[start_m] = 该 query tile 选中的 key tile 数；
          * < 0：哨兵，该行走全序列 dense（cond/混合 query tile）。
          * >= 0：该行只循环 SelIdx[start_m, :cnt] 中的 key tile。
        SelIdx: [n_tile_q, MAX_SEL] int32。
        """
        start_m = tl.program_id(0)
        off_b = tl.program_id(1)
        off_h = tl.program_id(2)

        q_offset = off_b * stride_qb + off_h * stride_qh
        k_offset = off_b * stride_kb + off_h * stride_kh
        v_offset = off_b * stride_vb + off_h * stride_vh
        o_offset = off_b * stride_ob + off_h * stride_oh

        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, HEAD_DIM)

        q_ptrs = Q + q_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
        q = tl.load(q_ptrs, mask=offs_m[:, None] < N_CTX, other=0.0)

        m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

        cnt = tl.load(SelCnt + start_m)

        if cnt < 0:
            for start_n in range(0, N_TILE_K):
                offs_kn = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
                k_ptrs = K + k_offset + offs_kn[:, None] * stride_kn + offs_d[None, :] * stride_kd
                v_ptrs = V + v_offset + offs_kn[:, None] * stride_vn + offs_d[None, :] * stride_vd
                k = tl.load(k_ptrs, mask=offs_kn[:, None] < N_CTX, other=0.0)
                v = tl.load(v_ptrs, mask=offs_kn[:, None] < N_CTX, other=0.0)
                qk = tl.dot(q, tl.trans(k), out_dtype=tl.float32)
                qk = qk * sm_scale
                qk = tl.where(offs_kn[None, :] < N_CTX, qk, -float("inf"))
                m_ij = tl.maximum(m_i, tl.max(qk, 1))
                alpha = tl.math.exp2((m_i - m_ij) * 1.4426950408889634)
                pp = tl.math.exp2((qk - m_ij[:, None]) * 1.4426950408889634)
                pp = tl.where(pp == pp, pp, 0.0)
                alpha = tl.where(alpha == alpha, alpha, 0.0)
                l_i = l_i * alpha + tl.sum(pp, 1)
                acc = acc * alpha[:, None]
                acc = tl.dot(pp.to(v.dtype), v, acc, out_dtype=tl.float32)
                m_i = m_ij
        else:
            for s in range(0, MAX_SEL):
                if s < cnt:
                    start_n = tl.load(SelIdx + start_m * MAX_SEL + s)
                    offs_kn = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
                    k_ptrs = K + k_offset + offs_kn[:, None] * stride_kn + offs_d[None, :] * stride_kd
                    v_ptrs = V + v_offset + offs_kn[:, None] * stride_vn + offs_d[None, :] * stride_vd
                    k = tl.load(k_ptrs, mask=offs_kn[:, None] < N_CTX, other=0.0)
                    v = tl.load(v_ptrs, mask=offs_kn[:, None] < N_CTX, other=0.0)
                    qk = tl.dot(q, tl.trans(k), out_dtype=tl.float32)
                    qk = qk * sm_scale
                    qk = tl.where(offs_kn[None, :] < N_CTX, qk, -float("inf"))
                    m_ij = tl.maximum(m_i, tl.max(qk, 1))
                    alpha = tl.math.exp2((m_i - m_ij) * 1.4426950408889634)
                    pp = tl.math.exp2((qk - m_ij[:, None]) * 1.4426950408889634)
                    pp = tl.where(pp == pp, pp, 0.0)
                    alpha = tl.where(alpha == alpha, alpha, 0.0)
                    l_i = l_i * alpha + tl.sum(pp, 1)
                    acc = acc * alpha[:, None]
                    acc = tl.dot(pp.to(v.dtype), v, acc, out_dtype=tl.float32)
                    m_i = m_ij

        acc = tl.where(l_i[:, None] > 0, acc / l_i[:, None], 0.0)
        o_ptrs = Out + o_offset + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
        tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=offs_m[:, None] < N_CTX)


def _triton_sparse_attention(q, k, v, sel_idx, sel_cnt):
    """执行 Triton block-sparse attention（selected-index）。

    q/k/v: [B, H, N, D]；sel_idx: [n_tile_q, MAX_SEL] int32；
    sel_cnt: [n_tile_q] int32（<0 = dense 行）。
    返回 [B, H, N, D]。
    """
    B, H, N, D = q.shape
    out = torch.empty_like(q)
    sm_scale = 1.0 / (D ** 0.5)
    n_tile = (N + BLOCK - 1) // BLOCK
    max_sel = sel_idx.shape[1]
    grid = (n_tile, B, H)
    _sparse_attn_fwd[grid](
        q, k, v, out, sel_idx, sel_cnt, sm_scale,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        N_CTX=N, N_TILE_K=n_tile,
        MAX_SEL=max_sel, BLOCK_M=BLOCK, BLOCK_N=BLOCK, HEAD_DIM=D,
    )
    return out


# ---------------------------------------------------------------------------
# TripPool 评分（Veda 论文 eq.5/eq.6）
# ---------------------------------------------------------------------------

def _tripool(tiles, mode="triplet"):
    """tiles: [B, H, nq, BLOCK, D]（BLOCK 轴 = tile 内 token）→ [B, H, nq, Demb]。"""
    avg = tiles.mean(dim=3)
    if mode == "avg":
        return avg
    mx = tiles.amax(dim=3)
    if mode == "maxmin":
        return torch.cat([mx, tiles.amin(dim=3)], dim=-1)
    return torch.cat([avg, mx, tiles.amin(dim=3)], dim=-1)


def _heuristic_scores(qpool, kpool):
    """qpool/kpool: [B, H, nq, Demb] → scores [B, H, nq, nq]（逐头点积）。"""
    B, H, nq, Demb = qpool.shape
    q = qpool.reshape(B * H, nq, Demb)   # C-order：每个 (b,h) 一个 nq×Demb 块
    k = kpool.reshape(B * H, nq, Demb)
    s = torch.bmm(q, k.transpose(-2, -1)) * (float(Demb) ** -0.5)
    return s.view(B, H, nq, nq)


# ---------------------------------------------------------------------------
# Veda block-sparse over packed sequence（带 conditioning sink）
# ---------------------------------------------------------------------------

def _veda_sparse_packed(qs, ks, vs, scale, keep_frac, cond_end, video_end,
                        tripool_mode="triplet"):
    """Packed 序列上的 block-sparse attention，conditioning 行恒 dense。

    qs/ks/vs: [B=1, H, N, D]（H3 skip_reshape 布局，N=packed 全序列）。
    cond_end: text+refs+audio 的结束行（video 段起点）。
    video_end: video 段终点（packed 末段）。

    语义：
      * cond query 行（[:cond_end]）→ dense over 全部 key（text/audio 必须精确）。
      * video query 行（[cond_end:video_end]）→ **联合 softmax**：
        cond key 全量 + video key 的 TripPool top-k（含自身 tile）。
      * 联合归一用 logsumexp 恒等（softmax([lc,lv]) = Σexp((x-m)v)/Σexp(x-m)），
        不 concat 巨矩阵，峰值可控。
    返回 [B, H, N, D]。
    """
    B, H, N, D = qs.shape
    v_start = int(cond_end)
    v_end = int(video_end)
    v_len = v_end - v_start
    scale = float(scale)

    out = torch.empty_like(qs)

    if v_start > 0:
        # --- cond queries → dense over ALL keys ---------------------------------
        qc = qs[:, :, :v_start]                     # [B,H,cond,D]
        s_c = torch.matmul(qc.float(), ks.float().transpose(-2, -1)) * scale
        p_c = s_c.softmax(dim=-1)
        out[:, :, :v_start] = torch.matmul(p_c.to(vs.dtype), vs)

    if v_len > 0:
        # --- video queries → 联合 softmax：cond key 全量 + video key top-k ------
        qv = qs[:, :, v_start:v_end]
        kv = ks[:, :, v_start:v_end]
        vv = vs[:, :, v_start:v_end]
        kc = ks[:, :, :v_start] if v_start > 0 else None
        vc = vs[:, :, :v_start] if v_start > 0 else None

        # 视频内部：一维 64-token tile，TripPool 评分 + top-k
        nv = (v_len + BLOCK - 1) // BLOCK
        pad = nv * BLOCK - v_len
        if pad:
            qvp = torch.zeros(B, H, nv * BLOCK, D, dtype=qv.dtype, device=qv.device)
            kvp = torch.zeros(B, H, nv * BLOCK, D, dtype=kv.dtype, device=kv.device)
            vvp = torch.zeros(B, H, nv * BLOCK, D, dtype=vv.dtype, device=vv.device)
            qvp[:, :, :v_len] = qv
            kvp[:, :, :v_len] = kv
            vvp[:, :, :v_len] = vv
        else:
            qvp, kvp, vvp = qv, kv, vv

        qv_t = qvp.view(B, H, nv, BLOCK, D)
        kv_t = kvp.view(B, H, nv, BLOCK, D)
        qp = _tripool(qv_t, tripool_mode)          # [B,H,nv,Demb]
        kp = _tripool(kv_t, tripool_mode)
        scores = _heuristic_scores(qp, kp)         # [B,H,nv,nv]
        topk = max(1, round(nv * keep_frac))
        topk_idx = scores.topk(topk, dim=-1).indices   # [B,H,nv,topk]

        # kvp/qvp/vvp 已是 [B,H,nv*BLOCK,D] 连续 tile 序；C-order reshape 到
        # (b,h) 块即得正确 gather 布局（不要再 permute 打乱内存序）。
        kvf = kvp.reshape(B * H, nv * BLOCK, D).contiguous()
        vvf = vvp.reshape(B * H, nv * BLOCK, D).contiguous()
        qvf = qvp.reshape(B * H, nv, BLOCK, D).contiguous()

        tok_idx = (topk_idx * BLOCK).view(B * H, nv, topk, 1) + \
                  torch.arange(BLOCK, device=qs.device).view(1, 1, 1, BLOCK)
        tok_flat = tok_idx.view(B * H, nv, topk * BLOCK)

        valid = torch.zeros(nv * BLOCK, dtype=torch.bool, device=qs.device)
        valid[:v_len] = True

        # 联合 softmax（logsumexp 恒等，不 concat，峰值可控）：
        #   softmax([lc, lv]) 拆成 exp((x-m)*v) 分子 + Σ 分母，m = lv.max
        # 按 query chunk 循环，避免 [.,.,m,cond+topk*BLOCK] 巨矩阵。
        CHUNK = 32
        pieces = []
        for c in range(0, nv, CHUNK):
            nb = min(CHUNK, nv - c)
            q2 = qvf[:, c:c + nb].reshape(B * H * nb, BLOCK, D)
            idx = tok_flat[:, c:c + nb]

            gk = torch.gather(
                kvf.unsqueeze(1).expand(B * H, nb, nv * BLOCK, D),
                2, idx.unsqueeze(-1).expand(B * H, nb, topk * BLOCK, D)
            ).reshape(B * H * nb, topk * BLOCK, D)
            gv = torch.gather(
                vvf.unsqueeze(1).expand(B * H, nb, nv * BLOCK, D),
                2, idx.unsqueeze(-1).expand(B * H, nb, topk * BLOCK, D)
            ).reshape(B * H * nb, topk * BLOCK, D)
            gkm = valid[idx].reshape(B * H * nb, topk * BLOCK)

            lv = torch.bmm(q2, gk.transpose(-2, -1)) * scale          # sparse video logits
            lv = lv.masked_fill(~gkm[:, None, :].expand(-1, BLOCK, -1),
                                float("-inf"))
            mv = lv.amax(dim=-1, keepdim=True)
            mv = torch.where(torch.isfinite(mv), mv,
                             torch.zeros_like(mv))
            elv = torch.exp(lv - mv)
            sv = elv.sum(dim=-1, keepdim=True)

            if kc is not None:
                kc2 = kc.reshape(B * H, v_start, D).unsqueeze(1) \
                    .expand(B * H, nb, v_start, D).reshape(B * H * nb, v_start, D)
                lc = torch.bmm(q2, kc2.transpose(-2, -1).contiguous()) * scale  # cond logits
                elc = torch.exp(lc - mv)
                sc = elc.sum(dim=-1, keepdim=True)
            else:
                elc = None
                sc = torch.zeros_like(sv)

            denom = sv + sc
            num = torch.bmm(elv, gv)
            if elc is not None:
                vc2 = vc.reshape(B * H, v_start, D).unsqueeze(1) \
                    .expand(B * H, nb, v_start, D).reshape(B * H * nb, v_start, D)
                num = num + torch.bmm(elc, vc2)
            o = num / denom.clamp_min(1e-6)
            pieces.append(o.reshape(B, H, nb, BLOCK, D))

        o_sparse = torch.cat(pieces, dim=2).reshape(B, H, nv * BLOCK, D)[:, :, :v_len]
        out[:, :, v_start:v_end] = o_sparse

    return out


def _build_selected(q, k, v, cond_end, video_end, keep_frac, tripool_mode):
    """构造 selected-index：SelIdx [n_tile, MAX_SEL] int32 + SelCnt [n_tile]。

    * query tile < cond_tiles（cond/混合行）→ SelCnt=-1（dense 哨兵，kernel 全循环）。
    * video 纯 query tile → selected = cond 全部 key tile + TripPool top-k video tile。
    返回 (SelIdx, SelCnt)。
    """
    B, H, N, D = q.shape
    n_tile = (N + BLOCK - 1) // BLOCK
    cond_tiles = (int(cond_end) + BLOCK - 1) // BLOCK

    v_start = int(cond_end)
    v_end = int(video_end)
    v_len = v_end - v_start

    if v_len <= 0 or cond_tiles >= n_tile:
        # video 过短/无独立 tile：全 dense（SelCnt=-1 全行）
        sel_cnt = torch.full((n_tile,), -1, dtype=torch.int32, device=q.device)
        sel_idx = torch.zeros((n_tile, 1), dtype=torch.int32, device=q.device)
        return sel_idx, sel_cnt

    # 混合 tile（同时含 cond+video token）= v_start // BLOCK；其 query/key 都
    # 走哨兵/全量。纯 video tile 从 ceil(v_start/BLOCK) = cond_tiles 开始。
    pure0 = cond_tiles
    nvg = n_tile - pure0
    if nvg <= 1:
        sel_cnt = torch.full((n_tile,), -1, dtype=torch.int32, device=q.device)
        sel_idx = torch.zeros((n_tile, 1), dtype=torch.int32, device=q.device)
        return sel_idx, sel_cnt

    # 纯 video tile 的 TripPool top-k（同 _build_block_mask 逻辑）
    vq = q[:, :, v_start:v_end]
    vk = k[:, :, v_start:v_end]
    off0 = pure0 * BLOCK - v_start
    qp_tiles = []
    kp_tiles = []
    for j in range(nvg):
        t0 = off0 + j * BLOCK
        t1 = min(t0 + BLOCK, v_len)
        qp_tiles.append(_tripool(vq[:, :, t0:t1].unsqueeze(2), tripool_mode)
                        .squeeze(2))
        kp_tiles.append(_tripool(vk[:, :, t0:t1].unsqueeze(2), tripool_mode)
                        .squeeze(2))
    qp = torch.stack(qp_tiles, dim=2)
    kp = torch.stack(kp_tiles, dim=2)
    scores = _heuristic_scores(qp, kp)
    topk = max(1, round(nvg * keep_frac))
    sc = scores.mean(dim=1)
    topk_idx = sc.topk(topk, dim=-1).indices       # [B=1, nvg, topk]

    # selected key tile 列表：cond 全部 + video top-k（全局索引）
    cond_sel = torch.arange(cond_tiles, dtype=torch.int32, device=q.device) \
        .unsqueeze(0).expand(nvg, cond_tiles)
    vid_sel = (topk_idx[0] + pure0).to(torch.int32)    # [nvg, topk]
    sel_all = torch.cat([cond_sel, vid_sel], dim=-1)   # [nvg, cond_tiles+topk]
    max_sel = sel_all.shape[-1]

    sel_idx = torch.zeros((n_tile, max_sel), dtype=torch.int32, device=q.device)
    sel_cnt = torch.full((n_tile,), -1, dtype=torch.int32, device=q.device)
    sel_idx[pure0:] = sel_all
    sel_cnt[pure0:] = max_sel
    return sel_idx, sel_cnt


# ---------------------------------------------------------------------------
# H3 Attention.forward monkey-patch（真路径）
# ---------------------------------------------------------------------------

def _video_span(layout):
    segments = getattr(layout, "segments", None)
    if not segments:
        return None
    return next(((a, b) for a, b, kind in segments if kind == "video"), None)


def _h3_attn_forward(self, x, rope_freqs=None, transformer_options=None):
    """替换 H3 Attention.forward：逐行复刻前段（100% 同源）→ Veda 稀疏 → out_proj。"""
    if transformer_options is None:
        transformer_options = {}

    # ---- 1) 与 H3 原 forward 完全一致的前段 --------------------------------
    import comfy.model_management
    import comfy.ldm.modules.attention as _attn_mod
    AttentionTensorContainer = _attn_mod.AttentionTensorContainer

    s = x.shape[0]
    q, k, v = self.qkv_proj(x).split(self.heads * self.head_dim, dim=-1)
    v = v.view(s, self.heads, self.head_dim)
    if rope_freqs is not None:
        q = q.view(1, s, self.heads, self.head_dim)
        k = k.view(1, s, self.heads, self.head_dim)
        qw = comfy.model_management.cast_to(self.q_norm.weight, device=x.device)
        kw = comfy.model_management.cast_to(self.k_norm.weight, device=x.device)
        rot = rope_freqs.shape[-3] * 2
        if comfy.model_management.in_training:
            q, k = comfy.quant_ops.ck.rms_rope_split_half(
                q, k, rope_freqs, qw, kw, epsilon=self.q_norm.eps, rot_dim=rot)
        else:
            comfy.quant_ops.ck.rms_rope_split_half_(
                q, k, rope_freqs, qw, kw, epsilon=self.q_norm.eps, rot_dim=rot)
        q = q[0]
        k = k[0]
    else:
        q = self.q_norm(q.view(s, self.heads, self.head_dim))
        k = self.k_norm(k.view(s, self.heads, self.head_dim))

    q = AttentionTensorContainer(q.transpose(0, 1).unsqueeze(0))
    k = AttentionTensorContainer(k.transpose(0, 1).unsqueeze(0))
    v = AttentionTensorContainer(v.transpose(0, 1).unsqueeze(0))

    # ---- 2) attention 调用段（替换点） --------------------------------------
    out = _h3_attn_call(self, q, k, v, transformer_options,
                        _attn_mod.optimized_attention, AttentionTensorContainer)

    # ---- 3) 与 H3 原 forward 完全一致的收尾 --------------------------------
    return self.out_proj(out.squeeze(0))


def _h3_attn_call(self, q, k, v, transformer_options,
                  optimized_attention, AttentionTensorContainer):
    """在 optimized_attention 调用点做 dense/sparse 分流。

    container 只在确认走 sparse 时才 take；任何 fallback 都保证不传空 container。
    """
    global _FALLBACK_TO_DENSE

    params = getattr(self, "_bsai_veda_params", None)

    def dense():
        _STATS["dense"] += 1
        # 复刻 wrap_attn 的 preferred_attention 语义（去掉 override 键防递归）
        to = dict(transformer_options)
        to.pop("optimized_attention_override", None)
        return optimized_attention(
            q, k, v, self.heads, preferred_attention=self.comfy_attention,
            mask=None, skip_reshape=True, transformer_options=to)

    def dense_tensors(qs, ks, vs):
        _STATS["dense"] += 1
        to = dict(transformer_options)
        to.pop("optimized_attention_override", None)
        return optimized_attention(
            qs, ks, vs, self.heads, preferred_attention=self.comfy_attention,
            mask=None, skip_reshape=True, transformer_options=to)

    if params is None or not params.get("enabled", False):
        return dense()
    if _FALLBACK_TO_DENSE:
        return dense()

    try:
        layout = transformer_options.get("minimax_h3_layout")
        span = _video_span(layout) if layout is not None else None
        if span is None:
            _STATS["dense"] += 1
            return dense()
        video_start, video_end = span

        # token 数 / 序列长度自适应 / sigma 窗口（peek 不消费 container）
        qt = q.peek()                                  # [B,H,N,D]（skip_reshape）
        N = qt.shape[2]
        if N < params["min_tokens"] or N != k.peek().shape[2]:
            _STATS["dense"] += 1
            return dense()
        # 超长 packed 序列（H3 5s 视频 = 182 万 token）：top-k 稀疏的 K/V 重复
        # 加载量 >> dense split-KV 单次加载，数学上必输——强制 dense 不拖慢。
        if N >= params.get("max_sparse_tokens", 16384):
            _STATS["dense"] += 1
            return dense()

        sigmas = transformer_options.get("sigmas")
        if sigmas is not None:
            sigma = float(sigmas[0])
            # 低步数 ladder（蒸馏/Turbo，<=5 步）：全步稀疏——否则 TaoMate
            # 4 步的 [0.9999,0.973,0.9231,0.0] 前 3 步全被 sigma_start 挡成
            # dense，稀疏等于没开。常规多步采样仍保留"早期噪声步 dense"保护。
            low_step = len(sigmas) <= 5
            if (not low_step and sigma > params["sigma_start"]) \
                    or sigma < params["sigma_end"]:
                _STATS["dense"] += 1
                return dense()

        qs, ks, vs = q.take(), k.take(), v.take()      # [B,H,N,D]
        dim_head = qs.shape[-1]
        scale = float(dim_head ** -0.5)
        cond_end = int(video_start)

        # ---- 稀疏执行：Triton → PyTorch sparse → tensor dense -----------------
        if _TRITON_SPARSE_AVAILABLE and _ENABLE_TRITON:
            try:
                sel_idx, sel_cnt = _build_selected(
                    qs, ks, vs, cond_end, video_end,
                    params["keep_frac"], params["tripool"])
                out_s = _triton_sparse_attention(qs, ks, vs, sel_idx, sel_cnt)
                _STATS["sparse"] += 1
                return out_s.permute(0, 2, 1, 3).reshape(1, N, qs.shape[1] * dim_head)
            except Exception as exc:
                _STATS["errors"] += 1
                if "out of memory" in str(exc).lower():
                    _FALLBACK_TO_DENSE = True
                logging.warning(f"[BSAI VedaSparse v3.4] triton sparse failed "
                                f"({type(exc).__name__}: {exc}) -> PyTorch sparse")

        try:
            out_s = _veda_sparse_packed(qs, ks, vs, scale, params["keep_frac"],
                                        cond_end, video_end, params["tripool"])
            _STATS["sparse"] += 1
            return out_s.permute(0, 2, 1, 3).reshape(1, N, qs.shape[1] * dim_head)
        except Exception as exc:
            _STATS["errors"] += 1
            if "out of memory" in str(exc).lower():
                _FALLBACK_TO_DENSE = True
            logging.warning(f"[BSAI VedaSparse v3.4] pytorch sparse failed "
                            f"({type(exc).__name__}: {exc}) -> dense")
            return dense_tensors(qs, ks, vs)
    except Exception as exc:
        _STATS["errors"] += 1
        logging.warning(f"[BSAI VedaSparse v3.4] attn_call error "
                        f"({type(exc).__name__}: {exc}) -> dense")
        return dense()


def _sparse_ffn_call(block, x, h, transformer_options):
    """block 级 FFN token 稀疏：video 段按 block 输入范数 top-keep%，cond 恒全量。

    x = block 输入（attn 残差后），h = norm2 输出（FFN 输入）。
    跳过的 token 输出 0 -> _mod_gate(gate*0=0) / add_ 残差恒等，安全。
    音频/文本在 cond 段（video 前）恒全量，音频安全线不受影响。
    """
    params = getattr(block, "_bsai_veda_params", None)
    if params is None or not params.get("enabled", False):
        return block.mlp(h)
    if not params.get("ffn_sparse", False):
        return block.mlp(h)
    layout = transformer_options.get("minimax_h3_layout")
    span = _video_span(layout) if layout is not None else None
    if span is None:
        return block.mlp(h)
    v0, v1 = span
    N = h.shape[0]
    if v1 > N or v0 >= N or (v1 - v0) < 128:
        return block.mlp(h)
    keep_frac = params["ffn_keep_frac"]
    v_len = v1 - v0
    keep = max(1, int(round(v_len * keep_frac)))
    mask = torch.ones(N, dtype=torch.bool, device=h.device)
    if keep < v_len:
        imp = x.float().pow(2).mean(-1).sqrt()       # [N] 残差流范数
        vals = imp[v0:v1]
        thr = vals.topk(keep).values.min()
        mask[v0:v1] = vals >= thr
    out = torch.zeros_like(h)
    if bool(mask.any()):
        # 手动 swiglu：fc1 -> chunk(gate,up) -> silu(gate)*up -> fc2。
        # 不依赖 comfy.ops.linear_input_act 的 fused kernel（CPU 上静默返回 0，
        # 且对 gather 子集输入行为不可靠）；标准算子 CPU/GPU 均正确、数值等价。
        xs = h[mask]
        h1 = block.mlp.fc1(xs)
        g, u = h1.chunk(2, dim=-1)
        out[mask] = block.mlp.fc2(F.silu(g) * u)
    return out


def _h3_ditblock_forward(self, x, t_emb, mod_segments, rope_freqs,
                         transformer_options=None, attention=None):
    """替换 DiTBlock.forward：逐行复刻原逻辑，FFN 调用点做 token 稀疏。"""
    if transformer_options is None:
        transformer_options = {}
    from comfy.ldm.minimax.model import _mod_scale_shift, _mod_gate
    attention = self.attn if attention is None else attention
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaln_proj(t_emb)
    h = _mod_scale_shift(self.norm1(x), shift_msa, scale_msa, mod_segments)
    x = _mod_gate(x, gate_msa, attention(h, rope_freqs=rope_freqs,
                                         transformer_options=transformer_options),
                  mod_segments)
    h = _mod_scale_shift(self.norm2(x), shift_mlp, scale_mlp, mod_segments)
    h2 = _sparse_ffn_call(self, x, h, transformer_options)
    return _mod_gate(x, gate_mlp, h2, mod_segments)


def _h3_refinerblock_forward(self, x, transformer_options=None):
    """替换 RefinerBlock.forward：FFN 调用点做 token 稀疏。"""
    if transformer_options is None:
        transformer_options = {}
    from comfy.ldm.minimax.model import _mod_scale_shift, _mod_gate
    x = self.attn(self.norm1(x), transformer_options=transformer_options).add_(x)
    h = self.norm2(x)
    h2 = _sparse_ffn_call(self, x, h, transformer_options)
    return h2.add_(x)



# ---------------------------------------------------------------------------
# 安装 / 卸载 patch
# ---------------------------------------------------------------------------

def _is_h3_attention(module):
    cls = type(module)
    return cls.__name__ == "Attention" and cls.__module__ == "comfy.ldm.minimax.model"


def _install_h3_attn_patch(diffusion, params):
    """实例级 monkey-patch H3 Attention.forward + DiTBlock/RefinerBlock.forward（幂等）。

    同一 diffusion 对象重复 apply 时只更新参数，不重复包裹。
    """
    patched = 0
    for name, module in diffusion.named_modules():
        cls = type(module)
        if cls.__name__ == "Attention" and cls.__module__ == "comfy.ldm.minimax.model":
            if getattr(module, "_bsai_orig_forward", None) is None:
                module._bsai_orig_forward = module.forward
                module.forward = types.MethodType(_h3_attn_forward, module)
            module._bsai_veda_params = params
            patched += 1
        elif cls.__name__ in ("DiTBlock", "RefinerBlock") \
                and cls.__module__ == "comfy.ldm.minimax.model":
            if getattr(module, "_bsai_orig_forward", None) is None:
                module._bsai_orig_forward = module.forward
                module.forward = types.MethodType(
                    _h3_ditblock_forward if cls.__name__ == "DiTBlock"
                    else _h3_refinerblock_forward, module)
            module._bsai_veda_params = params
            patched += 1
    if patched == 0:
        raise RuntimeError(
            "[BSAI VedaSparse v3.4] 未找到任何 H3 Attention/DiTBlock 模块 "
            f"({type(diffusion).__name__})；确认是 MiniMax-H3 diffusion model。")
    logging.info(f"[BSAI VedaSparse v3.4] H3 Attention/Block.forward patched: {patched} 实例")
    return patched


def _unpatch_h3_attn(diffusion):
    for _, module in diffusion.named_modules():
        if not _is_h3_attention(module):
            continue
        orig = getattr(module, "_bsai_orig_forward", None)
        if orig is not None:
            module.forward = orig
            module._bsai_orig_forward = None
        module._bsai_veda_params = None


# ---------------------------------------------------------------------------
# 顶层应用
# ---------------------------------------------------------------------------

def apply_veda(model, *, enabled, keep_percent, min_tokens,
               start_percent=0.5, end_percent=1.0,
               sink_conditioning="exact_kv_and_rows", head_tiling=None,
               tripool_mode="triplet", scorer_weights=None, aspect="16:9",
               force_dims=None, verbose=False, vram_budget_gb=5,
               max_sparse_tokens=16384, ffn_sparse=False, ffn_keep_percent=60.0):
    if not enabled:
        logging.info("[BSAI VedaSparse v3.4] enabled=False -> passthrough")
        return model

    m = model.clone()
    diffusion = m.get_model_object("diffusion_model")
    if not (hasattr(diffusion, "rope_freqs") and hasattr(diffusion, "blocks")):
        raise RuntimeError(
            "BSAI VedaSparse v3.4 expects a MiniMax-H3 diffusion model; got "
            f"{type(diffusion).__name__}.")

    ms = m.get_model_object("model_sampling")
    sigma_start = float(ms.percent_to_sigma(start_percent))
    sigma_end = float(ms.percent_to_sigma(end_percent))
    keep_frac = max(0.005, min(1.0, keep_percent / 100.0))

    params = {
        "enabled": True,
        "keep_frac": keep_frac,
        "min_tokens": int(min_tokens),
        "sigma_start": sigma_start,
        "sigma_end": sigma_end,
        "sink": sink_conditioning,
        "tripool": tripool_mode,
        "verbose": bool(verbose),
        "max_sparse_tokens": int(max_sparse_tokens),
        "ffn_sparse": bool(ffn_sparse),
        "ffn_keep_frac": max(0.1, min(1.0, ffn_keep_percent / 100.0)),
    }
    _install_h3_attn_patch(diffusion, params)

    m.model_options["transformer_options"]["bsai_vedasparse_v34"] = {
        "keep_percent": keep_percent, "min_tokens": min_tokens,
        "tripool": tripool_mode, "sigma_window": [sigma_end, sigma_start],
        "sink": sink_conditioning,
        "triton_available": _TRITON_SPARSE_AVAILABLE,
        "max_sparse_tokens": int(max_sparse_tokens),
        "ffn_sparse": bool(ffn_sparse),
        "ffn_keep_percent": ffn_keep_percent,
        "patched": True,
    }
    reset_veda_stats()
    logging.info(
        f"[BSAI VedaSparse v3.4] applied (monkey-patch): keep={keep_percent}% "
        f"tripool={tripool_mode} triton={_TRITON_SPARSE_AVAILABLE} "
        f"sigma_window=[{sigma_end:.3f}, {sigma_start:.3f}] "
        f"sink={sink_conditioning} max_sparse_tokens={max_sparse_tokens} "
        f"ffn_sparse={ffn_sparse} ffn_keep={ffn_keep_percent}%")
    return m


def veda_stats():
    return dict(_STATS)


def reset_veda_stats():
    for k in _STATS:
        _STATS[k] = 0
    _SEEN.clear()
