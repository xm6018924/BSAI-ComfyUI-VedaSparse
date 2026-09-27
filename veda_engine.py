"""BSAI VedaSparse — Veda 蒸馏稀疏注意力引擎 for MiniMax-H3.

What it is
----------
VedaSparse 是 Veda（ByteDance + HKU，ICML 2026，arXiv:2605.30325）蒸馏稀疏注意力
在 MiniMax-H3 上的落地实现。相比 FastVideo VSA（固定 64-token 立方 tile + 平均池化
评分），Veda 有两个关键升级（论文实测 tile 召回率 66.4% vs 34.2%）：

  1. Statistics-Aware Tile Scoring（统计感知 tile 评分）
     VSA 用平均池化压缩 tile，会稀释峰值信号；Veda 用 TripPool 描述符
     （Avg ⊕ Max ⊕ Min 三统计量拼接）保留 tile 内峰值，评分更贴近全注意力。
     论文 Table 2：Triplet 0.912 < Avg 0.965 < MaxMin 0.982（训练损失越低越好）。

  2. Head-Aware Tiling（按头时空分块）
     不同注意力头关注不同的时空结构（局部空间 vs 长程时间），Veda 为每个头
     分配 (pt, ph, pw) 时空分块配置（pt·ph·pw = 硬件 tile 大小 64），减少
     结构与全注意力失配（论文 Fig.6：不同头不同 tiling 召回率差异巨大）。

  3. Tile-Skipping Sparse Attention（tile 跳过稀疏注意力）
     每个 query tile 只对 top-k 个 key tile 精确计算（默认 10%），其余跳过；
     文本/音频/参考 conditioning 行始终保持 dense（精确），避免提示词和
     音轨退化。这是 ComfyUI 里把理论稀疏变成实际墙钟加速的执行层。

Scorer modes（评分器两种模式）
------------------------------
  * heuristic（默认，无需任何权重）：φ = identity，用 TripPool 描述符直接点积
    评分，公式即论文 eq.6 在单位投影下的形式。开箱即用。
  * distilled（预留）：加载 VedaSparse 蒸馏预测器权重（263MB FP8，每头独立
    MLP 投影 φ_q/φ_k），key 规范见 ``VEDA_SCORER_KEY_SPEC``。权重文件放入
    ComfyUI/models/veda_scorers/。当官方权重公开后，无需改代码即可启用蒸馏评分。

Usage（与 FastH3 同 seam）
--------------------------
通过 transformer_options["optimized_attention_override"] 注入，ComfyUI 的
H3 注意力会回调本引擎。只打 ModelPatcher.clone() + model_options，不改
ComfyUI 内部源码，升级无碍。

Reference
---------
Veda: Scalable Video Diffusion via Distilled Sparse Attention
Shihao Han, Hao Yang, Xiaofeng Mei, Xinting Hu, Yi Jiang, Xiaojuan Qi
ICML 2026 · arXiv:2605.30325
"""

import logging
import math
import sys
from functools import partial

import torch
import torch.nn.functional as F

BLOCK = 64                       # H3 tile size (与 FastVideo VSA / ComfyUI 一致)
_HEAD_TILING_FACTORIZATIONS = (  # (pt, ph, pw) 且 pt*ph*pw = 64 的候选集合
    (4, 4, 4),   # 均匀立方（VSA 默认，时空均衡）
    (8, 4, 2),   # 时间主导
    (2, 4, 8),   # 空间主导
    (4, 8, 2),   # 时间-高度主导
    (2, 8, 4),   # 高度主导
    (8, 2, 4),   # 时间-宽度主导
    (4, 2, 8),   # 宽度主导
    (1, 8, 8),   # 纯空间
    (8, 8, 1),   # 纯时间
    (1, 4, 16),  # 极细空间
    (16, 4, 1),  # 极细时间
    (1, 2, 32),  # 超宽空间
    (32, 2, 1),  # 超长时间
)
# 默认按头循环分配的分组（论文 Fig.6 显示不同头需要不同 tiling；组内头共享
# tile 结构以控制 gather 开销，评分仍逐头计算）。组数=4 是加速/质量的平衡点。
_DEFAULT_HEAD_TILING_GROUPS = ((4, 4, 4), (8, 4, 2), (2, 4, 8), (4, 8, 2))

_STATS = {"sparse": 0, "dense": 0, "heuristic": 0, "distilled": 0,
          "errors": 0, "fallback_1d": 0}
_SEEN = set()
_SPAN_INSTALLED = set()
_PATCHED_LAYOUTS = set()
_SPANS = {}                    # id(position_ids) -> (layout, video_span, audio_span, latent_dims)

# VedaSparse 蒸馏预测器权重 key 规范（预留；官方权重发布后按此加载）
#   vedascorer.layer_{l}.head_{h}.q_proj   [in=3*d_head, out=d_latent]
#   vedascorer.layer_{l}.head_{h}.k_proj   [in=3*d_head, out=d_latent]
VEDA_SCORER_KEY_SPEC = "vedascorer.layer_{l}.head_{h}.{qk}_proj"


# ---------------------------------------------------------------------------
# H3 packed-layout span 发布（独立实现，与 FastH3 思路一致但不互相依赖）
# ---------------------------------------------------------------------------

def _video_span(layout):
    segments = getattr(layout, "segments", None)
    if not segments:
        return None
    return next(((a, b) for a, b, kind in segments if kind == "video"), None)


def _audio_span(layout):
    segments = getattr(layout, "segments", None)
    if not segments:
        return None
    return next(((a, b) for a, b, kind in segments if kind == "audio"), None)


def _patch_packed_layout(module):
    """记录每个 PackedLayout 的 video/audio span 与 latent 三维尺寸，不改动布局对象。"""
    layout_cls = getattr(module, "PackedLayout", None)
    if layout_cls is None:
        raise RuntimeError(f"{module.__name__} has no PackedLayout")
    if id(layout_cls) in _PATCHED_LAYOUTS:
        return
    original_init = layout_cls.__init__

    def __init__(self, text_len, latent_t, latent_h, latent_w, audio_t, *args, **kwargs):
        original_init(self, text_len, latent_t, latent_h, latent_w, audio_t,
                      *args, **kwargs)
        try:
            span = _video_span(self)
            latent_dims = (int(latent_t), int(latent_h), int(latent_w))
        except Exception:                                   # never break construction
            span, latent_dims = None, None
        if torch.is_tensor(getattr(self, "position_ids", None)) and span is not None:
            _SPANS[id(self.position_ids)] = (self, span, _audio_span(self), latent_dims)

    layout_cls.__init__ = __init__
    _PATCHED_LAYOUTS.add(id(layout_cls))


def install_h3_span(model):
    """幂等地把 H3 的 video/audio span 与 latent 尺寸发布进 transformer_options。

    非 H3 扩散模型（缺 .blocks/.rope_freqs/._forward）直接抛错，由节点捕获。
    可对同一对象重复调用。
    """
    if id(model) in _SPAN_INSTALLED:
        return
    for attr in ("rope_freqs", "_forward", "blocks"):
        if not hasattr(model, attr):
            raise RuntimeError(
                "BSAI VedaSparse expects a MiniMax-H3 diffusion model "
                f"(.{attr} missing on {type(model).__name__}).")

    _patch_packed_layout(sys.modules[type(model).__module__])
    original_forward = model._forward
    original_rope = model.rope_freqs

    def _forward(x, timestep, context, transformer_options={}, **kwargs):
        model._veda_options = transformer_options
        try:
            return original_forward(x, timestep, context,
                                    transformer_options=transformer_options, **kwargs)
        finally:
            model._veda_options = None
            transformer_options.pop("h3_video_span", None)
            transformer_options.pop("h3_audio_span", None)
            transformer_options.pop("h3_latent_dims", None)

    def rope_freqs(position_ids, device):
        entry = _SPANS.get(id(position_ids))
        if entry is not None:
            options = getattr(model, "_veda_options", None)
            if options is not None:
                options["h3_video_span"] = entry[1]
                options["h3_audio_span"] = entry[2]
                options["h3_latent_dims"] = entry[3]
        return original_rope(position_ids, device)

    model._forward = _forward
    model.rope_freqs = rope_freqs
    _SPAN_INSTALLED.add(id(model))


def veda_stats():
    """进程内分发计数（只读诊断）。"""
    return dict(_STATS)


def reset_veda_stats():
    for key in _STATS:
        _STATS[key] = 0
    _SEEN.clear()


def _log_once(key, message):
    if key not in _SEEN:
        _SEEN.add(key)
        logging.info(f"[BSAI VedaSparse] {message}")


# ---------------------------------------------------------------------------
# TripPool 描述符与评分
# ---------------------------------------------------------------------------

def _tripool(tiles, mode="triplet"):
    """tiles: [B, n_tiles, tile_tokens, H, D] -> [B, n_tiles, H, D_emb].

    TripPool = Avg ⊕ Max ⊕ Min（论文 eq.5）。mode 控制统计量组合：
      triplet: avg+max+min（论文最优，Table 2 最低损失）
      maxmin:  max+min
      avg:     仅平均（即 VSA 的评分信号，保留作对比）
    """
    avg = tiles.mean(dim=2)                                  # [B, n, H, D]
    if mode == "avg":
        return avg
    mx = tiles.amax(dim=2)
    if mode == "maxmin":
        return torch.cat([mx, tiles.amin(dim=2)], dim=-1)
    return torch.cat([avg, mx, tiles.amin(dim=2)], dim=-1)   # [B, n, H, 3D]


def _heuristic_scores(qpool, kpool, scale):
    """φ=identity 的 TripPool 评分（论文 eq.6 单位投影形式）。

    qpool/kpool: [B, n_tiles, H, D_emb] -> scores [B, H, n_tiles, n_tiles]。
    """
    B, nq, H, Demb = qpool.shape
    q = qpool.permute(0, 2, 1, 3).reshape(B * H, nq, Demb)
    k = kpool.permute(0, 2, 1, 3).reshape(B * H, nq, Demb)
    d_emb = float(Demb) ** -0.5
    s = torch.bmm(q, k.transpose(-2, -1)) * d_emb           # [BH, nq, nq]
    return s.view(B, H, nq, nq)                             # [B, H, nq, nq]


class _DistilledScorer(torch.nn.Module):
    """VedaSparse 蒸馏预测器（预留）：每头独立 MLP 投影 φ_q / φ_k。

    加载 safetensors 后启用；未提供权重时节点走 heuristic 路径，本类不实例化。
    Key 规范见 VEDA_SCORER_KEY_SPEC：
      vedascorer.layer_{l}.head_{h}.q_proj / k_proj   [in=3*d_head, out=d_latent]
    """

    def __init__(self, state_dict, num_heads, d_head):
        super().__init__()
        self.proj = torch.nn.ModuleList()
        layers = set()
        heads = set()
        for key in state_dict:
            parts = key.split(".")
            if len(parts) == 5 and parts[0] == "vedascorer":
                layers.add(int(parts[1]))
                heads.add(int(parts[2]))
        for l in sorted(layers):
            head_projs = torch.nn.ModuleList()
            for h in sorted(heads):
                qp = torch.nn.Linear(state_dict[f"vedascorer.layer_{l}.head_{h}.q_proj"]
                                     .shape[-1],
                                     state_dict[f"vedascorer.layer_{l}.head_{h}.q_proj"]
                                     .shape[0], bias=False)
                kp = torch.nn.Linear(state_dict[f"vedascorer.layer_{l}.head_{h}.k_proj"]
                                     .shape[-1],
                                     state_dict[f"vedascorer.layer_{l}.head_{h}.k_proj"]
                                     .shape[0], bias=False)
                with torch.no_grad():
                    qp.weight.copy_(state_dict[f"vedascorer.layer_{l}.head_{h}.q_proj"])
                    kp.weight.copy_(state_dict[f"vedascorer.layer_{l}.head_{h}.k_proj"])
                head_projs.append(torch.nn.ModuleDict({"q": qp, "k": kp}))
            head_projs.to(next(iter(state_dict.values())).dtype)
            self.proj.append(head_projs)
        self.layers = sorted(layers)
        self.heads = sorted(heads)

    def score(self, qpool, kpool, layer, head, d_latent_scale=True):
        """对指定层/头计算投影评分。qpool/kpool: [B, n, D_emb]。"""
        hp = self.proj[self.layers.index(layer)][self.heads.index(head)]
        q = hp["q"](qpool)
        k = hp["k"](kpool)
        return torch.bmm(q, k.transpose(-2, -1)) * (q.shape[-1] ** -0.5)


# ---------------------------------------------------------------------------
# Head-Aware Tiling
# ---------------------------------------------------------------------------

def resolve_geometry(video_span_len, latent_dims, aspect, force_dims=None):
    """把 video span 解析为 (T, H, W) 三维 token 布局。

    优先级：
      1. force_dims（节点手动指定 (T,H,W)）
      2. PackedLayout 记录的 latent_dims（含 patchify 前后两种校验）
      3. duration + aspect 反推（latent 短边 24 = 768px/32，patchify 1×2×2 后）
      4. 失败返回 None -> 引擎退避一维立方 tile（仅评分升级，保持兼容）
    """
    if force_dims is not None:
        t, h, w = force_dims
        if t * h * w == video_span_len:
            return (t, h, w)
    if latent_dims is not None:
        t, h, w = latent_dims
        if t * h * w == video_span_len:
            return (t, h, w)
        t2, h2, w2 = t, h // 2, w // 2                     # patchify 1×2×2 变体
        if h % 2 == 0 and w % 2 == 0 and t2 * h2 * w2 == video_span_len:
            return (t2, h2, w2)
    # 按画幅比例反推：短边 latent 24（768px/32），宽边按 aspect
    ratio = {"16:9": 16 / 9, "9:16": 9 / 16, "4:3": 4 / 3,
             "3:4": 3 / 4, "1:1": 1.0}.get(aspect, 16 / 9)
    if video_span_len > 0:
        h = 24
        w = max(1, int(round(h * ratio)))
        # 尝试几个合理 (h, w) 组合
        for hh in (24, 32, 16, 12, 8):
            for ww in (int(round(hh * ratio)),):
                if ww <= 0:
                    continue
                if video_span_len % (hh * ww) == 0:
                    t = video_span_len // (hh * ww)
                    if t > 0:
                        return (t, hh, ww)
        # 最后尝试任意 (h,w) 分解（质因数近似）
        s = int(round(math.sqrt(video_span_len)))
        for hh in range(max(1, s // 8), s + 1):
            if hh * hh > video_span_len:
                break
            if video_span_len % hh == 0:
                ww = video_span_len // hh
                if ww <= 8 * hh:
                    return (1, hh, ww)
    return None


def _tile_3d(xv, T, H, W, pt, ph, pw, heads, d):
    """把 video span [B, vn, Hd, D] 按 (pt, ph, pw) 分块。

    返回 (tiles, n_tiles, (Tp, Hp, Wp), key_mask)：
      tiles: [B, n_tiles, pt*ph*pw, Hd, D]
      key_mask: [n_tiles, pt*ph*pw]（False=padding token，不应参与 softmax）
    token 排列假设为 (t, h, w) 展平（t 最慢、w 最快，MM-RoPE 三维）。
    """
    B, vn, Hd, D = xv.shape
    x3 = xv.view(B, T, H, W, Hd, D)
    Tp, Hp, Wp = -(-T // pt), -(-H // ph), -(-W // pw)
    if Tp * pt != T or Hp * ph != H or Wp * pw != W:
        pad_t = Tp * pt - T
        pad_h = Hp * ph - H
        pad_w = Wp * pw - W
        x3 = F.pad(x3, (0, 0, 0, 0, 0, pad_w, 0, pad_h, 0, pad_t))
    # [B, Tp, pt, Hp, ph, Wp, pw, Hd, D] -> tiles [B, Tp*Hp*Wp, pt*ph*pw, Hd, D]
    tiles = x3.view(B, Tp, pt, Hp, ph, Wp, pw, Hd, D)
    tiles = tiles.permute(0, 1, 3, 5, 2, 4, 6, 7, 8)
    tiles = tiles.reshape(B, Tp * Hp * Wp, pt * ph * pw, Hd, D)
    # 有效 key mask：真实 (T,H,W) 网格内的 token 为 True
    valid = torch.zeros(Tp * pt, Hp * ph, Wp * pw, dtype=torch.bool,
                        device=xv.device)
    valid[:T, :H, :W] = True
    key_mask = valid.view(Tp, pt, Hp, ph, Wp, pw).permute(0, 2, 4, 1, 3, 5)
    key_mask = key_mask.reshape(Tp * Hp * Wp, pt * ph * pw)
    return tiles, Tp * Hp * Wp, (Tp, Hp, Wp), key_mask


# ---------------------------------------------------------------------------
# tile-skipping 稀疏注意力（每 tiling 组一个结构，组内头共享；评分逐头）
# ---------------------------------------------------------------------------

def _veda_sparse(qs, ks, vs, scale, keep_frac, video_start, video_end,
                 sink_conditioning, verbose, head_tilings, tripool_mode,
                 scorer, latent_dims, aspect, force_dims):
    """Block-sparse attention over the video span with head-aware tiling.

    qs/ks/vs: [B, N, H, D]（N=packed 序列，H=注意力头，D=head_dim）。
    head_tilings: per-head 的 (pt, ph, pw) 列表（长度=H，或长度<H 时循环）。
    返回 [B, N, H, D]。
    """
    B, N, Hd, D = qs.shape
    cond_end = int(video_start)
    vn = int(video_end) - cond_end
    out = torch.empty_like(qs)

    def _attn(qr, kr, vr):
        b, lq, h, d = qr.shape
        lk = kr.shape[1]
        q2 = qr.transpose(1, 2).reshape(b * h, lq, d)
        k2 = kr.transpose(1, 2).reshape(b * h, lk, d).transpose(-2, -1)
        att = torch.bmm(q2, k2) * scale
        att = att.softmax(dim=-1)
        o = torch.bmm(att, vr.transpose(1, 2).reshape(b * h, lk, d))
        return o.reshape(b, h, lq, d).transpose(1, 2)

    # --- conditioning（text/audio/ref）行：始终 dense，逐块限内存 -------------
    if cond_end > 0:
        for i in range(0, cond_end, 256):
            j = min(i + 256, cond_end)
            out[:, i:j] = _attn(qs[:, i:j], ks, vs)
    if vn <= 0:
        out[:, cond_end:] = _attn(qs[:, cond_end:], ks, vs)
        return out

    # --- 几何解析 ----------------------------------------------------------
    geom = resolve_geometry(vn, latent_dims, aspect, force_dims)
    qv_raw = qs[:, cond_end:video_end]
    kv_raw = ks[:, cond_end:video_end]
    vv_raw = vs[:, cond_end:video_end]

    if geom is None:
        # 一维退避：固定 64-token 立方 tile（FastVideo VSA 几何），
        # 评分仍升级为 TripPool。保证任何分辨率都能跑。
        _STATS["fallback_1d"] += 1
        if verbose:
            _log_once("fallback1d", "geometry unresolved -> 64-token cubic tiling")
        return _veda_1d_fallback(qv_raw, kv_raw, vv_raw, qs, ks, vs, scale,
                                 keep_frac, cond_end, video_end,
                                 sink_conditioning, tripool_mode, scorer, out)

    T, Hlat, Wlat = geom
    # 归一化 head 分块表：长度=Hd，不足循环
    tilings = []
    for h in range(Hd):
        tilings.append(head_tilings[h % len(head_tilings)])

    # 按 tiling 分组（相同 (pt,ph,pw) 的头归一组，共享 tile 结构）
    groups = {}
    for h, cfg in enumerate(tilings):
        groups.setdefault(cfg, []).append(h)
    group_cfgs = list(groups.keys())

    # 为每组的 tile 结构做准备（q/k/v 都按组切）
    per_group = {}
    for cfg in group_cfgs:
        pt, ph, pw = cfg
        qt, nt, (Tp, Hp, Wp), _ = _tile_3d(qv_raw, T, Hlat, Wlat, pt, ph, pw, Hd, D)
        kt, _, _, km = _tile_3d(kv_raw, T, Hlat, Wlat, pt, ph, pw, Hd, D)
        vt, _, _, _ = _tile_3d(vv_raw, T, Hlat, Wlat, pt, ph, pw, Hd, D)
        per_group[cfg] = (qt, kt, vt, nt, km)

    # --- 逐组：TripPool 评分（逐头） + top-k + gather 稀疏注意力 --------------
    use_cond = sink_conditioning != "off" and cond_end > 0
    if use_cond:
        ck = ks[:, :cond_end]
        cv = vs[:, :cond_end]

    ov_parts = {}                                           # head -> out tensor
    for cfg in group_cfgs:
        heads = groups[cfg]
        qt, kt, vt, nt, km = per_group[cfg]
        topk = max(1, round(nt * keep_frac))
        # TripPool 描述符：tiles [B, nt, ts, H, D] -> pool [B, nt, H, Demb]
        qp = _tripool(qt, tripool_mode)
        kp = _tripool(kt, tripool_mode)
        hs = heads

        if scorer is not None and scorer.layers:
            # 蒸馏模式：每头独立 MLP 投影（仅对存在的层/头；缺失头回退启发）
            qp_p = qp.permute(0, 2, 1, 3).reshape(B * Hd, nt, qp.shape[-1])
            kp_p = kp.permute(0, 2, 1, 3).reshape(B * Hd, nt, kp.shape[-1])
            scores_h = torch.empty(B * Hd, nt, nt, device=qs.device, dtype=qp.dtype)
            for h in range(Hd):
                try:
                    s = scorer.score(qp_p[h * B:(h + 1) * B] if B > 1 else qp_p,
                                     kp_p[h * B:(h + 1) * B] if B > 1 else kp_p,
                                     layer=0, head=h)
                except (IndexError, KeyError):
                    q2 = qp_p[h * B:(h + 1) * B] if B > 1 else qp_p
                    k2 = kp_p[h * B:(h + 1) * B] if B > 1 else kp_p
                    s = torch.bmm(q2, k2.transpose(-2, -1)) * (q2.shape[-1] ** -0.5)
                scores_h[h * B:(h + 1) * B] = s if B > 1 else s
            scores = scores_h.view(B, Hd, nt, nt)
            _STATS["distilled"] += 1
        else:
            scores = _heuristic_scores(qp, kp, scale)       # [B, H, nt, nt]
            _STATS["heuristic"] += 1

        # top-k 逐头
        sc = scores.reshape(B * Hd, nt, nt)
        topk_idx = sc.topk(topk, dim=-1).indices            # [BH, nt, topk]

        # gather：每组头共享 tile 索引结构 -> 直接按整组 gather
        hs_t = torch.tensor(heads, device=qs.device)
        # 显存优化：只 gather 当前头 h 的 k/v 列（[B,nt,ts,D]），
        # 避免原实现把全部 H 头复制（[B,nt,ts,H*D]）再取一列的 H× 浪费。
        for h in heads:
            # 当前头的索引（在 BH 大索引中的位置）
            row = h * B
            idx = topk_idx[row:row + B]                     # [B, nt, topk]
            ts_ = qt.shape[2]
            kth = kt[:, :, :, h, :]                         # [B, nt, ts, D]
            vth = vt[:, :, :, h, :]
            kthf = kth.reshape(B, nt, ts_ * D)
            vthf = vth.reshape(B, nt, ts_ * D)
            idx_flat = idx.reshape(B, nt * topk)
            gkf = torch.gather(kthf, 1, idx_flat.unsqueeze(-1).expand(
                B, nt * topk, ts_ * D))
            gvf = torch.gather(vthf, 1, idx_flat.unsqueeze(-1).expand(
                B, nt * topk, ts_ * D))
            gk = gkf.reshape(B, nt, topk * ts_, D)
            gv = gvf.reshape(B, nt, topk * ts_, D)
            # key mask（tile 内 padding token 不参与 softmax；按 batch 高级索引收集）
            kmf = km.reshape(nt, -1)                         # [nt, ts_]
            gkm = kmf[idx_flat].reshape(B, nt, topk * ts_)   # [B, nt, topk*ts_]
            if use_cond:
                k2 = torch.cat([ck[:, :, h, :].unsqueeze(1).expand(B, nt, cond_end, D),
                                gk], dim=2)
                v2 = torch.cat([cv[:, :, h, :].unsqueeze(1).expand(B, nt, cond_end, D),
                                gv], dim=2)
                key_mask = torch.cat(
                    [torch.ones(B, nt, cond_end, dtype=torch.bool,
                                device=qs.device), gkm], dim=2)
            else:
                k2, v2 = gk, gv
                key_mask = gkm
            # 稀疏注意力：query tile 内部 token 与选中 key 精确计算
            qhh = qt[:, :, :, h, :]                          # [B, nt, ts, D]
            khh = k2                                          # [B, nt, klen, D]（已只含 h 头）
            vhh = v2
            att = torch.matmul(qhh, khh.transpose(-2, -1)) * scale
            att = att.masked_fill(
                ~key_mask[:, :, None, :].expand(B, nt, att.shape[2], -1),
                float("-inf"))
            att = att.softmax(dim=-1)
            att = att.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)  # 全掩码行防护
            oh = torch.matmul(att, vhh)                      # [B, nt, ts, D]
            ov_parts[h] = oh

    # --- 写回 video span（按 head 逐列） --------------------------------------
    # ov 形状 [B, vn, H, D]：需要把每头结果 un-tile 回原始 (t,h,w) 顺序
    # 简化：由于不同头 tile 结构不同，这里按组分别 un-tile 再写回。
    for cfg in group_cfgs:
        heads = groups[cfg]
        qt, kt, vt, nt, km = per_group[cfg]
        pt, ph, pw = cfg
        for h in heads:
            oh = ov_parts[h]                                 # [B, nt, ts, D]
            # un-tile：nt tiles x (pt*ph*pw) -> 原始 (T, H, W)
            Tp, Hp, Wp = -(-T // pt), -(-Hlat // ph), -(-Wlat // pw)
            oh3 = oh.view(B, Tp, Hp, Wp, pt, ph, pw, D)
            oh3 = oh3.permute(0, 1, 4, 2, 5, 3, 6, 7).reshape(B, Tp * pt, Hp * ph,
                                                              Wp * pw, D)
            ohv = oh3[:, :T, :Hlat, :Wlat].reshape(B, vn, D)
            out[:, cond_end:video_end, h, :] = ohv
    return out


def _veda_1d_fallback(qv, kv, vv, qs, ks, vs, scale, keep_frac, cond_end,
                      video_end, sink_conditioning, tripool_mode, scorer, out):
    """一维 64-token 立方 tile 退避（FastVideo VSA 几何 + TripPool 评分）。

    用于几何无法解析（任意分辨率/帧数）时保证可用性。
    """
    B, vn, Hd, D = qv.shape
    nvb = (vn + BLOCK - 1) // BLOCK
    pad = nvb * BLOCK - vn

    # conditioning（text/audio/ref）行始终 dense（与主路径一致）
    if cond_end > 0:
        for i in range(0, cond_end, 256):
            j = min(i + 256, cond_end)
            b2, nq, h2, d2 = qs[:, i:j].shape
            q2c = qs[:, i:j].transpose(1, 2).reshape(b2 * h2, nq, d2)
            k2c = ks.transpose(1, 2).reshape(b2 * h2, ks.shape[1], d2)
            attc = torch.bmm(q2c, k2c.transpose(-2, -1)) * scale
            attc = attc.softmax(dim=-1)
            oc = torch.bmm(attc, vs.transpose(1, 2).reshape(b2 * h2, vs.shape[1], d2))
            out[:, i:j] = oc.reshape(b2, h2, nq, d2).transpose(1, 2)

    def _slice(t):
        if not pad:
            return t
        return F.pad(t, (0, 0, 0, 0, 0, pad, 0, 0))

    qvv = _slice(qv).view(B, nvb, BLOCK, Hd, D)
    kvv = _slice(kv).view(B, nvb, BLOCK, Hd, D)
    vvv = _slice(vv).view(B, nvb, BLOCK, Hd, D)

    qp = _tripool(qvv, tripool_mode)
    kp = _tripool(kvv, tripool_mode)
    scores = _heuristic_scores(qp, kp, scale)               # [B, H, nvb, nvb]
    topk = max(1, round(nvb * keep_frac))
    sc = scores.reshape(B * Hd, nvb, nvb)
    topk_idx = sc.topk(topk, dim=-1).indices

    use_cond = sink_conditioning != "off" and cond_end > 0
    if use_cond:
        ck = ks[:, :cond_end].permute(0, 2, 1, 3).reshape(B * Hd, cond_end, D)
        cv = vs[:, :cond_end].permute(0, 2, 1, 3).reshape(B * Hd, cond_end, D)

    qv2 = qvv.permute(0, 3, 1, 2, 4).reshape(B * Hd, nvb, BLOCK, D)
    kvf = kvv.permute(0, 3, 1, 2, 4).reshape(B * Hd, nvb * BLOCK, D)
    vvf = vvv.permute(0, 3, 1, 2, 4).reshape(B * Hd, nvb * BLOCK, D)
    tok_idx = (topk_idx * BLOCK).unsqueeze(-1) + torch.arange(BLOCK, device=qs.device)
    tok_flat = tok_idx.reshape(B * Hd, nvb, topk * BLOCK)
    # 有效 video token mask（padding 不参与 softmax）
    valid_video = torch.zeros(nvb * BLOCK, dtype=torch.bool, device=qs.device)
    valid_video[:vn] = True

    CHUNK = 4
    pieces = []
    for i in range(0, nvb, CHUNK):
        j = min(i + CHUNK, nvb)
        nb = j - i
        q2 = qv2[:, i:j].reshape(B * Hd * nb, BLOCK, D)
        idx = tok_flat[:, i:j]
        gk = torch.gather(kvf.unsqueeze(1).expand(B * Hd, nb, nvb * BLOCK, D),
                          2, idx.unsqueeze(-1).expand(B * Hd, nb, topk * BLOCK, D)
                          ).reshape(B * Hd * nb, topk * BLOCK, D)
        gv = torch.gather(vvf.unsqueeze(1).expand(B * Hd, nb, nvb * BLOCK, D),
                          2, idx.unsqueeze(-1).expand(B * Hd, nb, topk * BLOCK, D)
                          ).reshape(B * Hd * nb, topk * BLOCK, D)
        gkm = valid_video[idx].reshape(B * Hd * nb, topk * BLOCK)
        if use_cond:
            k2 = torch.cat([ck.unsqueeze(1).expand(B * Hd, nb, cond_end, D)
                            .reshape(B * Hd * nb, cond_end, D), gk], dim=1)
            v2 = torch.cat([cv.unsqueeze(1).expand(B * Hd, nb, cond_end, D)
                            .reshape(B * Hd * nb, cond_end, D), gv], dim=1)
            key_mask = torch.cat(
                [torch.ones(B * Hd * nb, cond_end, dtype=torch.bool,
                            device=qs.device), gkm], dim=1)
        else:
            k2, v2 = gk, gv
            key_mask = gkm
        att = torch.bmm(q2, k2.transpose(-2, -1)) * scale
        att = att.masked_fill(~key_mask[:, None, :].expand(-1, BLOCK, -1),
                              float("-inf"))
        att = att.softmax(dim=-1)
        att = att.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
        pieces.append(torch.bmm(att, v2).reshape(B, Hd, nb, BLOCK, D))
    ov = torch.cat(pieces, dim=2).permute(0, 2, 3, 1, 4).reshape(B, nvb * BLOCK, Hd, D)
    out[:, cond_end:video_end] = ov[:, :vn]
    return out


# ---------------------------------------------------------------------------
# override builder（注入 transformer_options["optimized_attention_override"]）
# ---------------------------------------------------------------------------

def make_veda_override(*, keep_frac, min_tokens, sigma_start, sigma_end,
                       sink_conditioning, head_tilings, tripool_mode,
                       scorer, latent_dims, aspect, force_dims, verbose,
                       previous=None):
    def override(func, q, k, v, heads, mask=None, attn_precision=None,
                 skip_reshape=False, skip_output_reshape=False, **kwargs):
        def dense():
            target = func if previous is None else partial(previous, func)
            return target(q, k, v, heads, mask=mask, attn_precision=attn_precision,
                          skip_reshape=skip_reshape,
                          skip_output_reshape=skip_output_reshape, **kwargs)

        if mask is not None:
            _STATS["dense"] += 1
            return dense()

        if skip_reshape:
            b, _, _, dim_head = q.shape                        # BHND
            qs, ks, vs = (t.transpose(1, 2) for t in (q, k, v))
        else:
            b, _, dim_head = q.shape                           # B, N, heads*dim_head
            dim_head //= heads
            qs, ks, vs = (t.view(b, -1, heads, dim_head) for t in (q, k, v))

        # sigma 窗口（前 20% 步 dense 预热，与 FastH3/ComfyUI 一致）
        if sigma_start is not None or sigma_end is not None:
            sigmas = kwargs.get("transformer_options", {}).get("sigmas")
            if sigmas is not None:
                sigma = float(sigmas[0])
                if (sigma_start is not None and sigma > sigma_start) or \
                   (sigma_end is not None and sigma < sigma_end):
                    _STATS["dense"] += 1
                    return dense()

        options = kwargs.get("transformer_options") or {}
        span = options.get("h3_video_span")
        if span is None:
            _STATS["dense"] += 1
            if verbose:
                _log_once("nospan", "no H3 video span published; dense")
            return dense()
        video_start, video_end = span
        tokens = qs.shape[1]
        if tokens < min_tokens or tokens < int(video_end):
            _STATS["dense"] += 1
            return dense()
        if qs.shape[1] != ks.shape[1]:                         # cross-attention
            _STATS["dense"] += 1
            return dense()

        scale = kwargs.get("scale", dim_head ** -0.5)
        lat_dims = options.get("h3_latent_dims")
        try:
            out = _veda_sparse(qs, ks, vs, scale, keep_frac, video_start, video_end,
                               sink_conditioning, verbose, head_tilings,
                               tripool_mode, scorer, lat_dims, aspect, force_dims)
        except Exception as exc:
            _STATS["errors"] += 1
            if "out of memory" in str(exc).lower():
                _log_once("oom", "CUDA OOM in sparse attention; falling back to "
                                 "dense. If it recurs, reduce resolution/frames or "
                                 "use the fp8 model, and close other VRAM-heavy apps.")
            else:
                logging.error(f"[BSAI VedaSparse] engine failed ({exc}); dense fallback",
                              exc_info=verbose)
            return dense()
        _STATS["sparse"] += 1
        if skip_output_reshape:
            return out.transpose(1, 2)
        return out.reshape(b, -1, heads * dim_head)

    return override


# ---------------------------------------------------------------------------
# 顶层应用（由节点调用）
# ---------------------------------------------------------------------------

def apply_veda(model, *, enabled, keep_percent, min_tokens, start_percent,
               end_percent, sink_conditioning, head_tiling, tripool_mode,
               scorer_weights, aspect, force_dims, verbose,
               compose_with_foreign_patches=True):
    if not enabled:
        logging.info("[BSAI VedaSparse] disabled -> passthrough")
        return model

    m = model.clone()
    diffusion_model = m.get_model_object("diffusion_model")
    if not (hasattr(diffusion_model, "rope_freqs") and hasattr(diffusion_model, "_forward")):
        raise RuntimeError(
            "BSAI VedaSparse expects a MiniMax-H3 diffusion model; got "
            f"{type(diffusion_model).__name__}.")

    install_h3_span(diffusion_model)
    ms = m.get_model_object("model_sampling")
    sigma_start = float(ms.percent_to_sigma(start_percent))
    sigma_end = float(ms.percent_to_sigma(end_percent))
    keep_frac = max(0.005, min(1.0, keep_percent / 100.0))

    # 蒸馏评分器（可选）
    scorer = None
    if scorer_weights is not None and isinstance(scorer_weights, dict) and scorer_weights:
        try:
            scorer = _DistilledScorer(scorer_weights, heads=64, d_head=128)
            logging.info("[BSAI VedaSparse] distilled scorer loaded "
                         f"({len(scorer_weights)} tensors)")
        except Exception as exc:
            logging.warning(f"[BSAI VedaSparse] scorer load failed ({exc}); "
                            "heuristic TripPool scoring")
            scorer = None

    previous = m.model_options["transformer_options"].get("optimized_attention_override")
    if previous is not None:
        logging.info("[BSAI VedaSparse] chaining onto an existing attention override")

    if compose_with_foreign_patches:
        _install_compose_hooks(diffusion_model, "attn")

    m.model_options["transformer_options"]["optimized_attention_override"] = \
        make_veda_override(keep_frac=keep_frac, min_tokens=min_tokens,
                           sigma_start=sigma_start, sigma_end=sigma_end,
                           sink_conditioning=sink_conditioning,
                           head_tilings=head_tiling, tripool_mode=tripool_mode,
                           scorer=scorer, latent_dims=None, aspect=aspect,
                           force_dims=force_dims, verbose=verbose, previous=previous)
    m.model_options["transformer_options"]["bsai_vedasparse"] = {
        "keep_percent": keep_percent, "min_tokens": min_tokens,
        "sink": sink_conditioning, "tiling": list(head_tiling),
        "tripool": tripool_mode,
        "scorer": "distilled" if scorer is not None else "heuristic",
        "sigma_start": sigma_start, "sigma_end": sigma_end}
    reset_veda_stats()
    logging.info(f"[BSAI VedaSparse] applied: keep={keep_percent}% "
                 f"sink={sink_conditioning} tiling={head_tiling} "
                 f"tripool={tripool_mode} scorer={'distilled' if scorer else 'heuristic'}")
    return m


# ---------------------------------------------------------------------------
# 与外部 attention object-patch 的兼容（同 FastH3 思路，独立实现）
# ---------------------------------------------------------------------------

_COMPOSE_HOOKED = set()


def _compose_module_patch(module, patched_forward):
    stock = type(module).forward

    def forward(*args, **kwargs):
        options = kwargs.get("transformer_options")
        if not isinstance(options, dict):
            options = next((a for a in args
                            if isinstance(a, dict) and "bsai_vedasparse" in a), {})
        gate = options.get("bsai_vedasparse")
        x = args[0] if args else None
        tensor = x[0] if isinstance(x, list) and len(x) == 1 and torch.is_tensor(x[0]) else x
        take = gate is not None and torch.is_tensor(tensor) and tensor.device.type == "cuda"
        if take:
            tokens = tensor.shape[0] if tensor.ndim == 2 else tensor.shape[1]
            take = tokens >= gate.get("min_tokens", 0)
        if take:
            sigmas = options.get("sigmas")
            if sigmas is not None:
                sigma = float(sigmas[0])
                take = not (gate.get("sigma_start") is not None and sigma > gate["sigma_start"]) \
                       and not (gate.get("sigma_end") is not None and sigma < gate["sigma_end"])
        if take:
            if tensor is not x:
                x.clear()
                args = (tensor,) + args[1:]
            return stock(module, *args, **kwargs)
        return patched_forward(*args, **kwargs)

    forward._bsai_veda_composed = True
    return forward


def _install_compose_hooks(model, attn_attr):
    if id(model) in _COMPOSE_HOOKED:
        return

    def pre_hook(block, args):
        attn = getattr(block, attn_attr, None)
        if attn is None:
            return None
        fwd = attn.__dict__.get("forward")
        if fwd is None or getattr(fwd, "_bsai_veda_composed", False):
            return None
        if getattr(fwd, "_uses_optimized_attention", False):
            return None
        if getattr(fwd, "__func__", None) is type(attn).forward:
            return None
        attn.forward = _compose_module_patch(attn, fwd)
        _log_once(("composed", attn_attr),
                  f"composing with a patched {attn_attr}.forward; VedaSparse takes "
                  "eligible self-attention calls, the patch keeps the rest")
        return None

    for block in model.blocks:
        block.register_forward_pre_hook(pre_hook)
    _COMPOSE_HOOKED.add(id(model))
