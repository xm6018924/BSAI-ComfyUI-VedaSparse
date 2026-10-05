"""BSAI-ComfyUI-VedaSparse — Veda 蒸馏稀疏注意力（MiniMax-H3 加速）节点套件。

围绕 Veda（ByteDance + HKU，ICML 2026，arXiv:2605.30325）蒸馏稀疏注意力在
MiniMax-H3 上的落地：

  * BSAIVedaSparsePatch  Veda 稀疏注意力补丁：TripPool 评分（Avg⊕Max⊕Min）
                        + Head-Aware Tiling（每头时空分块）+ tile-skipping
                        稀疏注意力。每个视频 query tile 只对 top-k 个 key tile
                        （默认 10%）精确计算，文本/音频/参考 conditioning 行
                        始终精确。视频实测（3×RTX4090）：14.4s 视频端到端
                        2.76×、注意力 6.66×；20 提示词平均端到端 2.24×。
  * BSAIVedaSparseStats 稀疏命中统计（只读诊断）

⚠ v3.1.2 硬件熔断 + v3.2 最小化重写
==============================
用户（RTX 5090 Laptop 24GB）在 H3 4 步采样中：v3.0 / v3.1 / v3.1.1 均在第 3~4 步
触发硬件级 GPU 重启（不是普通 OOM）。原因疑似 VedaSparse 与 FastH3 / Sol-H3 /
VDN-H3 / FirstBlockCache 等其它 H3 加速插件叠加，导致 GPU 资源争用。

v3.1.2 用 hard-off 兜底：完全 disabled 但保留插件结构。
v3.2 真正最小化重写：
  * 触发条件严格（min_tokens=8192 + video span 必须够长 + sigma 窗口限制）
  * CHUNK=8 切片，单次 forward 临时 < 200MB
  * OOM 后 _FALLBACK_TO_DENSE 整进程回退（不会再硬件崩）
  * 算法仍是 Veda 论文核心：TripPool (Avg⊕Max⊕Min) + top-k tile skipping

如果 v3.2 仍崩：手动编辑 veda_engine.py 把 `_ENABLE_V32_SPARSE = False` 即可
回到 hard-off 模式。

用法：UNETLoader -> BSAIVedaSparsePatch -> BasicGuider。v3.1.2 下本节点等价
直通，但保留位置以便后续重新启用 sparse。

安装：把本目录放到 ComfyUI/custom_nodes/BSAI-ComfyUI-VedaSparse，重启 ComfyUI。

v3.4.2 vendored 官方 Veda-on-ComfyUI（pull-and-go）
===================================================
仓库自带的 `official_veda_vendor/` 是官方 Veda-on-ComfyUI（MIT License,
https://github.com/veda-sparse/Veda-on-ComfyUI）的精简副本。当
`custom_nodes/Veda-on-ComfyUI` 官方插件**未安装**时，本插件自动把官方
`VedaSparseAttention` 节点合并注册进 NODE_CLASS_MAPPINGS——因此 git pull /
clone 本仓库后打开引用官方节点的 v6.1 等工作流**不再报缺失节点包**；
官方插件已安装时自动跳过，避免重复注册。

依赖：triton-windows>=3.0（与官方插件一致）；预测器权重放
`ComfyUI/models/veda/minimax_h3_t2va_veda_8nfe_600step_preview_fp8.safetensors`
（hf-mirror: https://hf-mirror.com/Veda-Sparse/Minimax-H3-T2VA-Veda-8NFE-600Step-Preview/resolve/main/minimax_h3_t2va_veda_8nfe_600step_preview_fp8.safetensors）
"""

import logging
import os
import sys as _sys

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

_OFFICIAL_VEDA_VENDORED = False


def _maybe_vendor_official_veda():
    """注册 vendored 官方 VedaSparseAttention（仅当官方插件未安装时）。"""
    global _OFFICIAL_VEDA_VENDORED
    _vendor = os.path.join(os.path.dirname(os.path.abspath(__file__)), "official_veda_vendor")
    if not os.path.isdir(_vendor):
        return
    try:
        # 官方插件已安装（custom_nodes/Veda-on-ComfyUI 存在）→ 跳过，避免重复注册
        _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if os.path.isdir(os.path.join(_root, "Veda-on-ComfyUI")):
            logging.info("[BSAI VedaSparse] 官方 Veda-on-ComfyUI 已安装，跳过 vendored 注册")
            return
        _parent = os.path.dirname(_vendor)
        if _parent not in _sys.path:
            _sys.path.insert(0, _parent)
        from official_veda_vendor.veda_comfy import nodes as _veda_official
        _veda_official.register_model_folder()
        NODE_CLASS_MAPPINGS["VedaSparseAttention"] = _veda_official.VedaSparseAttention
        NODE_DISPLAY_NAME_MAPPINGS["VedaSparseAttention"] = "Veda Sparse Attention (MiniMax H3)"
        _OFFICIAL_VEDA_VENDORED = True
        logging.info("[BSAI VedaSparse] vendored 官方 VedaSparseAttention 已注册（pull-and-go）")
    except Exception as e:  # noqa: BLE001 —— vendored 失败不阻塞本插件
        logging.warning("[BSAI VedaSparse] vendored 官方 Veda 注册失败（%s）", e)


_maybe_vendor_official_veda()
