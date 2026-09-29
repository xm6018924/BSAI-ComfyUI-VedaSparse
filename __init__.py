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
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
