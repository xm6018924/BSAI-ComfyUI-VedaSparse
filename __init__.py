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

评分器两种模式：
  * heuristic（默认）：TripPool 描述符直接点积评分，无需权重，开箱即用。
  * distilled（可选）：加载 VedaSparse 蒸馏预测器权重（263MB FP8，每头独立
    MLP 投影 φ_q/φ_k），权重放入 ComfyUI/models/veda_scorers/。

用法：UNETLoader -> BSAIVedaSparsePatch -> BasicGuider。可与官方 Turbo 8 步
LoRA（minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16）叠加，同时获得
8 步蒸馏 + Veda 稀疏注意力。

安装：把本目录放到 ComfyUI/custom_nodes/BSAI-ComfyUI-VedaSparse，重启 ComfyUI。
全部节点仅通过 ModelPatcher.clone() / model_options 注入，不修改 ComfyUI 内部源码。
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
