"""BSAI-ComfyUI-VedaSparse — Veda 蒸馏稀疏注意力 ComfyUI 节点套件。

把 Veda（ByteDance + HKU，ICML 2026，arXiv:2605.30325）蒸馏稀疏注意力落地到
MiniMax-H3 的 ComfyUI 原生节点：

  * BSAIVedaSparsePatch  Veda 稀疏注意力补丁（TripPool 评分 + Head-Aware Tiling
                         + tile-skipping），注入 H3 模型
  * BSAIVedaSparseStats Veda 稀疏命中统计（只读诊断）

用法
----
在 H3 工作流中：UNETLoader -> BSAIVedaSparsePatch -> BasicGuider。
与官方 Turbo 8 步 LoRA（minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16）叠加，
可同时获得 8 步蒸馏 + Veda 稀疏注意力（视频实测 14.4s 视频端到端 2.76×、
注意力 6.66×）。

评分器两种模式：
  * heuristic（默认）：TripPool（Avg⊕Max⊕Min）描述符直接点积评分，无需权重。
  * distilled（可选）：加载 VedaSparse 蒸馏预测器权重（263MB FP8，每头独立
    MLP 投影）。权重放入 ComfyUI/models/veda_scorers/，key 规范见
    veda_engine.VEDA_SCORER_KEY_SPEC。官方权重公开后无需改代码即可启用。

全部节点仅通过 ModelPatcher.clone() / model_options 注入，不修改 ComfyUI
内部源码，升级无碍。
"""

import logging
import os

import folder_paths
from comfy.utils import load_torch_file

from . import veda_engine as _ve

# head-aware tiling 预设
TILING_BALANCED = "balanced (4,4,4)"
TILING_HEAD_AWARE = "head-aware (4 groups)"
TILING_DUAL = "dual-fast (2 groups)"
TILING_TEMPORAL = "temporal-first (8,4,2)"
TILING_SPATIAL = "spatial-first (2,4,8)"
TILING_EXTREME_SPATIAL = "extreme-spatial (1,8,8)"
TILING_CUSTOM = "custom"
TILING_PRESETS = (TILING_DUAL, TILING_HEAD_AWARE, TILING_BALANCED, TILING_TEMPORAL,
                  TILING_SPATIAL, TILING_EXTREME_SPATIAL, TILING_CUSTOM)

_PRESET_TILINGS = {
    TILING_BALANCED: ((4, 4, 4),),
    TILING_DUAL: ((4, 4, 4), (8, 4, 2)),
    TILING_HEAD_AWARE: ((4, 4, 4), (8, 4, 2), (2, 4, 8), (4, 8, 2)),
    TILING_TEMPORAL: ((8, 4, 2),),
    TILING_SPATIAL: ((2, 4, 8),),
    TILING_EXTREME_SPATIAL: ((1, 8, 8),),
}

TRIPOOL_TRIPLET = "triplet (avg+max+min)"
TRIPOOL_MAXMIN = "maxmin (max+min)"
TRIPOOL_AVG = "avg only (VSA-like)"
TRIPOOL_MODES = (TRIPOOL_TRIPLET, TRIPOOL_MAXMIN, TRIPOOL_AVG)
_TRIPOOL_VALUES = {TRIPOOL_TRIPLET: "triplet",
                   TRIPOOL_MAXMIN: "maxmin",
                   TRIPOOL_AVG: "avg"}

SINK_OFF = "off"
SINK_EXACT = "exact_kv_and_rows"
SINK_MODES = (SINK_EXACT, SINK_OFF)

ASPECTS = ("auto", "16:9", "9:16", "4:3", "3:4", "1:1")

_HEAD_TILING_CACHE = {}


def _parse_tiling(preset, custom):
    if preset in _PRESET_TILINGS:
        return _PRESET_TILINGS[preset]
    # custom: "pt,ph,pw" 或 "pt,ph,pw;pt,ph,pw;..."
    out = []
    for part in custom.split(";"):
        part = part.strip()
        if not part:
            continue
        nums = [int(x.strip()) for x in part.split(",")]
        if len(nums) != 3:
            raise ValueError(f"[BSAI VedaSparse] 非法 tiling 配置 '{part}'，"
                             "应为 'pt,ph,pw'（如 4,4,4）。")
        pt, ph, pw = nums
        if pt * ph * pw != 64:
            raise ValueError(f"[BSAI VedaSparse] tiling {nums} 乘积 {pt*ph*pw} != 64"
                             "（硬件 tile 大小）。")
        out.append((pt, ph, pw))
    if not out:
        raise ValueError("[BSAI VedaSparse] custom tiling 为空。")
    return tuple(out)


def _parse_force_dims(s):
    if not s or not s.strip():
        return None
    nums = [int(x.strip()) for x in s.split(",")]
    if len(nums) != 3 or any(n <= 0 for n in nums):
        raise ValueError(f"[BSAI VedaSparse] force_dims '{s}' 非法，应为 'T,H,W'"
                         "（正整数）。")
    return tuple(nums)


def _load_scorer(name):
    """从 ComfyUI/models/veda_scorers/ 加载蒸馏预测器权重。"""
    if not name or name == "None (heuristic TripPool)":
        return None
    path = folder_paths.get_full_path("veda_scorers", name)
    if path is None:
        # 兜底：尝试 loras 目录
        path = folder_paths.get_full_path("loras", name)
    if path is None or not os.path.isfile(path):
        logging.warning(f"[BSAI VedaSparse] 找不到评分器权重 {name}；回退启发评分")
        return None
    return load_torch_file(path)


class BSAIVedaSparsePatch:
    """Veda 蒸馏稀疏注意力补丁。

    每个视频 query tile 只对 top-k 个 key tile（默认 10%）精确计算，其余跳过；
    文本/音频/参考 conditioning 行始终 dense。TripPool 评分 + Head-Aware
    Tiling 相对 FastVideo VSA 提升 tile 召回率（论文 66.4% vs 34.2%），
    相同稀疏度下画面更稳，或相同质量下可开更高稀疏度。
    """

    @classmethod
    def INPUT_TYPES(cls):
        try:
            scorers = ["None (heuristic TripPool)"] + sorted(
                folder_paths.get_filename_list("veda_scorers"))
        except Exception:
            scorers = ["None (heuristic TripPool)"]
        return {"required": {
            "model": ("MODEL",),
            "enabled": ("BOOLEAN", {
                "default": True,
                "label_on": "启用 Veda 稀疏",
                "label_off": "直通（不加速）",
                "tooltip": "关闭时模型原样通过，不影响任何现有工作流。"}),
            "keep_percent": ("FLOAT", {
                "default": 5.0, "min": 1.0, "max": 100.0, "step": 1.0,
                "tooltip": "每个 query tile 保留的关键 key tile 百分比。10 = 90% 稀疏"
                           "（视频实测配置）；8 步 V2 + TabLoRA 建议 10~20。"}),
            "head_tiling": (TILING_PRESETS, {
                "default": TILING_DUAL,
                "tooltip": "Head-Aware Tiling 预设。head-aware 按头循环分配 4 种"
                           "时空分块；balanced 全部头用 (4,4,4) 立方（≈VSA 几何 +"
                           "TripPool 评分）；temporal/spatial 强调时间或空间结构。"}),
            "custom_tiling": ("STRING", {
                "default": "4,4,4;8,4,2;2,4,8;4,8,2",
                "multiline": False,
                "tooltip": "head_tiling=custom 时的分块表，分号分隔多个 "
                           "'pt,ph,pw'，每个 pt*ph*pw 必须 = 64。"}),
            "tripool_mode": (TRIPOOL_MODES, {
                "default": TRIPOOL_TRIPLET,
                "tooltip": "tile 描述符统计量：triplet=Avg⊕Max⊕Min（论文最优，"
                           "保留 tile 内峰值信号）；avg=仅平均（VSA 式，易稀释"
                           "峰值，作对比用）。"}),
            "scorer_weights": (scorers, {
                "default": "None (heuristic TripPool)",
                "tooltip": "蒸馏预测器权重（可选）。放入 "
                           "ComfyUI/models/veda_scorers/。有权重=蒸馏评分；"
                           "无权重=启发 TripPool 评分（开箱即用）。"}),
            "aspect": (ASPECTS, {
                "default": "auto",
                "tooltip": "生成画幅，用于解析 video span 的 (T,H,W) 几何。"
                           "auto 优先用 H3 布局自带的 latent 尺寸。"}),
            "force_dims": ("STRING", {
                "default": "",
                "multiline": False,
                "tooltip": "手动指定 video latent 尺寸 'T,H,W'（可选，一般留空）。"
                           "例如 30,24,42。留空=自动解析。"}),
            "start_percent": ("FLOAT", {
                "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                "tooltip": "去噪进度百分比起点；之前保持 dense 预热，保护初始结构。"}),
            "end_percent": ("FLOAT", {
                "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                "tooltip": "去噪进度百分比终点；之后保持 dense。"}),
            "min_tokens": ("INT", {
                "default": 4096, "min": 0, "max": 1000000,
                "tooltip": "序列 token 数低于此值时走 dense（短片加速不明显，"
                           "避免 overhead）。"}),
            "sink_conditioning": (SINK_MODES, {
                "default": SINK_EXACT,
                "tooltip": "conditioning（文本/音频/参考）行保持精确：exact=KV 和"
                           "query 行都精确（推荐）；off=仅 query 行精确。"}),
            "verbose": ("BOOLEAN", {
                "default": False,
                "tooltip": "输出详细日志（含引擎降级原因）。"}),
        }}

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("MODEL",)
    FUNCTION = "apply_veda"
    CATEGORY = "BSAI/VedaSparse"

    def apply_veda(self, model, enabled, keep_percent, head_tiling,
                   custom_tiling, tripool_mode, scorer_weights, aspect,
                   force_dims, start_percent, end_percent, min_tokens,
                   sink_conditioning, verbose):
        tilings = _parse_tiling(head_tiling, custom_tiling)
        dims = _parse_force_dims(force_dims)
        sd = _load_scorer(scorer_weights)
        return (_ve.apply_veda(
            model, enabled=enabled, keep_percent=keep_percent,
            min_tokens=min_tokens, start_percent=start_percent,
            end_percent=end_percent, sink_conditioning=sink_conditioning,
            head_tiling=tilings, tripool_mode=_TRIPOOL_VALUES[tripool_mode],
            scorer_weights=sd, aspect=aspect if aspect != "auto" else "16:9",
            force_dims=dims, verbose=verbose),)


class BSAIVedaSparseStats:
    """Veda 稀疏注意力命中统计（只读诊断）。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("summary", "sparse_calls", "dense_calls", "heuristic_calls",
                    "distilled_calls")
    FUNCTION = "stats"
    CATEGORY = "BSAI/VedaSparse"

    def stats(self):
        s = _ve.veda_stats()
        summary = (f"sparse={s['sparse']} dense={s['dense']} "
                   f"heuristic={s['heuristic']} distilled={s['distilled']} "
                   f"errors={s['errors']} fallback_1d={s['fallback_1d']}")
        return (summary, str(s["sparse"]), str(s["dense"]),
                str(s["heuristic"]), str(s["distilled"]))


NODE_CLASS_MAPPINGS = {
    "BSAIVedaSparsePatch": BSAIVedaSparsePatch,
    "BSAIVedaSparseStats": BSAIVedaSparseStats,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BSAIVedaSparsePatch": "BSAI VedaSparse Patch",
    "BSAIVedaSparseStats": "BSAI VedaSparse Stats",
}
