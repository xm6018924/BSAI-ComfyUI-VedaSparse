# BSAI-ComfyUI-VedaSparse

**Veda 蒸馏稀疏注意力 for MiniMax-H3（ComfyUI 节点插件 / Custom Nodes）**
**Veda Distilled Sparse Attention for MiniMax-H3 — BSAI Series Plugin**

> 把 Veda（ByteDance + HKU, ICML 2026, arXiv:2605.30325）蒸馏稀疏注意力落地到 MiniMax-H3 的 ComfyUI 原生节点：每个视频 query tile 只对 top-k 个 key tile（默认 5–10%）精确计算，其余跳过；文本 / 音频 / 参考图 conditioning 行始终精确（dense），避免提示词与音轨退化。
>
> Brings Veda (ByteDance + HKU, ICML 2026, arXiv:2605.30325) distilled sparse attention to MiniMax-H3 as native ComfyUI nodes: each video query tile attends exactly to top-k key tiles (default 5–10%) and skips the rest; text / audio / reference conditioning rows always stay dense to protect prompts and audio.

由 [B 站视频《MiniMax H3 最新加速来了！十四秒视频提速二点七六倍！》](https://www.bilibili.com/video/BV1Uca46uExC/) 引出，基于 Veda 论文公开算法实现。

---

## 📖 一、插件说明 / Overview

### 核心技术 / Core Technology

| 组件 / Component | 作用 / Role | 相比 FastVideo VSA 的升级 / vs FastVideo VSA |
|---|---|---|
| **TripPool 评分**（统计感知 tile 评分） | 用 Avg⊕Max⊕Min 三统计量描述每个 tile，评分贴近全注意力 | VSA 只用平均池化稀释峰值信号；论文实测 tile 召回率 **66.4% vs 34.2%** |
| **Head-Aware Tiling**（按头时空分块） | 每头分配 `(pt,ph,pw)` 时空分块（`pt·ph·pw=64`），匹配不同头的时空关注偏好 | VSA 所有头统一 64-token 立方 tile，结构失配大 |
| **Tile-Skipping**（tile 跳过） | 每个 query tile 只对 top-k key tile 精确计算 | 执行层：把稀疏变成实际墙钟加速 |

### 实测加速 / Benchmarks（3×RTX4090，8 步去噪 + TabLoRA）

| 视频时长 / Clip | 端到端加速 / E2E | 注意力加速 / Attention |
|---|---|---:|
| 5.17 s | 1.57× | — |
| 10.1 s | 2.21× | — |
| **14.4 s** | **2.76×** | **6.66×** |
| 20 提示词平均 | 2.24× | 5.92× |
| 最强单条 | 3.08× | 6.87× |

---

## 📥 二、安装方法 / Installation

### 方法一：Git Clone（推荐 / Recommended）

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/xm6018924/BSAI-ComfyUI-VedaSparse.git
```

重启 ComfyUI（或点击「刷新节点列表 / Refresh Nodes」）。

### 方法二：手动复制 / Manual Copy

将 `BSAI-ComfyUI-VedaSparse` 文件夹放入 `ComfyUI/custom_nodes/`，重启 ComfyUI。

**依赖 / Dependencies**：仅需 ComfyUI 内置环境（torch、`comfy.utils`），无需额外 pip 包。需含 MiniMax-H3 模型支持的 ComfyUI 版本（≥ 0.33.0）。

---

## 🎮 三、节点说明 / Nodes

### BSAIVedaSparsePatch

| 参数 / Param | 默认 / Default | 说明 / Description |
|---|---|---|
| `model` | — | H3 diffusion model（UNETLoader 输出） |
| `enabled` | true | 启用稀疏；false = 模型原样直通 |
| `keep_percent` | 5.0 | 每个 query tile 保留的 key tile 百分比（5 = 95% 稀疏） |
| `head_tiling` | dual-fast (2 groups) | 按头分块预设：balanced / head-aware / dual-fast / temporal-first / spatial-first / extreme-spatial / custom |
| `custom_tiling` | `4,4,4;8,4,2;2,4,8;4,8,2` | custom 分块表，每个 `pt,ph,pw` 乘积须 = 64 |
| `tripool_mode` | triplet | tile 描述符：triplet（论文最优）/ maxmin / avg |
| `scorer_weights` | None (heuristic) | 蒸馏预测器权重（放 `models/veda_scorers/`） |
| `aspect` | auto | 画幅（解析 (T,H,W) 几何） |
| `force_dims` | "" | 手动指定 video latent 尺寸 `T,H,W`（一般留空） |
| `start_percent` | 0.0 | 去噪起点之前保持 dense 预热（**建议 0.2，保护初始结构与音频**） |
| `end_percent` | 1.0 | 去噪终点之后保持 dense |
| `min_tokens` | 4096 | 序列 token 低于此值走 dense（避免短片 overhead） |
| `sink_conditioning` | exact_kv_and_rows | 文本/音频/参考行保持精确：exact（推荐）/ off |
| `verbose` | false | 详细日志 |

### BSAIVedaSparseStats

只读诊断节点，输出 `sparse/dense/heuristic/distilled/errors/fallback_1d` 计数——`sparse > 0 且 errors = 0` 即稀疏正常生效。

### 评分器两种模式 / Scorer Modes

| 模式 / Mode | 何时用 / When | 说明 / Notes |
|---|---|---|
| **heuristic（默认）** | 开箱即用 | φ=identity 的 TripPool 描述符直接点积评分（论文 eq.6 单位投影），无需权重 |
| **distilled（可选）** | 官方权重发布后 | 加载 VedaSparse 蒸馏预测器（263MB FP8，每头独立 MLP 投影 φ_q/φ_k），key 规范见 `VEDA_SCORER_KEY_SPEC` |

---

## 🔌 四、使用方法 / Usage

### 最小接入 / Minimal

```
UNETLoader (H3) ──▶ BSAIVedaSparsePatch ──▶ BasicGuider ──▶ SamplerCustomAdvanced
                        │                       │
                        └──▶ BasicScheduler ──────┘
```

所有参数有默认值，接上即生效。

### 推荐参数档 / Recommended Presets

| 场景 / Scenario | keep_percent | head_tiling | tripool_mode |
|---|---|---|---|
| 默认速度档（Turbo LoRA，速度优先） | **5** | dual-fast (2 groups) | triplet |
| 质量优先 / Quality first | 10~20 | head-aware (4 groups) | triplet |
| 极速档 / Fastest | 5 | balanced (4,4,4) | triplet |

> `keep_percent` 越低越快；5% 为默认速度档，与 10% 画质几乎无差别。

### 音频安全要点 / Audio-Safe Tips

- `start_percent` 建议 **0.2**：前 20% 去噪保持 dense，保护初始结构与音频。
- `sink_conditioning` 保持 `exact_kv_and_rows`：音频/文本/参考行始终精确，不受稀疏影响。
- 低步数（≤3 步）或激进量化底模（GGUF / NVFP4）下 H3 音频 latent 易退化；音频优先请用 **int8 系底模 + ≥4 步**。

---

## 🧩 五、示例工作流 / Example Workflow

**`example_workflows/BSAI_H3_VedaSparse_AllInOne_v1.0.json`** —— H3 多合一示例工作流（文生视频 / 图生视频 / 参考生视频 三合一）。

### 结构 / Structure

- **统一模型链**：`UNETLoader (minimax_h3_hybrid_fl2va_ref2va_b25-49-int8)` → `LoraLoader (官方 Turbo 4步 LoRA)` → `BSAIVedaSparsePatch (keep 5%, dual-fast, start 0.2)` → `BasicGuider` / `BasicScheduler (beta, 4步)` + `KSamplerSelect (euler)`。
- **文生视频 / Text-to-Video**：图片输入区 LoadImage 设为 bypass（`mode: 4`）即纯文生。
- **图生视频 / 参考生视频 / Image-to-Video & Reference-to-Video**：`easy ifElse` 开关（`PrimitiveBoolean`）切换：
  - `false` → 图生视频（`MiniMaxH3ImageToVideo`，首尾帧渐变）；
  - `true` → 参考生视频（`MiniMaxH3ReferenceToVideo`，参考图/参考视频 + 参考音频）。
- 提示词走 `BSAI_H3_PromptTemplate`（需 [BSAI-MiniMAX-H3-Prompt](https://github.com/xm6018924) 插件）多模态融合模板。
- 输出：`CreateVideo (24fps + 音轨)` → `SaveVideo`。

### 所需模型 / Required Models

| 用途 / Use | 文件 / File | 位置 / Path |
|---|---|---|
| 底模 UNET | `minimax_h3_hybrid_fl2va_ref2va_b25-49-int8.safetensors` | `models/diffusion_models/` |
| Turbo LoRA | `minimax_h3_fl2v_turbo_4step_v1.2_768p_comfyui_bf16.safetensors` | `models/loras/` |
| 文本编码器 / CLIP | `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` | `models/text_encoders/` |
| 视频 VAE | `minimax_h3_video_vae_int8_convrot.safetensors` | `models/vae/` |
| 音频 VAE | `minimax_h3_audio_vae_fp32.safetensors` | `models/vae/` |

> 使用其他 H3 底模（如官方 `minimax_h3_fl2va_pruned_int8_convrot`）时，直接改 UNETLoader 模型名即可，其余链路不变。
> Swapping in another H3 base model only requires changing the UNETLoader model name.

---

## ⚙️ 六、实现说明 / Implementation

- `veda_engine.py`：核心引擎
  - `_tripool()`：论文 eq.5 TripPool 描述符（Avg⊕Max⊕Min）
  - `_heuristic_scores()`：论文 eq.6 单位投影形式评分
  - `_tile_3d()`：Head-Aware Tiling（pt·ph·pw=64），返回 tile 与 key mask
  - `_veda_sparse()`：tile-skipping 稀疏注意力（conditioning 行始终 dense）
  - `_DistilledScorer`：蒸馏预测器（预留）
  - `install_h3_span()`：发布 H3 packed 布局的 video/audio span 与 latent 尺寸（幂等）
- `nodes.py`：ComfyUI 节点（仅 `ModelPatcher.clone()` / `model_options` 注入，不改 ComfyUI 内部源码）
- 引擎经 `transformer_options["optimized_attention_override"]` 接入；若已存在其他 attention patch（如 FastH3 VSA），自动链式组合。
- **gather 显存优化**：按 (batch,head,tile) 行号高级索引整 tile 连续拷贝 + 动态 CHUNK 分块（峰值约 300MB），消除原全头 gather 的 6.7GB 峰值；int32 索引。RTX 5090 Laptop 实测默认档约 422ms/层、4 组 keep=10% 约 777ms/层、1 组 keep=5% 约 404ms/层（dense SDPA 668ms/层）。

### 正确性验证 / Correctness

单元测试（`test_veda_engine.py`，24 项全过）含最强验证：**keep=100% 时稀疏注意力精确还原 dense 注意力（最大误差 2.4e-7）**。

---

## 🛠️ 七、兼容性 / Compatibility

- ComfyUI ≥ 0.33.0（含 MiniMax-H3 支持）。
- 与 FastH3 / TaoMate / T8 块缓存等外部注意力补丁链式兼容。
- 非 H3 扩散模型明确报错并 dense 回退，不影响其他工作流。

---

## ⚠️ 八、常见问题 / FAQ

**Q1：音频变成低频轰鸣 / Audio is low-frequency noise?**
低步数（≤4 步）+ 激进量化（GGUF/NVFP4）下 H3 音频 latent 易退化。请：①使用 int8 系底模；②`start_percent=0.2` 前 20% dense；③步数 ≥ 4（4 步 Turbo LoRA 配 4 步 beta/ladder 调度）。

**Q2：显存不足 / OOM?**
引擎已 CHUNK 分块（约 300MB 峰值）。仍 OOM 时降低分辨率/帧数或关闭其他占用显存的程序。

**Q3：看不到加速 / No speedup?**
确认 `enabled=true`、token 数 ≥ `min_tokens`（4096）、且为 H3 模型。短片段（<1s）稀疏收益小属正常。

---

## 📄 License

MIT（代码）。Veda 论文与 MiniMax-H3 模型版权归各自作者/公司所有。

---

## 🙏 参考与致谢 / References & Credits

- 视频 / Video：AI绘画KK《MiniMax H3 最新加速来了！十四秒视频提速二点七六倍！》https://www.bilibili.com/video/BV1Uca46uExC/
- 论文 / Paper：Veda: *Scalable Video Diffusion via Distilled Sparse Attention*（ICML 2026 · arXiv:2605.30325）
- FastVideo VSA（评分/几何参考）
- MiniMax H3（底模生态）
- 姊妹插件 / Sister plugins：BSAI-ComfyUI-FastH3、BSAI-ComfyUI-TaoMate、BSAI-ComfyUI-vdn-minimax-h3、BSAI-ComfyUI-MiniMax-H3-PDD-Acc
