# BSAI-ComfyUI-VedaSparse

**Veda 蒸馏稀疏注意力 for MiniMax-H3（ComfyUI 节点插件 / Custom Nodes）v3.4**
**Veda Distilled Sparse Attention for MiniMax-H3 — BSAI Series Plugin v3.4**

> 把 Veda（ByteDance + HKU, ICML 2026, arXiv:2605.30325）蒸馏稀疏注意力落地到 MiniMax-H3 的 ComfyUI 原生节点：每个视频 query tile 只对 top-k 个 key tile（默认 5%）精确计算，其余跳过；文本 / 音频 / 参考图 conditioning 行始终精确（dense），避免提示词与音轨退化。
>
> Brings Veda (ByteDance + HKU, ICML 2026, arXiv:2605.30325) distilled sparse attention to MiniMax-H3 as native ComfyUI nodes: each video query tile attends exactly to top-k key tiles (default 5%) and skips the rest; text / audio / reference conditioning rows always stay dense to protect prompts and audio.

由 [B 站视频《MiniMax H3 最新加速来了！十四秒视频提速二点七六倍！》](https://www.bilibili.com/video/BV1Uca46uExC/) 引出，基于 Veda 论文公开算法实现，在 **RTX 5090 Laptop 24GB** 用户机上完成多轮 A/B 实测与定稿。

---

## 📖 一、插件说明 / Overview

### 核心技术 / Core Technology

| 组件 / Component | 作用 / Role | 相比 FastVideo VSA 的升级 / vs FastVideo VSA |
|---|---|---|
| **TripPool 评分**（统计感知 tile 评分） | 用 Avg⊕Max⊕Min 三统计量描述每个 tile，评分贴近全注意力 | VSA 只用平均池化稀释峰值信号；论文实测 tile 召回率 **66.4% vs 34.2%** |
| **Head-Aware Tiling**（按头时空分块） | 每头分配 `(pt,ph,pw)` 时空分块（`pt·ph·pw=64`），匹配不同头的时空关注偏好 | VSA 所有头统一 64-token 立方 tile，结构失配大 |
| **Tile-Skipping**（tile 跳过） | 每个 query tile 只对 top-k key tile 精确计算 | 执行层：把稀疏变成实际墙钟加速 |

### v3.4 关键升级 / v3.4 Highlights

| 升级点 / Upgrade | 说明 / Description |
|---|---|
| **Monkey-patch 接入** | 直接 patch H3 `Attention/Block.forward` 的 attention 段（不再依赖 `optimized_attention_override` 链），与任何外部注意力插件零冲突 |
| **cond 行恒 dense** | 文本 / 音频 / 参考图 conditioning 行永远精确计算，稀疏只作用于视频 query 行——提示词与音轨零劣化 |
| **低步数全步稀疏** | ≤5 步调度（含 Turbo 4 步）自动全步进入稀疏，无需手动改 sigma 窗口 |
| **sigma_window [0, 0.98]** | 默认窗口覆盖去噪全程（首步不进稀疏由用户按需放宽，实测 0.98 上限安全） |
| **FFN 稀疏默认关闭** | `ffn_sparse=False`——FFN 硬跳稀疏曾在低步数下造成画面噪点（已实测复现并否决） |
| **sink = exact_kv_and_rows** | 文本/音频/参考行 key 与 row 均保持精确，音频安全 |

### v3.4.1 官方 Veda 全兼容（不互斥）/ v3.4.1 Official Veda Coexistence

| 升级点 / Upgrade | 说明 / Description |
|---|---|
| **dense 透传 override** | `dense()` / `dense_tensors()` 不再移除 `optimized_attention_override`——官方 Veda-on-ComfyUI 节点正是通过该键注入 attention override；旧版无条件 pop 会吞掉官方 Veda（互斥根源）。v3.4 是 `Attention.forward` 级 monkey-patch，官方是调用点级 override，透传不会递归回本 forward，安全 |
| **override_priority 让路** | 新参数 `override_priority`：`auto`（默认，检测到外部 override 即让路 dense，由官方 Veda 接管，避免双重稀疏叠加）/ `veda34`（强制 v3.4 优先，忽略外部 override）/ `official`（总是让路）。三种模式均可与官方 Veda 共存，**不再互斥** |
| **实测（RTX 5090 Laptop 24GB · 8 步 Turbo）** | v6.1 工作流链式接入官方 Veda（90% 稀疏）→ FastH3 VSA（兜底）→ 原生：`Attention computed 25.9%（跳过 74.1%）`，400 次调用全稀疏，8 步 2:30 完成；Veda 拒绝的调用自动回落 FastH3，零冲突 |

### v3.4.2 vendored 官方 Veda-on-ComfyUI（pull-and-go）/ v3.4.2 Vendored Official Veda (Pull-and-Go)

| 升级点 / Upgrade | 说明 / Description |
|---|---|
| **仓库自带官方节点** | 仓库新增 `official_veda_vendor/`——官方 [Veda-on-ComfyUI](https://github.com/veda-sparse/Veda-on-ComfyUI)（MIT License）的精简副本。**git pull / clone 本仓库后，`VedaSparseAttention` 节点自动可用，打开 v6.1 等引用官方节点的工作流不再报「缺失节点包 / missing node package」** |
| **自动去重** | 检测到 `custom_nodes/Veda-on-ComfyUI` 官方插件已安装时自动跳过 vendored 注册（不重复注册）；未安装时自动合并注册——C 盘（官方已装）与 G 盘/其他电脑（未装）行为均正确 |
| **predictor 下载指引** | 预测器权重（275MB）需放在 `ComfyUI/models/veda/`：`minimax_h3_t2va_veda_8nfe_600step_preview_fp8.safetensors`。下载（hf-mirror）：`https://hf-mirror.com/Veda-Sparse/Minimax-H3-T2VA-Veda-8NFE-600Step-Preview/resolve/main/minimax_h3_t2va_veda_8nfe_600step_preview_fp8.safetensors`；SHA-256 `2a8d8845c5342756a2781e8e69563940e4bb573c9a40ebb534915ff8fd76573a` |
| **依赖** | vendored 官方节点运行需 `triton-windows>=3.0`（Windows，与官方插件一致）；缺失时节点注册不受影响，运行时 fallback 到 eager（详见官方文档） |

### 实测加速 / Benchmarks（论文环境：3×RTX4090，8 步去噪 + TabLoRA）

| 视频时长 / Clip | 端到端加速 / E2E | 注意力加速 / Attention |
|---|---|---:|
| 5.17 s | 1.57× | — |
| 10.1 s | 2.21× | — |
| **14.4 s** | **2.76×** | **6.66×** |
| 20 提示词平均 | 2.24× | 5.92× |
| 最强单条 | 3.08× | 6.87× |

### 本机 A/B 验收（RTX 5090 Laptop 24GB · turbo4step · keep=5%）/ Verified on Dev Machine

| 版本 / Version | 总耗时 / Total | 画面 / Video | 音频 / Audio |
|---|---|---|---:|
| **v3.4 稀疏开** | 202.70 s（热）/ 314.33 s（冷） | 无噪点、无劣化 | 正常（sub<100Hz ≈ 5%） |
| **直通（enabled=false）** | 219.58 s / 24.18 s/it | 同 | 正常（sub ≈ 4%） |
| **结论 / Verdict** | **零额外开销，纯收益** | 零劣化 | 零劣化 |

> 在同机同配置下，v3.4 稀疏开 vs 直通：画面逐帧无差异、音频频谱一致、速度不劣于直通——稀疏加速是**纯收益**。

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

**依赖 / Dependencies**：仅需 ComfyUI 内置环境（torch、`comfy.utils`），无需额外 pip 包。需含 MiniMax-H3 模型支持的 ComfyUI 版本（≥ 0.33.0）。 使用 vendored 官方 Veda 节点时另需 `triton-windows>=3.0`（Windows）。

---

## 🎮 三、节点说明 / Nodes

### BSAIVedaSparsePatch（v3.4）

| 参数 / Param | 默认 / Default | 说明 / Description |
|---|---|---|
| `model` | — | H3 diffusion model（UNETLoader 输出） |
| `enabled` | false | 启用稀疏；false = 模型原样直通（保留做 A/B 基线） |
| `keep_percent` | 5.0 | 每个 query tile 保留的 key tile 百分比（5 = 95% 稀疏） |
| `head_tiling` | dual-fast (2 groups) | 按头分块预设：balanced / head-aware / dual-fast / temporal-first / spatial-first / extreme-spatial / custom |
| `custom_tiling` | `4,4,4;8,4,2;2,4,8;4,8,2` | custom 分块表，每个 `pt,ph,pw` 乘积须 = 64 |
| `tripool_mode` | triplet | tile 描述符：triplet（论文最优）/ maxmin / avg |
| `scorer_weights` | None (heuristic) | 蒸馏预测器权重（放 `models/veda_scorers/`） |
| `aspect` | auto | 画幅（解析 (T,H,W) 几何） |
| `force_dims` | "" | 手动指定 video latent 尺寸 `T,H,W`（一般留空） |
| `sigma_window` | `0.000, 0.980` | 稀疏生效的 sigma 窗口（去噪起点前保持 dense）；放宽上限到 1.0 可让首步也稀疏，需自行实测音频安全性 |
| `start_percent` | 0.2 | 旧版兼容参数（v3.4 由 sigma_window 控制，保留不影响） |
| `end_percent` | 1.0 | 旧版兼容参数 |
| `max_sparse_tokens` | 16384 | 序列 token 超过此值才启用稀疏（短片自动 dense，避免 overhead） |
| `ffn_sparse` | false | FFN 稀疏总开关——**默认关**（硬跳 FFN 曾致画面噪点，已否决） |
| `ffn_keep` | 60.0 | FFN 稀疏保留比例（仅 ffn_sparse=true 时生效） |
| `sink_conditioning` | exact_kv_and_rows | 文本/音频/参考行保持精确：exact（推荐）/ off |
| `override_priority` | auto | 与外部 attention override（官方 Veda-on-ComfyUI）协作：auto=检测到外部 override 时 v3.4 让路（官方接管）/ veda34=强制 v3.4 优先（忽略外部 override）/ official=总是让路。三种模式均可共存，不互斥 |
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
UNETLoader (H3) ──▶ LoraLoaderModelOnly (Turbo 4步) ──▶ BSAIVedaSparsePatch ──▶ BasicGuider ──▶ SamplerCustomAdvanced
                                                              │                       │
                                                              └──▶ BasicScheduler ──────┘
```

所有参数有默认值，接上即生效。

### 推荐参数档 / Recommended Presets（v3.4 实测定稿）

| 场景 / Scenario | keep_percent | head_tiling | tripool_mode | sigma_window |
|---|---|---|---|---|
| **正式档（Turbo 4 步，音频优先）** | **5** | dual-fast (2 groups) | triplet | `0.000, 0.980` |
| 质量优先 / Quality first | 10~20 | head-aware (4 groups) | triplet | `0.000, 0.980` |
| 极速档 / Fastest | 5 | balanced (4,4,4) | triplet | `0.000, 1.000`（需实测） |

> `keep_percent` 越低越快；5% 为正式档，与 10% 画质几乎无差别。

### 音频安全要点 / Audio-Safe Tips（v3.4 实测结论）

- **底模只用 int8 系**（`minimax_h3_hybrid_fl2va_ref2va_b25-49-int8.safetensors`）——GGUF / NVFP4 量化底模在本机实测音频退化（低频轰鸣），已弃用。
- **步数 ≥ 4**：Turbo 4 步 LoRA + beta/ladder 4 步为正式配置；**TaoMate 3 步在 int8 底模下音频双复现崩溃，已否决**。
- `sink_conditioning=exact_kv_and_rows` + `ffn_sparse=False` 保持默认。
- **运行环境**：单实例运行；GPU 长时间满载（连续多任务）后建议先冷却/重启实例再跑关键任务（本机实测 GPU 高负载状态下音频 latent 会稳定劣化，与稀疏/VAE 配置无关）。

---

## 🧩 五、示例工作流 / Example Workflow

**`example_workflows/BSAI VedaSparse · 蒸馏稀疏注意力 H3多合一示例工作流 v1.0 - 固定seed验证.json`** —— 唯一正式示例工作流（H3 多合一：文生视频 / 图生视频 / 参考生视频，v3.4 定稿 + 固定 seed 945967344952729，本机验收通过基线）。

### 结构 / Structure

- **统一模型链**：`UNETLoader (minimax_h3_hybrid_fl2va_ref2va_b25-49-int8)` → `LoraLoaderModelOnly (官方 Turbo 4步 LoRA)` → `BSAIVedaSparsePatch (keep 5%, dual-fast, sigma 0-0.98)` → `BasicGuider` / `BasicScheduler (beta, 4步)` + `KSamplerSelect (euler)`。
  > `BSAIVedaSparsePatch.enabled` 默认为 **false**（保留作音频/质量基线对比）；需要稀疏加速时在节点上打开 `enabled` 即可，参数已预置好。
- **文生视频 / Text-to-Video**：图片输入区 LoadImage 设为 bypass（`mode: 4`）即纯文生。
- **图生视频 / 参考生视频 / Image-to-Video & Reference-to-Video**：`easy ifElse` 开关（`PrimitiveBoolean`）切换：
  - `true`（默认）→ 参考生视频（`MiniMaxH3ReferenceToVideo`，参考图/参考视频 + 参考音频）；
  - `false` → 图生视频（`MiniMaxH3ImageToVideo`，首尾帧渐变）。
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
  - **v3.4**：`install_h3_patch()` 直接 monkey-patch H3 `Attention.forward`（104 实例），cond 行恒 dense、video 行 TripPool top-k；≤5 步调度全步稀疏；`ffn_sparse=False` 默认
  - `_DistilledScorer`：蒸馏预测器（预留）
- `nodes.py`：ComfyUI 节点（仅 `ModelPatcher.clone()` / `model_options` 注入，不改 ComfyUI 内部源码）
- **gather 显存优化**：按 (batch,head,tile) 行号高级索引整 tile 连续拷贝 + 动态 CHUNK 分块（峰值约 300MB），消除原全头 gather 的 6.7GB 峰值；int32 索引。RTX 5090 Laptop 实测默认档约 422ms/层、4 组 keep=10% 约 777ms/层、1 组 keep=5% 约 404ms/层（dense SDPA 668ms/层）。

### 正确性验证 / Correctness

单元测试（`test_veda_engine.py`，24 项全过）含最强验证：**keep=100% 时稀疏注意力精确还原 dense 注意力（最大误差 2.4e-7）**。

---

## 🛠️ 七、兼容性 / Compatibility

- ComfyUI ≥ 0.33.0（含 MiniMax-H3 支持）。
- v3.4 monkey-patch 与 FastH3 / TaoMate / T8 块缓存 / UniBlockSwap 等外部插件链式兼容（独立 patch attention 段，不冲突）。
- **官方 Veda-on-ComfyUI 全兼容（v3.4.1）**：不再互斥。官方节点通过 `optimized_attention_override` 注入稀疏，v3.4 检测到即让路（`override_priority=auto` 默认），Veda 拒绝的调用回落本引擎，最后原生兜底——三阶链式协作。
- **vendored 官方节点（v3.4.2）**：官方插件未装时由仓库自带副本自动注册 `VedaSparseAttention`（pull-and-go），官方已装时跳过——任何机器 pull 后工作流均不报缺节点包。
- 非 H3 扩散模型明确报错并 dense 回退，不影响其他工作流。

---

## ⚠️ 八、常见问题 / FAQ

**Q1：音频变成低频轰鸣 / Audio is low-frequency noise?**
① 底模必须用 int8 系（GGUF/NVFP4 已实测弃用）；② 步数 ≥ 4（Turbo 4 步 LoRA 配 4 步 beta 调度）；③ `ffn_sparse=False` + `sink=exact_kv_and_rows` 保持默认；④ 若在 GPU 长时满载/多实例并行后跑出轰鸣，先冷却/重启实例再跑（环境问题，非稀疏配置）。

**Q2：显存不足 / OOM?**
引擎已 CHUNK 分块（约 300MB 峰值）。仍 OOM 时降低分辨率/帧数或关闭其他占用显存的程序。

**Q3：看不到加速 / No speedup?**
确认 `enabled=true`、token 数 ≥ `max_sparse_tokens`（16384）、且为 H3 模型。短片段（<1s）稀疏收益小属正常。

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
