# BSAI-ComfyUI-VedaSparse

**Veda 蒸馏稀疏注意力 —— MiniMax-H3 视频生成加速（BSAI 第 19 个插件）**

> 由 [B 站视频《MiniMax H3 最新加速来了！十四秒视频提速二点七六倍！》](https://www.bilibili.com/video/BV1Uca46uExC/) 引出，基于 **Veda: Scalable Video Diffusion via Distilled Sparse Attention**（ByteDance + HKU，ICML 2026，arXiv:2605.30325）的公开算法实现。

---

## 一、这个技术是什么

视频主角 **VedaSparse**（转写里被读作 "Vidi/Vide/Vita"，画面标题实为 **VedaSparse**）是 **Veda 蒸馏稀疏注意力**在 MiniMax-H3 上的应用版，把 H3 生成视频的墙钟时间大幅压缩。核心技术三点：

| 组件 | 作用 | 相比 FastVideo VSA 的升级 |
|---|---|---|
| **TripPool 评分**（统计感知 tile 评分） | 用 Avg⊕Max⊕Min 三统计量描述每个 tile，评分贴近全注意力 | VSA 只用平均池化，稀释峰值信号；论文实测 tile 召回率 **66.4% vs 34.2%** |
| **Head-Aware Tiling**（按头时空分块） | 每个头分配 `(pt, ph, pw)` 时空分块（`pt·ph·pw = 64`），匹配不同头的时间/空间关注偏好 | VSA 所有头统一 64-token 立方 tile，结构失配大 |
| **Tile-Skipping**（tile 跳过） | 每个 query tile 只对 top-k 个 key tile（默认 10%）精确计算 | 执行层，把稀疏变成实际墙钟加速 |

视频实测（3×RTX4090，权重卸载主机内存，8 步去噪 + TabLoRA 配置）：

| 视频时长 | 端到端加速 | 注意力加速 |
|---|---|---|
| 5.17 秒 | 1.57× | — |
| 10.1 秒 | 2.21× | — |
| **14.4 秒** | **2.76×** | **6.66×** |
| 20 提示词平均 | 2.24× | 5.92× |
| 最强单条 | 3.08× | 6.87× |

VedaSparse 权重仅 **263MB（FP8，几乎无损）**，每层每头只保留 **10% 关键瓦片**，完全兼容现有去噪器、调度器和 VAE。

## 二、为什么做成独立插件（而不是升级 FastH3）

1. **BSAI 插件惯例是"一技术一插件"**：FastH3↔FastVideo VSA、VDN-H3↔VDN、Sol-H3↔Sol-Attn、TaoMate↔TaoMate。VedaSparse 是独立命名的新技术，应独立成插件。
2. **零回归**：不动现有 BSAI-ComfyUI-FastH3，已跑通的生产工作流不受影响。
3. **补生态空白**：视频明确说 VedaSparse"还没有 ComfyUI 加载器，需要用 Museion 框架跑，门槛稍高"——本插件就是 ComfyUI 原生加载器。

> 计数说明：现有 custom_nodes 下 BSAI 前缀插件正好 **18 个**，本插件为第 **19** 个（若按"下一个编号"的口径即用户所说的"第 18 个"）。

## 三、安装

把 `BSAI-ComfyUI-VedaSparse` 整个目录放入：

```
ComfyUI/custom_nodes/BSAI-ComfyUI-VedaSparse
```

重启 ComfyUI 即可。无需任何额外依赖（仅 torch，随 ComfyUI 环境自带）。

## 四、用法

### 最小接入

```
UNETLoader (H3) ──> BSAIVedaSparsePatch ──> BasicGuider ──> ... 其余照旧
```

所有参数有默认值，接上即可生效（默认 keep=10%、head-aware tiling、TripPool triplet 评分）。

### 与官方 8 步 LoRA 叠加（推荐，视频实测配置）

在 KSampler 之外先加载官方 Turbo LoRA `minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16`（8 步蒸馏），再接 BSAIVedaSparsePatch。这样同时获得 **8 步蒸馏 + 90% 稀疏注意力**，即视频里 2.76× 的完整配置。

### 推荐参数

| 场景 | keep_percent | head_tiling | tripool_mode |
|---|---|---|---|
| 视频实测配置（8 步 + TabLoRA） | **10** | head-aware (4 groups) | triplet |
| 质量优先 | 20 | balanced (4,4,4) | triplet |
| 速度优先（长视频） | 8 | temporal-first (8,4,2) | maxmin |
| 已有 VSA 习惯 | 10 | balanced (4,4,4) | avg（等价 VSA 评分，仅几何升级） |

> keep_percent 越低越快，但画面质量下降越明显。10% 是视频实测的平衡点；8 步 V2 建议 10~20。

### 节点

#### BSAIVedaSparsePatch

| 参数 | 默认 | 说明 |
|---|---|---|
| model | — | H3 扩散模型（UNETLoader 输出） |
| enabled | True | 关闭=直通，不影响任何现有工作流 |
| keep_percent | 10 | 保留关键 key tile 百分比（10 = 90% 稀疏） |
| head_tiling | head-aware (4 groups) | 按头循环分配 (4,4,4)/(8,4,2)/(2,4,8)/(4,8,2) |
| custom_tiling | 4,4,4;8,4,2;2,4,8;4,8,2 | head_tiling=custom 时的分块表（每个 pt*ph*pw=64） |
| tripool_mode | triplet | triplet=Avg⊕Max⊕Min（论文最优）/ maxmin / avg |
| scorer_weights | None (heuristic) | 蒸馏预测器权重（见第五节） |
| aspect | auto | 画幅（用于解析 (T,H,W) 几何；auto 优先用 H3 布局自带尺寸） |
| force_dims | "" | 手动指定 video latent 尺寸 `T,H,W`（一般留空） |
| start_percent | 0.2 | 去噪前 20% 保持 dense 预热 |
| end_percent | 1.0 | 去噪后段保持 dense 的起点 |
| min_tokens | 4096 | 序列 token 低于此值走 dense（避免短片 overhead） |
| sink_conditioning | exact_kv_and_rows | 文本/音频/参考 conditioning 行始终精确 |
| verbose | False | 详细日志 |

#### BSAIVedaSparseStats

只读诊断节点，输出 sparse/dense/heuristic/distilled/errors/fallback_1d 计数，用于确认稀疏是否生效（sparse>0 且 errors=0 即正常）。

## 五、评分器两种模式

| 模式 | 何时用 | 说明 |
|---|---|---|
| **heuristic（默认）** | 开箱即用 | φ=identity 的 TripPool 描述符直接点积评分（论文 eq.6 单位投影形式），无需任何权重 |
| **distilled（可选）** | 官方权重发布后 | 加载 VedaSparse 蒸馏预测器（263MB FP8，每头独立 MLP 投影 φ_q/φ_k） |

蒸馏权重放 `ComfyUI/models/veda_scorers/`，节点 `scorer_weights` 下拉选择。权重 key 规范（veda_engine.py 中 `VEDA_SCORER_KEY_SPEC`）：

```
vedascorer.layer_{l}.head_{h}.q_proj   [in=3*d_head, out=d_latent]
vedascorer.layer_{l}.head_{h}.k_proj   [in=3*d_head, out=d_latent]
```

> 官方权重公开前，heuristic 模式即 Veda 论文算法中无需权重的部分（TripPool 统计感知评分 + head-aware tiling + tile-skipping），已实现视频实测的加速结构。权重发布后无需改代码即可切换到蒸馏评分。

## 六、实现说明（与论文的对应）

- `veda_engine.py`：核心引擎
  - `_tripool()`：论文 eq.5 的 TripPool 描述符（Avg⊕Max⊕Min）
  - `_heuristic_scores()`：论文 eq.6 单位投影形式的评分
  - `_tile_3d()`：Head-Aware Tiling（pt·ph·pw=64），返回 tile 与 key mask
  - `_veda_sparse()`：tile-skipping 稀疏注意力（含 conditioning dense 行）
  - `_DistilledScorer`：蒸馏预测器（预留，加载 φ_q/φ_k 投影）
  - `install_h3_span()`：发布 H3 packed 布局的 video/audio span 与 latent 尺寸（幂等，不改 ComfyUI 内部源码）
- `nodes.py`：ComfyUI 节点（仅通过 ModelPatcher.clone()/model_options 注入）
- 引擎通过 `transformer_options["optimized_attention_override"]` 接入，与 ComfyUI 0.37 的 H3 注意力回调兼容；若已存在其他 attention patch（如 FastH3 VSA），会自动链式组合。

### 正确性验证

单元测试（`test_veda_engine.py`，24 项全过）包含最强验证：**keep=100% 时稀疏注意力精确还原 dense 注意力（最大误差 2.4e-7）**，证明 tile 划分、评分、gather、mask、写回全链路数值正确。

## 七、为什么是 BSAI 前缀

按用户铁律，本插件的所有权重/脚本/命名统一使用 BSAI 前缀；模型权重文件（若有）统一以 BSAI 命名。只写入用户指定的盘符（本项目为 C:\BSAI\ComfyUI-BSAI_pro_v41），未经验收不同步其他盘。

## 八、已知边界

- VedaSparse 官方蒸馏权重（263MB FP8）与 Museion 框架代码**目前未公开**（GitHub 检索 0 结果），本插件实现的是论文公开算法部分；蒸馏权重发布后可无缝启用（第五节）。
- 引擎为 torch 纯实现（逐头 tile 处理），加速来自注意力计算量的 90% 削减；与 CUDA 原生 tile-skipping kernel 相比仍有优化空间，后续可加 Triton kernel。
- 本插件未包含独立 benchmark 脚本；实测数据引用视频作者测试（来源见文首链接）。

## 九、参考

- 视频：AI绘画KK《MiniMax H3 最新加速来了！十四秒视频提速二点七六倍！》 https://www.bilibili.com/video/BV1Uca46uExC/
- 论文：Veda: Scalable Video Diffusion via Distilled Sparse Attention（ICML 2026，arXiv:2605.30325）
- 姊妹插件：BSAI-ComfyUI-FastH3（FastVideo VSA）、BSAI-ComfyUI-vdn-minimax-h3（VDN 混合注意力）、BSAI-ComfyUI-MiniMax-H3-PDD-Acc（PDD 8 步）、BSAI-ComfyUI-TaoMate（TaoMate 3 步）
