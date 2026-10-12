# -*- coding: utf-8 -*-
"""基于 VedaSparse v1.0 示例工作流生成 R2VA 官方节点版示例工作流。"""
import json, uuid, sys, io

src = r"G:\BSAI-ComfyUI-intel-XPU-GPU-NPU-aki\ComfyUI\custom_nodes\BSAI-ComfyUI-VedaSparse\example_workflows\BSAI VedaSparse · 蒸馏稀疏注意力 H3多合一示例工作流 v1.0 - 固定seed验证.json"
out = r"G:\BSAI-ComfyUI-intel-XPU-GPU-NPU-aki\ComfyUI\custom_nodes\BSAI-ComfyUI-VedaSparse\example_workflows\BSAI VedaSparse · 官方 R2VA 预测器 H3 参考生视频示例工作流 v1.1.json"

with io.open(src, "r", encoding="utf-8") as f:
    d = json.load(f)

# 1) 移除自有引擎节点 BSAIVedaSparsePatch (id=3)
d["nodes"] = [n for n in d["nodes"] if n["id"] != 3]

# 2) 新增 vendored 官方节点 VedaSparseAttention（predictor 预设 R2VA）
new_node = {
    "id": 119,
    "type": "VedaSparseAttention",
    "pos": [605.7, 8.0],
    "size": [430, 214],
    "flags": {},
    "order": 17,
    "mode": 0,
    "inputs": [{"name": "model", "type": "MODEL", "link": 139}],
    "outputs": [{"name": "MODEL", "type": "MODEL", "slot_index": 0, "links": [102, 103]}],
    "title": "Veda Sparse Attention · R2VA 预测器（官方节点）",
    "properties": {
        "aux_id": "xm6018924/BSAI-ComfyUI-VedaSparse",
        "ver": "v3.4.2-r2va",
        "Node name for S&R": "VedaSparseAttention"
    },
    "widgets_values": [
        "minimax_h3_r2va_veda_preview_fp8.safetensors",
        "90%",
        "90%",
        "",
        "",
        False
    ],
    "widgets_values_named": {
        "predictor": "minimax_h3_r2va_veda_preview_fp8.safetensors",
        "generated_sparsity": "90%",
        "reference_sparsity": "90%",
        "full_attention_layers": "",
        "full_attention_steps": "",
        "verbose": False
    },
    "bgcolor": "rgba(0,0,0,0)"
}
d["nodes"].append(new_node)

# 3) 重连 MODEL 链路：LoRA(118) -> VedaSparseAttention(119) -> scheduler/guider
for l in d["links"]:
    if l[1] == 3:
        l[1] = 119
    if l[3] == 3:
        l[3] = 119

# 4) 输出文件名前缀
for n in d["nodes"]:
    if n["type"] == "SaveVideo":
        n["widgets_values"][0] = "BSAI_VedaSparse_R2VA"
        n["widgets_values_named"]["filename_prefix"] = "BSAI_VedaSparse_R2VA"

d["last_node_id"] = 119
d["id"] = str(uuid.uuid4())

with io.open(out, "w", encoding="utf-8") as f:
    json.dump(d, f, ensure_ascii=False, indent=2)

# 校验：所有 link 的 from/to 节点存在，所有节点引用 link 存在
node_ids = {n["id"] for n in d["nodes"]}
link_ids = {l[0] for l in d["links"]}
errs = []
for l in d["links"]:
    if l[1] not in node_ids: errs.append("link %s from-node %s missing" % (l[0], l[1]))
    if l[3] not in node_ids: errs.append("link %s to-node %s missing" % (l[0], l[3]))
for n in d["nodes"]:
    for i in (n.get("inputs") or []):
        if i.get("link") is not None and i["link"] not in link_ids:
            errs.append("node %s input %s link %s missing" % (n["id"], i["name"], i["link"]))
    for o in (n.get("outputs") or []):
        for lk in (o.get("links") or []):
            if lk not in link_ids:
                errs.append("node %s output %s link %s missing" % (n["id"], o["name"], lk))
print("nodes:", len(d["nodes"]), "links:", len(d["links"]))
print("v119 present:", 119 in node_ids, "| old#3 removed:", 3 not in node_ids)
print("errors:", errs if errs else "NONE")
print("written:", out)
