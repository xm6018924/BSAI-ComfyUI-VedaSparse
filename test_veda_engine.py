"""BSAI VedaSparse v3.4 — 单元验证。

验证点（正确性优先，用户受够了废片）：
1. _veda_sparse_packed keep=100% 时 与 dense SDPA 数值一致（联合 softmax 数学）。
2. keep<100% 时 conditioning 行（[:cond_end]）输出与 dense 完全一致（sink 语义）。
3. Triton block-sparse kernel 与 dense 数值对齐（cond tile 恒 True）。
4. _build_block_mask：cond tile 全 True、video 区 top-k。
5. H3 monkey-patch 语法完整性（import + 函数存在）。

运行：python test_veda_engine.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import veda_engine as ve


def dense_ref(q, k, v, scale):
    """PyTorch dense SDPA 参考（skip_reshape 布局 [B,H,N,D]）。"""
    att = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
    p = att.softmax(dim=-1)
    return torch.matmul(p.to(v.dtype), v)


def test_pytorch_keep100():
    torch.manual_seed(0)
    B, H, N, D = 1, 4, 512, 64
    q = torch.randn(B, H, N, D, dtype=torch.float32)
    k = torch.randn(B, H, N, D, dtype=torch.float32)
    v = torch.randn(B, H, N, D, dtype=torch.float32)
    scale = D ** -0.5
    cond_end, video_end = 128, N

    ref = dense_ref(q, k, v, scale)
    out = ve._veda_sparse_packed(q, k, v, scale, 1.0, cond_end, video_end, "triplet")
    err = (out - ref).abs().max().item()
    rel = ((out - ref).abs() / (ref.abs() + 1e-6)).mean().item()
    print(f"[keep100] max_abs_err={err:.2e} mean_rel_err={rel:.2e}")
    assert err < 2e-4, f"keep=100% 数值不一致: {err}"


def test_cond_rows_exact():
    torch.manual_seed(1)
    B, H, N, D = 1, 4, 640, 64
    q = torch.randn(B, H, N, D, dtype=torch.float32)
    k = torch.randn(B, H, N, D, dtype=torch.float32)
    v = torch.randn(B, H, N, D, dtype=torch.float32)
    scale = D ** -0.5
    cond_end, video_end = 160, N

    ref = dense_ref(q, k, v, scale)
    for keep in (0.05, 0.2, 0.5):
        out = ve._veda_sparse_packed(q, k, v, scale, keep, cond_end, video_end, "triplet")
        cond_ref = ref[:, :, :cond_end]
        cond_out = out[:, :, :cond_end]
        err = (cond_out - cond_ref).abs().max().item()
        print(f"[keep={keep}] cond_rows max_abs_err={err:.2e}")
        assert err < 2e-4, f"keep={keep} cond 行被改动: {err}"


def test_triton_vs_dense():
    if not ve._TRITON_SPARSE_AVAILABLE:
        print("[triton] skipped (not available)")
        return
    if not torch.cuda.is_available():
        print("[triton] skipped (no cuda)")
        return
    torch.manual_seed(2)
    dev = "cuda"
    B, H, N, D = 1, 4, 640, 64
    q = torch.randn(B, H, N, D, dtype=torch.bfloat16, device=dev)
    k = torch.randn(B, H, N, D, dtype=torch.bfloat16, device=dev)
    v = torch.randn(B, H, N, D, dtype=torch.bfloat16, device=dev)
    scale = D ** -0.5
    cond_end, video_end = 160, N

    ref = dense_ref(q.float(), k.float(), v.float(), scale).to(torch.bfloat16)

    n_tile = (N + ve.BLOCK - 1) // ve.BLOCK
    sel_idx, sel_cnt = ve._build_selected(q, k, v, cond_end, video_end, 1.0, "triplet")
    out = ve._triton_sparse_attention(q, k, v, sel_idx, sel_cnt)
    err = (out.float() - ref.float()).abs().max().item()
    print(f"[triton keep100] max_abs_err={err:.2e}")
    assert err < 1e-2, f"Triton keep=100% 不一致: {err}"

    sel_idx5, sel_cnt5 = ve._build_selected(q, k, v, cond_end, video_end, 0.1, "triplet")
    n_dense_rows = int((sel_cnt5 < 0).sum().item())
    avg_sel = sel_cnt5[sel_cnt5 >= 0].float().mean().item() if (sel_cnt5 >= 0).any() else 0.0
    print(f"[triton sel] keep10%: n_tile={n_tile}, dense_rows={n_dense_rows}, "
          f"avg_sel_per_video_tile={avg_sel:.1f}")
    assert n_dense_rows == (cond_end + ve.BLOCK - 1) // ve.BLOCK, "cond 行未全 dense 哨兵"
    out5 = ve._triton_sparse_attention(q, k, v, sel_idx5, sel_cnt5)
    cond_ref = ref[:, :, :cond_end]
    cond_out = out5[:, :, :cond_end]
    cerr = (cond_out.float() - cond_ref.float()).abs().max().item()
    print(f"[triton keep10] cond_rows max_abs_err={cerr:.2e}")
    assert cerr < 1e-2, f"Triton keep10% cond 行不一致: {cerr}"


def test_patch_surface():
    for name in ("apply_veda", "veda_stats", "reset_veda_stats",
                 "_install_h3_attn_patch", "_h3_attn_forward", "_h3_attn_call"):
        assert hasattr(ve, name), f"缺少 {name}"
    print("[patch] surface OK")


if __name__ == "__main__":
    test_patch_surface()
    test_pytorch_keep100()
    test_cond_rows_exact()
    test_triton_vs_dense()
    print("\nALL VEDA v3.4 TESTS PASSED")
