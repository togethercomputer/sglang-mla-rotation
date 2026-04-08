"""
Fake (simulated) KV quantization: integer per-head (see ``fake_int_quant_kv_per_head``)
and NVFP4 / MXFP4 block quant matching ``KVFP4QuantizeUtil`` (see
``fake_nvfp4_quant_kv_per_head``).

Layout for both: ``(..., num_heads, head_dim)``.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

_kvfp4_tensor_mod = None


def _get_kvfp4_tensor_module():
    """Load ``kvfp4_tensor.py`` directly (avoids ``quantization`` package side effects)."""
    global _kvfp4_tensor_mod
    if _kvfp4_tensor_mod is None:
        path = Path(__file__).resolve().parent.parent / "layers" / "quantization" / "kvfp4_tensor.py"
        spec = importlib.util.spec_from_file_location("sglang_srt_kvfp4_tensor", path)
        _kvfp4_tensor_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_kvfp4_tensor_mod)
    return _kvfp4_tensor_mod


def fake_int_quant_kv_per_head(
    x: torch.Tensor,
    num_bits: int,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Per-head-vector asymmetric fake integer quant. Returns a float tensor like ``x``.

    Integer grid is signed, ``[-2**(num_bits-1), 2**(num_bits-1)-1]`` (e.g. int8).

    Parameters
    ----------
    x
        ``(..., num_heads, head_dim)``.
    num_bits
        Bit width of the signed grid (e.g. 4, 8, 16).
    eps
        Avoids div-by-zero when a head vector is constant.
    """
    if x.ndim < 2:
        raise ValueError(f"x must be at least 2D (..., num_heads, head_dim), got shape {tuple(x.shape)}")
    if not (1 <= num_bits <= 32):
        raise ValueError(f"num_bits must be in [1, 32], got {num_bits}")

    out_dtype = x.dtype
    if out_dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        raise TypeError(f"expected float tensor, got {out_dtype}")

    qmax = (1 << (num_bits - 1)) - 1
    qmin = -(1 << (num_bits - 1))

    xf = x.float()
    *lead, H, D = xf.shape
    flat = xf.reshape(-1, H, D)

    xmin = flat.amin(dim=-1)
    xmax = flat.amax(dim=-1)
    span = (xmax - xmin).clamp_min(eps)
    scale = span / float(qmax - qmin)
    zp = torch.round(qmin - xmin / scale.clamp_min(eps)).clamp(qmin, qmax)

    scale_e = scale.unsqueeze(-1)
    zp_e = zp.unsqueeze(-1)
    q = torch.round(flat / scale_e + zp_e).clamp(qmin, qmax)
    x_hat = (q - zp_e) * scale_e

    x_hat = x_hat.reshape(*lead, H, D)
    return x_hat.to(out_dtype)


def _nvfp4_fake_quant_bmh(tensor: torch.Tensor) -> torch.Tensor:
    """
    One MXFP4 round-trip in float, same numerics as
    ``KVFP4QuantizeUtil.batched_quantize`` + ``batched_dequantize`` but without
    pack/unpack. ``tensor`` is ``[B, M, N]`` float32 with ``M * N`` divisible by 16.
    """
    b, m, n = tensor.shape
    device = tensor.device
    kv = _get_kvfp4_tensor_module()
    bounds = kv.E2M1_BOUNDS.to(device=device, dtype=torch.float32)
    values = kv.E2M1_VALUES.to(device=device, dtype=torch.float32)

    reshaped = tensor.view(b, m * n // 16, 16)
    block_max = reshaped.abs().max(dim=-1, keepdim=True).values
    scale_exp = torch.ceil(torch.log2(torch.clamp(block_max / kv.E2M1_MAX, min=1e-10)))
    scaled = reshaped / torch.exp2(scale_exp)

    sign_bits = (scaled < 0).to(torch.uint8) << 3
    abs_vals = scaled.abs()
    magnitude_bits = torch.sum(abs_vals.unsqueeze(-1) >= bounds, dim=-1)
    fp4_vals = sign_bits + magnitude_bits.to(torch.uint8)

    sign_mask = (fp4_vals & 0x08) != 0
    magnitude_idx = fp4_vals & 0x07
    float_vals = values[magnitude_idx.long()]
    float_vals = torch.where(sign_mask, -float_vals, float_vals)
    out = float_vals * torch.exp2(scale_exp)
    return out.view(b, m, n)


def fake_nvfp4_quant_kv_per_head(x: torch.Tensor) -> torch.Tensor:
    """
    NVFP4 (E2M1 + per-16-element block scale) fake quant in one step.

    Uses the same rules as ``KVFP4QuantizeUtil`` in ``kvfp4_tensor.py``: blocks
    of 16 consecutive elements in memory order over the last two dimensions
    ``(num_heads, head_dim)`` (``head_dim`` varies fastest). This is **not** the
    same grouping as ``fake_int_quant_kv_per_head`` (which uses one scale per
    full head vector).

    Parameters
    ----------
    x
        ``(..., num_heads, head_dim)``. If ``num_heads * head_dim`` is not a
        multiple of 16, the flattened tail (``head_dim`` fastest) is zero-padded
        to the next multiple of 16 for NVFP4 blocks, then the padding is dropped.

    Returns
    -------
    Float tensor with the same shape and dtype as ``x``.
    """
    if x.ndim < 2:
        raise ValueError(f"x must be at least 2D (..., num_heads, head_dim), got shape {tuple(x.shape)}")
    out_dtype = x.dtype
    if out_dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        raise TypeError(f"expected float tensor, got {out_dtype}")
    *lead, h, d = x.shape
    xf = x.reshape(-1, h, d).float()
    L = h * d

    if L % 16 == 0:
        x_hat = _nvfp4_fake_quant_bmh(xf)
    else:
        pad = 16 - (L % 16)
        flat = xf.reshape(-1, L)
        flat = torch.cat([flat, flat.new_zeros(flat.shape[0], pad)], dim=-1)
        x_hat = _nvfp4_fake_quant_bmh(flat.unsqueeze(1)).squeeze(1)[:, :L].reshape(-1, h, d)

    return x_hat.reshape(*lead, h, d).to(out_dtype)


def _self_test():
    torch.manual_seed(0)
    B, S, H, D = 2, 4, 8, 64
    k = torch.randn(B, S, H, D)
    v = torch.randn(B, S, H, D)
    for bits in (4, 8, 16):
        kh = fake_int_quant_kv_per_head(k, bits)
        vh = fake_int_quant_kv_per_head(v, bits)
        assert kh.shape == k.shape and vh.dtype == k.dtype
        assert (kh - k).float().abs().mean() < 0.2 * (17 / (2**bits))

    mla = torch.randn(16, 1, 576)
    mh = fake_int_quant_kv_per_head(mla, 8)
    assert mh.shape == mla.shape

    # NVFP4: shape / dtype; idempotent under float32 (grid + scale are fixed points)
    k4 = torch.randn(2, 4, 8, 64)
    y = fake_nvfp4_quant_kv_per_head(k4)
    assert y.shape == k4.shape and y.dtype == k4.dtype
    y2 = fake_nvfp4_quant_kv_per_head(y)
    assert torch.allclose(y.float(), y2.float(), atol=0, rtol=0)

    # NVFP4 with h*d not divisible by 16 (zero-pad linear tail, then crop)
    k_odd = torch.randn(3, 2, 5)
    y_odd = fake_nvfp4_quant_kv_per_head(k_odd)
    assert y_odd.shape == k_odd.shape and y_odd.dtype == k_odd.dtype
    y_odd2 = fake_nvfp4_quant_kv_per_head(y_odd)
    assert torch.allclose(y_odd.float(), y_odd2.float(), atol=0, rtol=0)


if __name__ == "__main__":
    _self_test()
