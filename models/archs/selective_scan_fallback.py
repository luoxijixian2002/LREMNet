"""Pure-PyTorch fallback of ``mamba_ssm.ops.selective_scan_interface``.

The official ``mamba_ssm`` package requires a CUDA toolchain at install
time.  To keep LREMNet runnable on machines without it (CPU-only smoke
tests, evaluation, etc.), this module re-implements :func:`selective_scan_fn`
with the same calling convention used by ``models/ss2d.py`` and
``models/edgemamba.py``:

    out = selective_scan_fn(u, delta, A, B, C, D, z=None, delta_bias=None,
                            delta_softplus=False, return_last_state=False)

Shapes follow the official interface:

* ``u``:      ``(B, K*D, L)``     -- input sequence
* ``delta``:  ``(B, K*D, L)``     -- discretization step
* ``A``:      ``(K*D, N)``        -- state transition matrix (diagonal)
* ``B``, ``C``: ``(B, K, N, L)``  -- input / output projections (or ``(B, N, L)``)
* ``D``:      ``(K*D,)``          -- skip connection
* ``z``:      ``(B, D, L)`` or None
* ``delta_bias``: ``(K*D,)`` or None

The discrete recurrence implemented here is the standard S6 scan:

    delta  = softplus(delta + delta_bias)          (if delta_softplus)
    dA     = exp(delta * A)
    dB     = delta * B
    h_t    = dA_t * h_{t-1} + dB_t * x_t
    y_t    = sum_n(C_t,n * h_t,n) + D * x_t

NOTE
----
This implementation is exact but *much* slower than the fused CUDA
kernel.  It is intended as a correctness reference / CPU fallback.
For full-speed training, please install ``mamba_ssm``.
"""
import torch
import torch.nn.functional as F


@torch.jit.script
def _ssm_scan_core(u: torch.Tensor, delta: torch.Tensor, A: torch.Tensor,
                   B: torch.Tensor, C: torch.Tensor, D: torch.Tensor):
    # u, delta: (B, D, L) fp32
    # A: (B, D, N), B, C: (B, N, L), D: (B, D)
    Bb, Dd, L = u.shape
    N = A.shape[2]
    h = torch.zeros(Bb, Dd, N, dtype=torch.float32, device=u.device)
    ys = torch.empty(Bb, Dd, L, dtype=torch.float32, device=u.device)
    for i in range(L):
        dA = torch.exp(delta[:, :, i].unsqueeze(-1) * A)                  # (B, D, N)
        dB = delta[:, :, i].unsqueeze(-1) * B[:, :, i].unsqueeze(1)       # (B, D, N)
        h = dA * h + dB * u[:, :, i].unsqueeze(-1)
        ys[:, :, i] = torch.sum(h * C[:, :, i].unsqueeze(1), dim=-1) \
            + D * u[:, :, i]
    return ys


def selective_scan_fn(u, delta, A, B, C, D=None, z=None, delta_bias=None,
                      delta_softplus=False, return_last_state=False):
    u = u.float().contiguous()
    delta = delta.float().contiguous()
    A = A.float()
    B = B.float()
    C = C.float()
    D = D.float()

    if delta_bias is not None:
        delta = delta + delta_bias.view(1, -1, 1).to(delta.dtype)
    if delta_softplus:
        delta = F.softplus(delta)

    Bb = B.shape[0]
    if B.dim() == 4:  # grouped scan: B, C are (B, K, N, L)
        K = B.shape[1]
        N = B.shape[2]
        L = B.shape[3]
    else:             # plain scan: B, C are (B, N, L)
        K = 1
        N = B.shape[1]
        L = B.shape[2]
        B = B.view(Bb, 1, N, L)
        C = C.view(Bb, 1, N, L)

    D_total = u.shape[1]
    assert D_total % K == 0, f"D_total({D_total}) must be divisible by K({K})"
    Dg = D_total // K

    u = u.view(Bb, K, Dg, L).reshape(Bb * K, Dg, L)
    delta = delta.view(Bb, K, Dg, L).reshape(Bb * K, Dg, L)
    A = A.view(K, Dg, N).repeat(Bb, 1, 1)  # (Bb*K, Dg, N)
    B = B.reshape(Bb * K, N, L)
    C = C.reshape(Bb * K, N, L)
    D = D.view(K, Dg).repeat(Bb, 1)  # (Bb*K, Dg)

    out = _ssm_scan_core(u, delta, A, B, C, D)  # (B*K, Dg, L)
    out = out.view(Bb, K * Dg, L)

    if z is not None:
        out = out * F.silu(z.float())

    if return_last_state:
        raise NotImplementedError(
            "return_last_state=True is not supported by the CPU fallback. "
            "Install mamba_ssm for the fused CUDA implementation.")
    return out


def selective_scan_ref(u, delta, A, B, C, D=None, z=None, delta_bias=None,
                       delta_softplus=False, return_last_state=False):
    """Alias kept for API compatibility with mamba_ssm."""
    return selective_scan_fn(u, delta, A, B, C, D, z, delta_bias,
                             delta_softplus, return_last_state)


def test_fallback():
    """Quick self-test comparing against a hand-rolled scan."""
    torch.manual_seed(0)
    Bb, Dd, N, L = 2, 8, 4, 32
    u = torch.randn(Bb, Dd, L)
    delta = torch.rand(Bb, Dd, L) * 0.05
    A = -torch.rand(Dd, N)
    B = torch.randn(Bb, N, L)
    C = torch.randn(Bb, N, L)
    D = torch.ones(Dd)
    bias = torch.zeros(Dd)

    out = selective_scan_fn(u, delta, A, B, C, D, None, bias, True, False)

    # manual reference
    delta_s = F.softplus(delta + bias.view(1, -1, 1))
    h = torch.zeros(Bb, Dd, N)
    ref = torch.empty(Bb, Dd, L)
    for i in range(L):
        dA = torch.exp(delta_s[:, :, i].unsqueeze(-1) * A.unsqueeze(0))
        dB = delta_s[:, :, i].unsqueeze(-1) * B[:, :, i].unsqueeze(1)
        h = dA * h + dB * u[:, :, i].unsqueeze(-1)
        ref[:, :, i] = torch.sum(h * C[:, :, i].unsqueeze(1), dim=-1) + u[:, :, i] * D
    err = (out - ref).abs().max().item()
    print(f"fallback self-test max err = {err:.3e}")
    assert err < 1e-4


if __name__ == "__main__":
    test_fallback()
    print("selective_scan fallback OK")
