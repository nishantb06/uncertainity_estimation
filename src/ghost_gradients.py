"""Ghost-trick helpers for per-sample Linear weight gradient geometry.

For y = Wx, per-sample grad is δ ⊗ a. Inner products factorize as
⟨δ1, δ2⟩ · ⟨a1, a2⟩ without materializing the outer product.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class GhostGradientCapture:
    """Capture per-sample activations and output grads on selected Linear layers."""

    def __init__(self, model: nn.Module, *, module_prefix: str = "head"):
        self.model = model
        self.module_prefix = module_prefix
        self.activations: dict[str, torch.Tensor] = {}
        self.errors: dict[str, torch.Tensor] = {}
        self.hooks: list[Any] = []
        self._register()

    def _register(self) -> None:
        for name, module in self.model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            if self.module_prefix and not (
                name == self.module_prefix or name.startswith(self.module_prefix + ".")
            ):
                continue
            self.hooks.append(module.register_forward_hook(self._make_fwd_hook(name)))
            self.hooks.append(
                module.register_full_backward_hook(self._make_bwd_hook(name))
            )

    def _make_fwd_hook(self, name: str):
        def hook(module: nn.Module, inp: tuple, out: torch.Tensor) -> None:
            # Keep graph-free copy for algebra; shape (B, d_in)
            x = inp[0]
            if x.dim() != 2:
                x = x.reshape(x.shape[0], -1)
            self.activations[name] = x.detach()

        return hook

    def _make_bwd_hook(self, name: str):
        def hook(
            module: nn.Module,
            grad_input: tuple,
            grad_output: tuple,
        ) -> None:
            g = grad_output[0]
            if g is None:
                return
            if g.dim() != 2:
                g = g.reshape(g.shape[0], -1)
            self.errors[name] = g.detach()

        return hook

    def clear(self) -> None:
        self.activations.clear()
        self.errors.clear()

    def remove(self) -> None:
        for h in self.hooks:
            h.remove()
        self.hooks.clear()

    def factors(
        self,
        preconditioners: dict[str, tuple[torch.Tensor | None, torch.Tensor | None]]
        | None = None,
    ) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        """Return {layer: (a, delta)} optionally scaled by row/col preconditioners."""
        out: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        for name, a in self.activations.items():
            if name not in self.errors:
                continue
            d = self.errors[name]
            if preconditioners and name in preconditioners:
                p_out, p_in = preconditioners[name]
                if p_out is not None:
                    d = d * p_out.to(device=d.device, dtype=d.dtype).unsqueeze(0)
                if p_in is not None:
                    a = a * p_in.to(device=a.device, dtype=a.dtype).unsqueeze(0)
            out[name] = (a, d)
        return out


def ghost_inner_product_batched(
    a1: torch.Tensor,
    d1: torch.Tensor,
    a2: torch.Tensor,
    d2: torch.Tensor,
) -> torch.Tensor:
    """⟨δ1⊗a1, δ2⊗a2⟩ per sample. Inputs (B, *). Returns (B,)."""
    return (d1 * d2).sum(dim=-1) * (a1 * a2).sum(dim=-1)


def multi_layer_inner_product(
    factors1: dict[str, tuple[torch.Tensor, torch.Tensor]],
    factors2: dict[str, tuple[torch.Tensor, torch.Tensor]],
) -> torch.Tensor:
    """Sum ghost IPs over shared layers. Returns (B,)."""
    names = sorted(set(factors1) & set(factors2))
    if not names:
        raise RuntimeError("No overlapping ghost layers between factor dicts")
    total: torch.Tensor | None = None
    for name in names:
        a1, d1 = factors1[name]
        a2, d2 = factors2[name]
        ip = ghost_inner_product_batched(a1, d1, a2, d2)
        total = ip if total is None else total + ip
    assert total is not None
    return total


def multi_layer_norm_sq(
    factors: dict[str, tuple[torch.Tensor, torch.Tensor]],
) -> torch.Tensor:
    """||g||^2 = sum_L ||δ||^2 ||a||^2 per sample. Returns (B,)."""
    total: torch.Tensor | None = None
    for a, d in factors.values():
        n = (d * d).sum(dim=-1) * (a * a).sum(dim=-1)
        total = n if total is None else total + n
    assert total is not None
    return total


def multi_layer_cosine(
    factors1: dict[str, tuple[torch.Tensor, torch.Tensor]],
    factors2: dict[str, tuple[torch.Tensor, torch.Tensor]],
    eps: float = 1e-12,
) -> torch.Tensor:
    """Cosine similarity of ghost gradients across layers. Returns (B,)."""
    ip = multi_layer_inner_product(factors1, factors2)
    n1 = multi_layer_norm_sq(factors1).clamp_min(eps).sqrt()
    n2 = multi_layer_norm_sq(factors2).clamp_min(eps).sqrt()
    return ip / (n1 * n2)


def load_head_preconditioners(
    optimizer_states: list[dict[str, Any]] | None,
    model: nn.Module,
    *,
    module_prefix: str = "head",
    eps: float = 1e-8,
    beta2: float = 0.999,
) -> dict[str, tuple[torch.Tensor | None, torch.Tensor | None]]:
    """Build row/col marginal AdamW preconditioners for head Linear weights.

    Lightning stores ``optimizer_states`` as a list of dicts with ``state`` keyed
    by parameter index matching ``param_groups``. If unavailable, returns {}.
    """
    if not optimizer_states:
        return {}

    # Map parameter object id -> (exp_avg_sq, step)
    param_to_v: dict[int, tuple[torch.Tensor, int]] = {}
    for opt_state in optimizer_states:
        state = opt_state.get("state") or {}
        param_groups = opt_state.get("param_groups") or []
        # Lightning often keys state by integer; rebuild via named params order
        # Prefer matching by walking model params against state entries when possible.
        _ = param_groups
        for key, buf in state.items():
            if not isinstance(buf, dict) or "exp_avg_sq" not in buf:
                continue
            # key may be int or param — store under hashable form later via shape match
            step = int(buf.get("step", 1))
            if torch.is_tensor(step):
                step = int(step.item())
            param_to_v[id(buf["exp_avg_sq"])] = (buf["exp_avg_sq"], step)

    # Fallback: zip named head Linear weights with Adam state values in insertion order
    v_list = [
        (buf["exp_avg_sq"], int(buf.get("step", 1) if not torch.is_tensor(buf.get("step", 1)) else buf["step"].item()))
        for opt_state in optimizer_states
        for buf in (opt_state.get("state") or {}).values()
        if isinstance(buf, dict) and "exp_avg_sq" in buf
    ]

    head_linears: list[tuple[str, nn.Linear]] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if not (name == module_prefix or name.startswith(module_prefix + ".")):
            continue
        head_linears.append((name, module))

    preconds: dict[str, tuple[torch.Tensor | None, torch.Tensor | None]] = {}
    if len(v_list) < len(head_linears):
        # Not enough state — identity
        return {}

    # Match by weight shape against unused v buffers
    unused = list(v_list)
    for name, linear in head_linears:
        w = linear.weight
        match_idx = next(
            (i for i, (v, _) in enumerate(unused) if tuple(v.shape) == tuple(w.shape)),
            None,
        )
        if match_idx is None:
            preconds[name] = (None, None)
            continue
        v, step = unused.pop(match_idx)
        bias_correction2 = 1.0 - (beta2 ** max(step, 1))
        v_hat = v / bias_correction2
        p = 1.0 / (v_hat.sqrt() + eps)
        # Row / col marginals of the diagonal preconditioner
        p_out = p.mean(dim=1).detach().cpu()  # (d_out,)
        p_in = p.mean(dim=0).detach().cpu()  # (d_in,)
        preconds[name] = (p_out, p_in)
        _ = param_to_v  # kept for clarity / future exact matching

    return preconds
