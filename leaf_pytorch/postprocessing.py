import torch
import torch.nn as nn


def _logit(p, eps: float = 1e-6):
    """Numerically stable logit."""
    p = torch.as_tensor(p).clamp(eps, 1 - eps)
    return torch.log(p / (1 - p))


def _softplus_inv(y):
    """Inverse softplus ensuring numerical stability."""
    y = torch.as_tensor(y)
    return torch.log(torch.expm1(y.clamp_min(1e-8)))


class PCENLayer(nn.Module):
    """Per-Channel Energy Normalization using cumsum-based EMA.

    This implementation follows the closed-form cumsum formulation to avoid
    explicit time-step loops. Parameters ``s``, ``alpha``, ``delta`` and ``r``
    are learned per channel and constrained to sensible ranges using sigmoid
    or softplus transforms.

    Args:
        in_channels: Number of input channels.
        alpha: Exponent for the EMA denominator.
        smooth_coef: Smoothing coefficient ``s`` of the EMA.
        delta: Bias added before the power ``r``.
        root: Inverse of the exponent ``r`` (``r = 1/root``).
        floor: Epsilon added inside the denominator for stability.
        trainable: If ``False`` parameters are frozen.
    """

    def __init__(
        self,
        in_channels: int,
        alpha: float = 0.96,
        smooth_coef: float = 0.04,
        delta: float = 2.0,
        root: float = 2.0,
        floor: float = 1e-6,
        trainable: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        self.n_channels = in_channels
        self.eps = float(floor)

        # ``r`` is the exponent in the PCEN formula. ``root`` in the original
        # implementation corresponds to ``1 / r``.
        r = 1.0 / float(root)

        # Store raw (unconstrained) params and map to valid ranges in forward.
        self._s_raw = nn.Parameter(
            torch.full((in_channels,), _logit(smooth_coef)), requires_grad=trainable
        )
        self._alpha_raw = nn.Parameter(
            torch.full((in_channels,), _logit(alpha)), requires_grad=trainable
        )
        self._r_raw = nn.Parameter(
            torch.full((in_channels,), _logit(r)), requires_grad=trainable
        )
        self._delta_raw = nn.Parameter(
            torch.full((in_channels,), _softplus_inv(delta)), requires_grad=trainable
        )

        # Bounds kept as buffers so they move with the module's device.
        self.register_buffer("_zero", torch.tensor(0.0))
        self.register_buffer("_one", torch.tensor(1.0))

    def _s(self):
        # map raw -> (0,1)
        return torch.sigmoid(self._s_raw)

    def _alpha(self):
        return torch.sigmoid(self._alpha_raw)

    def _r(self):
        return torch.sigmoid(self._r_raw)

    def _delta(self):
        return torch.nn.functional.softplus(self._delta_raw)

    def forward(self, x: torch.Tensor, return_m: bool = False):
        """Compute PCEN.

        Args:
            x: Input tensor of shape ``(B, C, T)``.
            return_m: If ``True`` also return the computed EMA ``M``.
        Returns:
            Tensor of shape ``(B, C, T)`` (and optionally ``M``).
        """

        assert (
            x.dim() == 3 and x.size(1) == self.n_channels
        ), "Input must be of shape (B, C, T)"
        E = x.transpose(1, 2)  # (B, T, C)
        B, T, C = E.shape

        s = self._s()
        a = 1.0 - s
        alpha = self._alpha()
        delta = self._delta()
        r = self._r()

        if T == 0:
            return (x, x) if return_m else x

        # Initial M_0 uses E_0 for stability.
        M0 = E[:, 0, :]

        if T == 1:
            M = M0[:, None, :]
        else:
            powers = torch.arange(1, T, device=E.device, dtype=E.dtype).unsqueeze(1)
            pow_seq = a.pow(powers)
            invpow_seq = a.pow(-powers)
            E_rest = E[:, 1:, :]
            R = s.view(1, 1, C) * invpow_seq.view(1, T - 1, C) * E_rest
            Csum = torch.cumsum(R, dim=1)
            M_tail = pow_seq.view(1, T - 1, C) * (M0.view(B, 1, C) + Csum)
            M = torch.cat([M0.view(B, 1, C), M_tail], dim=1)

        denom = (self.eps + M).pow(alpha.view(1, 1, C))
        pcen = (
            (E / denom + delta.view(1, 1, C)).pow(r.view(1, 1, C))
            - delta.view(1, 1, C).pow(r.view(1, 1, C))
        )
        pcen = pcen.transpose(1, 2)  # (B, C, T)
        M = M.transpose(1, 2)  # (B, C, T)
        return (pcen, M) if return_m else pcen

