"""Activation and weight observers for collecting quantization statistics."""
import torch
import torch.nn as nn


class MinMaxObserver(nn.Module):
    """Tracks per-tensor min/max across forward passes.

    After calibration, provides the quantization scale and zero-point
    for symmetric INT8 quantization: scale = max(|min|, |max|) / 127.
    """

    def __init__(self, per_channel: bool = False, ch_axis: int = 0):
        super().__init__()
        self.per_channel = per_channel
        self.ch_axis = ch_axis
        self.register_buffer("min_val", torch.tensor(float("inf")))
        self.register_buffer("max_val", torch.tensor(float("-inf")))
        self.register_buffer("num_batches", torch.tensor(0, dtype=torch.long))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            if self.per_channel:
                # Reduce all dims except ch_axis
                dims = [i for i in range(x.ndim) if i != self.ch_axis]
                if dims:
                    batch_min = x.amin(dim=dims)
                    batch_max = x.amax(dim=dims)
                else:
                    batch_min = x
                    batch_max = x
                if self.num_batches == 0:
                    self.min_val = batch_min.clone()
                    self.max_val = batch_max.clone()
                else:
                    self.min_val = torch.minimum(self.min_val, batch_min)
                    self.max_val = torch.maximum(self.max_val, batch_max)
            else:
                self.min_val = torch.minimum(self.min_val, x.min())
                self.max_val = torch.maximum(self.max_val, x.max())
            self.num_batches += 1
        return x

    def compute_scale(self) -> torch.Tensor:
        """Compute symmetric quantization scale: max_abs / 127."""
        max_abs = torch.maximum(self.min_val.abs(), self.max_val.abs())
        return (max_abs / 127.0).clamp(min=1e-8)

    def reset(self):
        self.min_val.fill_(float("inf"))
        self.max_val.fill_(float("-inf"))
        self.num_batches.zero_()


class PercentileObserver(nn.Module):
    """Tracks activation distribution using histograms for percentile-based clipping.

    Clips outliers at the given percentile (e.g., 99.99%) before computing
    the quantization scale. Often gives better accuracy than pure min/max.
    """

    def __init__(self, percentile: float = 99.99, num_bins: int = 2048):
        super().__init__()
        self.percentile = percentile
        self.num_bins = num_bins
        self.register_buffer("histogram", torch.zeros(num_bins))
        self.register_buffer("bin_edges", torch.zeros(num_bins + 1))
        self.register_buffer("min_val", torch.tensor(float("inf")))
        self.register_buffer("max_val", torch.tensor(float("-inf")))
        self.register_buffer("num_batches", torch.tensor(0, dtype=torch.long))
        self._initialized = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            x_flat = x.float().flatten()
            batch_min = x_flat.min()
            batch_max = x_flat.max()
            self.min_val = torch.minimum(self.min_val, batch_min)
            self.max_val = torch.maximum(self.max_val, batch_max)

            if not self._initialized:
                self.bin_edges = torch.linspace(
                    batch_min.item(), batch_max.item(),
                    self.num_bins + 1, device=x.device)
                self._initialized = True

            hist = torch.histc(x_flat, bins=self.num_bins,
                               min=self.bin_edges[0].item(),
                               max=self.bin_edges[-1].item())
            self.histogram += hist.to(self.histogram.device)
            self.num_batches += 1
        return x

    def compute_scale(self) -> torch.Tensor:
        """Compute scale from percentile clipping."""
        total = self.histogram.sum()
        if total == 0:
            return torch.tensor(1.0)
        cumsum = self.histogram.cumsum(dim=0) / total
        threshold = self.percentile / 100.0
        idx = (cumsum >= threshold).nonzero(as_tuple=True)[0]
        if len(idx) == 0:
            clip_val = self.max_val.abs()
        else:
            clip_val = self.bin_edges[idx[0] + 1].abs()
        # Also consider negative side
        idx_lo = (cumsum >= (1.0 - threshold)).nonzero(as_tuple=True)[0]
        if len(idx_lo) > 0:
            clip_val = torch.maximum(clip_val, self.bin_edges[idx_lo[0]].abs())
        return (clip_val / 127.0).clamp(min=1e-8)

    def reset(self):
        self.histogram.zero_()
        self.min_val.fill_(float("inf"))
        self.max_val.fill_(float("-inf"))
        self.num_batches.zero_()
        self._initialized = False
