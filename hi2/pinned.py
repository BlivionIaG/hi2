"""Pinned staging for the bulk host-device copies.

The QAD moves tens of gigabytes per layer between the host's hidden-state buffers
and the cards, and all of it is pageable: the copy engine addresses those host
pages directly. On this driver that ends in a protection fault on a host page
("Page not present or supervisor privilege", ``PERMISSION_FAULTS``) after a couple
of hours of it - three 27B runs died that way, in both directions (a read of the
host's weights, a write of a layer's output).

Routing the copies through a pinned window fixes the mechanism: the driver maps the
pinned pages once, and the copy engine only ever addresses memory that is mapped
and resident. The window is small (64 MiB by default) and reused, so the host
memory stays bounded and nothing has to be page-locked twice.

The transfers keep their shape and their dtype conversion: the cast still happens
on the device, after the copy, exactly as ``tensor.to(dev, dtype)`` would do it.
"""

from __future__ import annotations

import torch

DEFAULT_WINDOW = 64 << 20  # 64 MiB


class PinnedWindow:
    """A pinned host window the copy engine can address safely.

    One instance per thread: the copies run in parallel worker threads, and a
    shared buffer would be a race. The window holds the staging buffers, one per
    (shape, dtype) it has seen.
    """

    def __init__(self, nbytes: int = DEFAULT_WINDOW):
        self.nbytes = int(nbytes)
        self._bufs: dict[tuple, torch.Tensor] = {}

    def _buf(self, shape: tuple, dtype: torch.dtype) -> torch.Tensor:
        key = (shape, dtype)
        b = self._bufs.get(key)
        if b is None:
            b = torch.empty(shape, dtype=dtype, pin_memory=True)
            self._bufs[key] = b
        return b

    def _rows(self, src: torch.Tensor) -> int:
        """How many leading rows fit in the window."""
        per_row = src[0].numel() * src.element_size() if src.dim() else src.element_size()
        if per_row <= 0:
            return 1
        return max(1, self.nbytes // per_row)

    def to_device(self, src: torch.Tensor, dev, dtype=None) -> torch.Tensor:
        """A host tensor to a device tensor, through the pinned window.

        The result is a fresh device tensor, like ``src.to(dev, dtype)`` would give.
        """
        # A CPU target has nothing to stage (and ``.to("cpu")`` would hand back the
        # staging buffer itself, which the next copy would overwrite in place).
        if dev.type == "cpu" or src.device.type != "cpu":
            return src.to(dev, dtype) if dtype is not None else src.to(dev)
        if src.numel() == 0:
            return src.to(dev, dtype) if dtype is not None else src.to(dev)
        if src.numel() * src.element_size() <= self.nbytes:
            b = self._buf(tuple(src.shape), src.dtype)
            b.copy_(src)
            out = b.to(dev, non_blocking=False)
        else:
            rows = self._rows(src)
            out = torch.empty(tuple(src.shape), dtype=src.dtype, device=dev)
            for i in range(0, src.shape[0], rows):
                part = src[i : i + rows]
                b = self._buf(tuple(part.shape), src.dtype)
                b.copy_(part)
                out[i : i + rows].copy_(b.to(dev, non_blocking=False))
        if dtype is not None and dtype != out.dtype:
            out = out.to(dtype)
        return out

    def to_host(self, src: torch.Tensor) -> torch.Tensor:
        """A device tensor to a host tensor, through the pinned window."""
        if src.device.type == "cpu":
            return src
        if src.numel() == 0:
            return src.cpu()
        if src.numel() * src.element_size() <= self.nbytes:
            b = self._buf(tuple(src.shape), src.dtype)
            b.copy_(src)
            return b.clone()
        rows = self._rows(src)
        out = torch.empty(tuple(src.shape), dtype=src.dtype)
        for i in range(0, src.shape[0], rows):
            part = src[i : i + rows]
            b = self._buf(tuple(part.shape), src.dtype)
            b.copy_(part)
            out[i : i + rows].copy_(b)
        return out
