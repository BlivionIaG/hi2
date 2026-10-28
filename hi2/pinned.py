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

import threading

import torch

DEFAULT_WINDOW = 64 << 20  # 64 MiB


class PinnedWindow:
    """A pinned host window the copy engine can address safely.

    The staging buffers are **per thread**: the encoders shard a tensor across the
    devices and run the shards in parallel threads, and a shared buffer would have the
    shards overwrite each other's data. One window per thread, the buffers reused
    within it, so the pinned memory stays bounded.
    """

    def __init__(self, nbytes: int = DEFAULT_WINDOW):
        self.nbytes = int(nbytes)
        self._local = threading.local()

    def _buf(self, shape: tuple, dtype: torch.dtype) -> torch.Tensor:
        bufs = getattr(self._local, "bufs", None)
        if bufs is None:
            bufs = self._local.bufs = {}
        key = (shape, dtype)
        b = bufs.get(key)
        if b is None:
            b = torch.empty(shape, dtype=dtype, pin_memory=True)
            bufs[key] = b
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


def module_to_device(mod, dev, dtype=None, stager: PinnedWindow | None = None):
    """``mod.to(dev, dtype)``, with every parameter and buffer staged through a pinned
    window.

    The weights are the largest host-to-device traffic in the pipeline (a 27B layer is
    about 1.5 GB, and it is copied to each of the four cards), and they are pageable
    like everything else. Only the floating-point tensors are cast, as
    :meth:`torch.nn.Module.to` does.
    """
    stager = stager if stager is not None else PinnedWindow()
    with torch.no_grad():
        for p in mod.parameters(recurse=True):
            cast = dtype if (dtype is not None and p.is_floating_point()) else None
            p.data = stager.to_device(p.detach(), dev, dtype=cast)
        for b in mod.buffers(recurse=True):
            cast = dtype if (dtype is not None and b.is_floating_point()) else None
            b.data = stager.to_device(b.detach(), dev, dtype=cast)
    return mod


def module_to_host(mod, stager: PinnedWindow | None = None):
    """``mod.cpu()``, with every parameter and buffer staged through a pinned window.

    The final logit-KL phase brings the whole 27B model back to the host (tens of
    gigabytes in one go), which is the single largest device-to-host transfer in the
    pipeline.
    """
    stager = stager if stager is not None else PinnedWindow()
    with torch.no_grad():
        for p in mod.parameters(recurse=True):
            p.data = stager.to_host(p.detach())
        for b in mod.buffers(recurse=True):
            b.data = stager.to_host(b.detach())
    return mod
