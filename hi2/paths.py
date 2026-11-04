# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
"""Pure preflight checks shared by checkpoint, calibration and tuning writers."""

from collections.abc import Sequence
from pathlib import Path


def check_output_paths(*, inputs: Sequence[Path], outputs: Sequence[Path]) -> None:
    """Reject aliases and overlapping trees before any output is created.

    Missing paths are resolved through existing symlink parents. Input/input overlap
    is harmless; input/output and output/output overlap is not. Existing hard links
    are aliases too. This does not grant permission to reuse an existing output:
    the caller must separately establish compatible ownership of that destination.
    """
    try:
        sources = [(Path(p), Path(p).resolve()) for p in inputs]
        destinations = [(Path(p), Path(p).resolve()) for p in outputs]
    except (OSError, RuntimeError) as exc:
        raise ValueError(f"cannot resolve input/output paths: {exc}") from exc
    for i, (original, resolved) in enumerate(destinations):
        for other, target in sources + destinations[:i]:
            overlap = (
                resolved == target
                or resolved.is_relative_to(target)
                or target.is_relative_to(resolved)
            )
            if not overlap and resolved.exists() and target.exists():
                overlap = resolved.samefile(target)
            # Case-insensitive filesystems can spell an existing ancestor
            # differently even when a not-yet-created child has no inode.
            if not overlap and target.exists():
                overlap = any(
                    p.exists() and p.samefile(target) for p in resolved.parents
                )
            if not overlap and resolved.exists():
                overlap = any(
                    p.exists() and p.samefile(resolved) for p in target.parents
                )
            if overlap:
                raise ValueError(f"output path {original} overlaps {other}")
