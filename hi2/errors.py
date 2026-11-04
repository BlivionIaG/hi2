# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
"""Stable failure categories shared by local automation adapters."""


class IncompatibleResume(ValueError):
    """Preserved artifacts do not belong to the requested execution identity."""


def describe_error(exc):
    if isinstance(exc, KeyboardInterrupt):
        return (
            "interrupted",
            130,
            "Preserved work remains in place; inspect hi2 status WORKDIR and replay the unchanged recipe.",
        )
    if isinstance(exc, SystemExit) and isinstance(exc.code, str):
        return (
            "invalid_input",
            2,
            "Correct the input or recipe; use COMMAND --help and quantize --dry-run before retrying.",
        )
    if isinstance(exc, IncompatibleResume):
        return (
            "incompatible_resume",
            4,
            "Keep existing artifacts; choose fresh work/output paths for changed inputs or settings.",
        )
    if type(exc).__name__ == "QualityRejected":
        return (
            "quality_rejected",
            5,
            "Inspect hi2_quality.json; choose a higher-quality recipe and evaluate again on independent final data.",
        )
    if isinstance(exc, (ImportError, PermissionError, MemoryError, RuntimeError)):
        return (
            "backend_resource",
            3,
            "Run hi2 doctor; check dependencies, selected devices, permissions and phase-specific resource estimates.",
        )
    if isinstance(
        exc,
        (
            ValueError,
            TypeError,
            FileNotFoundError,
            IsADirectoryError,
            NotADirectoryError,
        ),
    ):
        return (
            "invalid_input",
            2,
            "Correct the input or recipe; use COMMAND --help and quantize --dry-run before retrying.",
        )
    if isinstance(exc, OSError):
        return (
            "backend_resource",
            3,
            "Check filesystem capacity and permissions; preserved work must be resumed with the same recipe.",
        )
    return (
        "worker_failure",
        6,
        "Retry with --debug for a traceback; inspect preserved run/job diagnostics before changing or deleting artifacts.",
    )
