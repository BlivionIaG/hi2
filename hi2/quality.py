# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
"""Explicit acceptance of a final artifact, separate from allocation predictions."""

import json
import math
from pathlib import Path

from .runtime import SCHEMA_VERSION, file_digest


class QualityRejected(ValueError):
    def __init__(self, report):
        self.report = report
        super().__init__(
            f"final quality gate {report['status']}: {report.get('failures', [])}"
        )


def assess(metrics: dict | None, *, max_kld=None, max_ppl_ratio=None) -> dict:
    criteria = {
        k: v
        for k, v in {"max_kld": max_kld, "max_ppl_ratio": max_ppl_ratio}.items()
        if v is not None
    }
    for name, limit in criteria.items():
        if (
            isinstance(limit, bool)
            or not isinstance(limit, (int, float))
            or not math.isfinite(limit)
            or limit < 0
        ):
            raise ValueError(f"{name} must be a finite nonnegative number")
    report = {
        "schema_version": SCHEMA_VERSION,
        "context": "dense_numerical",
        "criteria": criteria,
        "metrics": {
            key: None
            if isinstance(value, float) and not math.isfinite(value)
            else value
            for key, value in metrics.items()
        }
        if metrics is not None
        else None,
        "failures": [],
    }
    if metrics is None:
        return {**report, "status": "not_measured"}
    required = ("tokens", "ppl_original", "ppl_quantized", "kld_nats_per_token")
    if any(
        k not in metrics
        or isinstance(metrics[k], bool)
        or not isinstance(metrics[k], (int, float))
        or not math.isfinite(metrics[k])
        for k in required
    ):
        return {
            **report,
            "status": "not_measured",
            "failures": ["missing or nonfinite required measurements"],
        }
    if (
        metrics["tokens"] <= 0
        or metrics["ppl_original"] <= 0
        or metrics["ppl_quantized"] <= 0
    ):
        return {
            **report,
            "status": "not_measured",
            "failures": ["no usable tokens or invalid perplexity"],
        }
    ratio = metrics["ppl_quantized"] / metrics["ppl_original"]
    if not math.isfinite(ratio):
        return {
            **report,
            "status": "not_measured",
            "failures": ["nonfinite perplexity ratio"],
        }
    report["metrics"]["ppl_ratio"] = ratio
    if max_kld is not None and metrics["kld_nats_per_token"] > max_kld:
        report["failures"].append("max_kld")
    if max_ppl_ratio is not None and ratio > max_ppl_ratio:
        report["failures"].append("max_ppl_ratio")
    report["status"] = (
        "target_not_met"
        if report["failures"]
        else "target_met"
        if criteria
        else "measured"
    )
    return report


def check_evaluation_split(text: Path, excluded: list[Path]) -> dict:
    """Reject exact document overlap with known calibration/tuning inputs."""
    from .calib import read_documents

    text = Path(text).resolve()
    documents = set(read_documents(text))
    if not documents:
        raise ValueError("final evaluation corpus is empty")
    checked = []
    for path in excluded:
        path = Path(path).resolve()
        if path == text:
            raise ValueError(
                f"final evaluation input is also calibration/tuning data: {path}"
            )
        if overlap := documents.intersection(read_documents(path)):
            raise ValueError(
                f"final evaluation overlaps {path}: {len(overlap)} exact documents"
            )
        checked.append(str(path))
    return {
        "path": str(text),
        "sha256": file_digest(text),
        "documents": len(documents),
        "exact_overlap_checked_against": checked,
        "semantic_deduplication": "not_measured",
    }


def _recorded_calibration_paths(calibration) -> list[Path]:
    """Resolve overlap inputs only after checking their recorded content identity."""
    if not isinstance(calibration, dict) or not isinstance(
        calibration.get("paths", []), list
    ):
        raise ValueError("invalid recorded calibration identity")
    paths = []
    for item in calibration.get("paths", []):
        if not isinstance(item, dict):
            raise ValueError("invalid recorded calibration file identity")
        path, digest = item.get("path"), item.get("sha256")
        if (
            not isinstance(path, str)
            or not path
            or not Path(path).is_absolute()
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest.lower())
        ):
            raise ValueError("invalid recorded calibration file identity")
        path = Path(path)
        if file_digest(path) != digest.lower():
            raise ValueError(f"recorded calibration file identity differs: {path}")
        paths.append(path)
    return paths


def evaluate_quality(
    src: Path,
    dst: Path,
    text: Path,
    *,
    max_kld=None,
    max_ppl_ratio=None,
    excluded: list[Path] | None = None,
    strict=True,
    samples=8,
    seq_len=2048,
    device="cpu",
    dtype="float32",
    log=print,
) -> dict:
    from .cache import auxiliary_model_inputs, model_fingerprint
    from .checkpoint import (
        _artifact_output_lock,
        _publish_quality_report,
        validate_checkpoint,
    )
    from .evalq import evaluate

    if samples < 1 or seq_len < 2:
        raise ValueError("evaluation needs samples >= 1 and seq_len >= 2")
    assess(None, max_kld=max_kld, max_ppl_ratio=max_ppl_ratio)
    src, dst, text = Path(src), Path(dst), Path(text)
    with _artifact_output_lock(dst):
        validate_checkpoint(dst, log=log)
        manifest_path = dst / "hi2_run.json"
        manifest = (
            json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        )
        source = manifest.get("source", {})
        auxiliary_inputs = auxiliary_model_inputs()
        if auxiliary_inputs != source.get("auxiliary_inputs", []):
            raise ValueError(
                "evaluation auxiliary model inputs differ from the recorded "
                "quantization source"
            )
        expected_source = source.get("fingerprint")
        source_fingerprint = model_fingerprint(src)
        if expected_source is not None and source_fingerprint != expected_source:
            raise ValueError(
                "evaluation reference differs from the recorded quantization source"
            )
        known = _recorded_calibration_paths(manifest.get("calibration", {}))
        known.extend(Path(p) for p in (excluded or []))
        dataset = check_evaluation_split(text, known)
        metrics = evaluate(
            src,
            dst,
            text,
            samples=samples,
            seq_len=seq_len,
            device=device,
            skip_tokens=0,
            log=log,
            dtype=dtype,
        )
        report = assess(metrics, max_kld=max_kld, max_ppl_ratio=max_ppl_ratio)
        report.update(
            dataset=dataset,
            artifact={"path": str(dst.resolve()), "run_id": manifest.get("run_id")},
            native_runtime="not_measured",
            sampling={"samples": samples, "seq_len": seq_len, "dtype": dtype},
        )
        report["reference"] = {
            "source_fingerprint": source_fingerprint,
            "auxiliary_inputs": auxiliary_inputs,
            "matches_quantization_source": True
            if expected_source is not None
            else None,
        }
        _publish_quality_report(dst, report)
        if strict and report["status"] in ("not_measured", "target_not_met"):
            raise QualityRejected(report)
        return report
