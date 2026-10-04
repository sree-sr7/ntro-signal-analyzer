"""Inference-interface metadata for the production ONNX classifier.

Model weights are stored separately in ``models/production_mlp_final.onnx``.
This module defines the fallback class order and caller-side configuration.
"""

from dataclasses import dataclass

from core.common.enums import ModulationType


FEATURE_SCHEMA_VERSION = "phase3-modulation-features-v1"
MODEL_LABELS_METADATA_KEY = "modulation_labels"
DEFAULT_MODEL_CLASS_ORDER = (
    ModulationType.BPSK,
    ModulationType.QPSK,
    ModulationType.PSK8,
    ModulationType.QAM16,
    ModulationType.FSK2,
)


@dataclass(frozen=True)
class OnnxModelConfiguration:
    """Optional caller-supplied class and feature ordering for ONNX inference."""

    class_order: tuple[ModulationType, ...] | None = None
    feature_names: tuple[str, ...] | None = None
    feature_schema_version: str = FEATURE_SCHEMA_VERSION
