"""Classical bounded classifier and explicit CPU-only ONNX inference wrapper."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from core.common.enums import ClassifierStatus, ModulationType
from core.common.models import ClassificationResult, ModulationFeatureResult
from core.dsp.modulation_features import extract_modulation_features
from .model_def import DEFAULT_MODEL_CLASS_ORDER, MODEL_LABELS_METADATA_KEY, OnnxModelConfiguration


_PSK_FEATURES = {
    ModulationType.BPSK: ("psk2_residual", "psk2_phase_entropy"),
    ModulationType.QPSK: ("psk4_residual", "psk4_phase_entropy"),
    ModulationType.PSK8: ("psk8_residual", "psk8_phase_entropy"),
}

# This is the existing bounded-classifier floor, reused by the analyzer as a
# transparent structural signal-presence check.  These scores are geometric
# fit evidence, not calibrated class probabilities.
MIN_CLASS_EVIDENCE_SCORE = 0.35


def _baseline_scores(features: dict[str, float]) -> dict[ModulationType, float]:
    """Convert normalized geometric/phase features into bounded class scores."""
    scores: dict[ModulationType, float] = {}
    residual_scale = 0.24
    entropy_scale = 0.23
    for modulation, (residual_name, entropy_name) in _PSK_FEATURES.items():
        residual = features[residual_name]
        entropy_error = 1.0 - features[entropy_name]
        scores[modulation] = float(np.exp(
            -0.5 * (residual / residual_scale) ** 2
            -0.5 * (entropy_error / entropy_scale) ** 2
        ))
    qam_residual = features["qam16_residual"]
    amplitude_cv = features["amplitude_coefficient_of_variation"]
    # Unit-energy rectangular 16-QAM has three nonuniform radius levels and
    # amplitude CV near 0.34; this rejects PSK subsets lying on QAM corners.
    qam_radial_fit = np.exp(-0.5 * ((amplitude_cv - 0.34) / 0.16) ** 2)
    scores[ModulationType.QAM16] = float(
        np.exp(-0.5 * (qam_residual / 0.25) ** 2) * qam_radial_fit
    )
    fsk_quality = float(np.clip(features["fsk_quality"], 0.0, 1.0))
    scores[ModulationType.FSK2] = float(fsk_quality ** 1.5)
    # At waveform sample rate, a two-tone FSK signal can look like a phase
    # sequence to the PSK residual tests. A strong two-state frequency/dwell
    # fit is direct evidence for FSK, so suppress those competing fits.
    if fsk_quality > 0.55:
        non_fsk_factor = float((1.0 - fsk_quality) ** 2)
        for modulation in _PSK_FEATURES:
            scores[modulation] *= non_fsk_factor
        scores[ModulationType.QAM16] *= non_fsk_factor
    return scores


def strongest_modulation_evidence(
    features: dict[str, float],
) -> tuple[ModulationType, float]:
    """Return the strongest in-scope geometric fit score and its class.

    This reuses the current classical classifier's explicit fit rules. The
    returned score is not a probability and must not be presented as one.
    """
    scores = _baseline_scores(features)
    if not scores:
        return ModulationType.UNKNOWN, 0.0
    modulation, score = max(scores.items(), key=lambda item: item[1])
    return modulation, float(score)


def _classification_result(
    scores: dict[ModulationType, float],
    *,
    method: str,
    features: dict[str, float],
    sample_count: int,
    warnings: list[str] | None = None,
    status: ClassifierStatus = ClassifierStatus.SUCCESS,
) -> ClassificationResult:
    """Rank scores and return UNKNOWN when evidence lacks strength or margin."""
    ranked_scores = sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
    total = float(sum(max(value, 0.0) for _, value in ranked_scores))
    probabilities = (
        {key: float(max(value, 0.0) / total) for key, value in ranked_scores}
        if total > np.finfo(float).tiny
        else {}
    )
    ranked = sorted(probabilities.items(), key=lambda pair: pair[1], reverse=True)
    candidates = ranked[:3]
    result_warnings = list(warnings or [])
    if not ranked:
        return ClassificationResult(
            method=method, warnings=result_warnings + ["No usable class scores were produced"],
            status=status, features=features,
        )
    best_class, best_probability = ranked[0]
    second_probability = ranked[1][1] if len(ranked) > 1 else 0.0
    margin = best_probability - second_probability
    margin_fraction = margin / max(best_probability, np.finfo(float).tiny)
    sample_support = float(1.0 - np.exp(-sample_count / 64.0))
    confidence = float(np.clip(best_probability * margin_fraction * sample_support, 0.0, 1.0))
    # The margin measures separation from the runner-up; the score floor stops
    # uniformly poor/noise-like feature fits from being presented as a class.
    accepted = (
        best_probability >= 0.45
        and margin >= 0.12
        and scores[best_class] >= MIN_CLASS_EVIDENCE_SCORE
    )
    modulation = best_class if accepted else ModulationType.UNKNOWN
    if not accepted:
        result_warnings.append("Class evidence is weak or ambiguous; returning ranked candidates")
    return ClassificationResult(
        modulation=modulation,
        confidence=confidence if accepted else min(confidence, 0.49),
        method=method,
        probabilities=probabilities,
        top_candidates=candidates,
        warnings=result_warnings,
        status=status,
        features=features,
    )


def classify_classical(
    samples: np.ndarray,
    *,
    samples_per_symbol: float = 1.0,
) -> ClassificationResult:
    """Classify among BPSK/QPSK/8PSK/16QAM/2FSK with explicit baseline rules.

    Candidate scores combine normalized nearest-constellation residuals,
    PSK decision-label entropy, and FSK two-tone dwell quality. Scores are
    normalized into a ranking; confidence is the top-vs-runner-up probability
    margin multiplied by a sample-support factor. This is a baseline score,
    not a calibrated probability or arbitrary-modulation recognizer.
    """
    feature_result: ModulationFeatureResult = extract_modulation_features(
        samples, samples_per_symbol=samples_per_symbol
    )
    feature_map = feature_result.as_dict()
    if feature_result.sample_count < 32 or feature_result.warnings:
        return ClassificationResult(
            method="classical_baseline",
            status=ClassifierStatus.INSUFFICIENT_DATA,
            warnings=list(feature_result.warnings) or ["At least 32 samples are required for classification"],
            features=feature_map,
        )
    scores = _baseline_scores(feature_map)
    return _classification_result(
        scores,
        method="classical_baseline",
        features=feature_map,
        sample_count=feature_result.sample_count,
    )


def _coerce_label(label: str) -> ModulationType:
    """Parse common ONNX metadata label spellings into supported enums."""
    key = label.strip().lower().replace("-", "").replace("_", "")
    aliases = {
        "bpsk": ModulationType.BPSK,
        "qpsk": ModulationType.QPSK,
        "8psk": ModulationType.PSK8,
        "psk8": ModulationType.PSK8,
        "16qam": ModulationType.QAM16,
        "qam16": ModulationType.QAM16,
        "2fsk": ModulationType.FSK2,
        "fsk2": ModulationType.FSK2,
    }
    if key not in aliases:
        raise ValueError(f"Unsupported ONNX modulation label: {label}")
    return aliases[key]


class OnnxModulationClassifier:
    """Run a caller-provided ONNX model on CPU without implicit fallback.

    The wrapper never downloads or creates a model. Missing files, runtime
    import/load failures, shape mismatches, and inference errors are surfaced
    through a ``ClassificationResult`` with UNKNOWN modulation and status.
    ``modulation_labels`` model metadata may contain a JSON list of output
    labels; otherwise the supported five-class order from ``model_def`` is used.
    """

    def __init__(
        self,
        model_path: str | Path | None,
        *,
        configuration: OnnxModelConfiguration | None = None,
    ) -> None:
        self.model_path = Path(model_path) if model_path is not None else None
        self.configuration = configuration or OnnxModelConfiguration()
        self._session = None
        self._input = None
        self._class_order: tuple[ModulationType, ...] | None = self.configuration.class_order
        self._load_error: str | None = None
        self._load_status = ClassifierStatus.MODEL_UNAVAILABLE
        if self.model_path is None or not self.model_path.is_file():
            self._load_error = f"ONNX model unavailable: {self.model_path or 'no model path supplied'}"
            return
        try:
            import onnxruntime as ort

            self._session = ort.InferenceSession(
                str(self.model_path), providers=["CPUExecutionProvider"]
            )
            inputs = self._session.get_inputs()
            if len(inputs) != 1:
                raise ValueError(f"Expected one model input, found {len(inputs)}")
            self._input = inputs[0]
            metadata = self._session.get_modelmeta().custom_metadata_map or {}
            if self._class_order is None and MODEL_LABELS_METADATA_KEY in metadata:
                labels = json.loads(metadata[MODEL_LABELS_METADATA_KEY])
                if not isinstance(labels, list):
                    raise ValueError("modulation_labels metadata must be a JSON list")
                self._class_order = tuple(_coerce_label(str(label)) for label in labels)
            self._load_status = ClassifierStatus.SUCCESS
        except Exception as exc:  # runtime errors are returned structurally at inference time
            self._session = None
            self._input = None
            self._load_error = f"Unable to load ONNX model: {exc}"
            self._load_status = ClassifierStatus.ERROR

    @property
    def available(self) -> bool:
        """Whether the explicitly supplied model loaded successfully."""
        return self._session is not None and self._input is not None

    def _format_input(self, feature_result: ModulationFeatureResult) -> np.ndarray:
        """Validate the model's static feature dimensions and build its input tensor."""
        assert self._input is not None
        shape = list(self._input.shape)
        feature_count = feature_result.values.size
        if len(shape) == 1:
            if isinstance(shape[0], int) and shape[0] != feature_count:
                raise ValueError(f"Model expects {shape[0]} features; got {feature_count}")
            target_shape = [feature_count]
        else:
            tail = shape[1:]
            fixed_product = int(np.prod([dim for dim in tail if isinstance(dim, int) and dim > 0]))
            dynamic_count = sum(not (isinstance(dim, int) and dim > 0) for dim in tail)
            if dynamic_count == 0 and fixed_product != feature_count:
                raise ValueError(f"Model input shape {shape} does not match {feature_count} features")
            if dynamic_count:
                remaining = feature_count // max(fixed_product, 1)
                if feature_count % max(fixed_product, 1):
                    raise ValueError(f"Model input shape {shape} cannot hold {feature_count} features")
                target_shape = [1] + [1] * len(tail)
                dynamic_positions = [i + 1 for i, dim in enumerate(tail)
                                     if not (isinstance(dim, int) and dim > 0)]
                target_shape[dynamic_positions[-1]] = remaining
                for i, dim in enumerate(tail, start=1):
                    if isinstance(dim, int) and dim > 0:
                        target_shape[i] = dim
            else:
                target_shape = [1] + [int(dim) for dim in tail]
        try:
            return feature_result.values.astype(np.float32, copy=False).reshape(target_shape)
        except ValueError as exc:
            raise ValueError(f"Cannot reshape feature vector to model input shape {shape}") from exc

    def classify(
        self,
        samples: np.ndarray,
        *,
        samples_per_symbol: float = 1.0,
    ) -> ClassificationResult:
        """Run CPU inference and return labels/scores or an explicit failure."""
        if not self.available:
            unavailable = self._load_status is ClassifierStatus.MODEL_UNAVAILABLE
            return ClassificationResult(
                method="onnx",
                status=self._load_status,
                warnings=[
                    "No trained ONNX model is available; no baseline fallback was run"
                    if unavailable else "ONNX model failed to load; no baseline fallback was run"
                ],
                errors=[self._load_error or "ONNX model unavailable"],
            )
        feature_result = extract_modulation_features(samples, samples_per_symbol=samples_per_symbol)
        feature_map = feature_result.as_dict()
        if feature_result.warnings:
            return ClassificationResult(
                method="onnx", status=ClassifierStatus.INSUFFICIENT_DATA,
                warnings=list(feature_result.warnings), features=feature_map,
            )
        try:
            if self.configuration.feature_names is not None:
                if tuple(feature_result.feature_names) != self.configuration.feature_names:
                    raise ValueError("Feature names/order do not match the supplied ONNX configuration")
            tensor = self._format_input(feature_result)
            raw = self._session.run(None, {self._input.name: tensor})[0]
            output = np.asarray(raw, dtype=np.float64).reshape(-1)
            if output.size == 0 or not np.all(np.isfinite(output)):
                raise ValueError("Model returned empty or non-finite scores")
            class_order = self._class_order
            if class_order is None:
                if output.size != len(DEFAULT_MODEL_CLASS_ORDER):
                    raise ValueError("Output class count is unknown; provide labels or model metadata")
                class_order = DEFAULT_MODEL_CLASS_ORDER
            if len(class_order) != output.size:
                raise ValueError(f"Model returned {output.size} scores for {len(class_order)} labels")
            if np.all(output >= 0.0) and np.isclose(np.sum(output), 1.0, atol=1e-3):
                probabilities = output / np.sum(output)
            else:
                shifted = output - np.max(output)
                exp = np.exp(shifted)
                probabilities = exp / np.sum(exp)
            scores = {label: float(score) for label, score in zip(class_order, probabilities)}
            result = _classification_result(
                scores,
                method="onnx",
                features=feature_map,
                sample_count=feature_result.sample_count,
            )
            if result.modulation is ModulationType.UNKNOWN:
                result.warnings.append("ONNX scores did not meet the baseline acceptance margin")
            return result
        except Exception as exc:
            return ClassificationResult(
                method="onnx", status=ClassifierStatus.ERROR,
                warnings=["ONNX inference failed; no classical fallback was run"],
                errors=[str(exc)], features=feature_map,
            )
