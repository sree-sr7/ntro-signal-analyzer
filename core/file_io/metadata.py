"""Metadata normalization and merging utilities.

Provides functions to fill in derived fields (e.g. duration from
sample_count and sample_rate), validate consistency, and merge
user-provided overrides into existing metadata.
"""

import dataclasses
from core.common.models import SignalMetadata


def normalize_metadata(metadata: SignalMetadata) -> SignalMetadata:
    """Compute derived fields and check consistency.

    Returns a new SignalMetadata instance with derived fields filled in.
    Does not mutate the input.
    """
    result = dataclasses.replace(
        metadata,
        warnings=list(metadata.warnings),           # shallow copy
        source_metadata=dict(metadata.source_metadata),
    )

    if result.sample_rate is not None and result.sample_rate > 0:
        # Fill in duration if we have sample_count but no duration
        if result.sample_count > 0 and result.duration_seconds is None:
            result.duration_seconds = result.sample_count / result.sample_rate

        # Fill in sample_count if we have duration but no sample_count
        elif (
            result.duration_seconds is not None
            and result.duration_seconds > 0
            and result.sample_count == 0
        ):
            result.sample_count = int(result.duration_seconds * result.sample_rate)

        # Check consistency when both are present
        if result.sample_count > 0 and result.duration_seconds is not None:
            expected_duration = result.sample_count / result.sample_rate
            if abs(expected_duration - result.duration_seconds) > 0.001:
                result.warnings.append(
                    f"Duration mismatch: sample_count/sample_rate = "
                    f"{expected_duration:.6f}s, but duration_seconds = "
                    f"{result.duration_seconds:.6f}s"
                )

    return result


def merge_metadata(
    base: SignalMetadata,
    override: dict,
) -> SignalMetadata:
    """Apply override values to a copy of base metadata.

    Only fields that (a) exist on SignalMetadata and (b) have a
    non-None value in *override* are applied.

    Returns a new SignalMetadata; does not mutate *base*.
    """
    valid_overrides = {
        k: v
        for k, v in override.items()
        if v is not None and hasattr(base, k)
    }
    return dataclasses.replace(base, **valid_overrides)
