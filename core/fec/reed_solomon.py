"""Explicit, full-length Reed-Solomon byte codes using ``reedsolo`` 1.7.0.

Only the unshortened RS(255,223) and RS(255,239) configurations are supported.
Payload bytes and codeword bytes are in their input order; bit conversion at
the boundary uses the project's MSB-first ``bits_to_bytes``/``bytes_to_bits``
helpers. No padding, shortening, or chunking is performed by this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import reedsolo


@dataclass(frozen=True)
class ReedSolomonCodeSpec:
    """One explicit full-length Reed-Solomon configuration."""

    name: str
    n: int
    k: int
    parity_symbols: int
    correction_capability: int


RS_CODE_SPECS: dict[str, ReedSolomonCodeSpec] = {
    "RS_255_223": ReedSolomonCodeSpec(
        name="RS_255_223", n=255, k=223, parity_symbols=32, correction_capability=16
    ),
    "RS_255_239": ReedSolomonCodeSpec(
        name="RS_255_239", n=255, k=239, parity_symbols=16, correction_capability=8
    ),
}


@dataclass
class ReedSolomonDecodeResult:
    """Decode status and structural checks for one exact RS codeword."""

    code: str
    n: int
    k: int
    parity_symbols: int
    correction_capability: int
    decoded_bytes: bytes = b""
    corrected_codeword: bytes = b""
    corrected_error_count: int | None = None
    decode_success: bool = False
    valid: bool = False
    input_length: int = 0
    output_length: int = 0
    error_message: str | None = None
    diagnostics: dict[str, object] = field(default_factory=dict)


def _resolve_code(code: str | ReedSolomonCodeSpec) -> ReedSolomonCodeSpec:
    if isinstance(code, ReedSolomonCodeSpec):
        known = RS_CODE_SPECS.get(code.name)
        if known != code:
            raise ValueError("code specification must match a supported full-length RS entry")
        return known
    if not isinstance(code, str) or code not in RS_CODE_SPECS:
        supported = ", ".join(RS_CODE_SPECS)
        raise ValueError(f"Unsupported Reed-Solomon code {code!r}; supported codes: {supported}")
    return RS_CODE_SPECS[code]


def _as_bytes(data: bytes | bytearray | memoryview, *, label: str) -> bytes:
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise TypeError(f"{label} must be bytes, bytearray, or memoryview")
    return bytes(data)


def _new_codec(spec: ReedSolomonCodeSpec) -> reedsolo.RSCodec:
    # nsize=255 is explicit: reedsolo otherwise permits longer/chunked messages.
    return reedsolo.RSCodec(nsym=spec.parity_symbols, nsize=spec.n)


def rs_encode(
    data_bytes: bytes | bytearray | memoryview,
    code: str | ReedSolomonCodeSpec,
) -> bytes:
    """Encode exactly ``k`` input bytes into exactly one 255-byte codeword."""
    spec = _resolve_code(code)
    payload = _as_bytes(data_bytes, label="data_bytes")
    if len(payload) != spec.k:
        raise ValueError(f"{spec.name} requires exactly {spec.k} data bytes; received {len(payload)}")
    encoded = bytes(_new_codec(spec).encode(payload))
    if len(encoded) != spec.n:
        raise RuntimeError(f"reedsolo returned {len(encoded)} bytes; expected exactly {spec.n}")
    return encoded


def rs_decode(
    codeword_bytes: bytes | bytearray | memoryview,
    code: str | ReedSolomonCodeSpec,
) -> ReedSolomonDecodeResult:
    """Decode one exact codeword and validate the returned codeword structure.

    ``reedsolo`` 1.7.0 returns ``(message, corrected_message, errata_positions)``.
    A returned message alone is not trusted: validity also requires the expected
    payload/codeword lengths, no more than ``t`` reported errata, and an exact
    match between the corrected word and re-encoding the returned payload.
    """
    spec = _resolve_code(code)
    raw = _as_bytes(codeword_bytes, label="codeword_bytes")
    base = {
        "code": spec.name,
        "n": spec.n,
        "k": spec.k,
        "parity_symbols": spec.parity_symbols,
        "correction_capability": spec.correction_capability,
        "input_length": len(raw),
    }
    if len(raw) != spec.n:
        return ReedSolomonDecodeResult(
            **base,
            error_message=f"{spec.name} requires exactly {spec.n} codeword bytes; received {len(raw)}",
            diagnostics={"exact_codeword_length": False, "shortening": "not supported"},
        )

    try:
        decoded_value, corrected_value, errata_positions = _new_codec(spec).decode(raw)
    except (reedsolo.ReedSolomonError, ValueError, IndexError, ZeroDivisionError) as exc:
        return ReedSolomonDecodeResult(
            **base,
            error_message=str(exc),
            diagnostics={"exact_codeword_length": True, "reedsolo_decode_returned": False},
        )

    decoded = bytes(decoded_value)
    corrected = bytes(corrected_value)
    errata_count = len(errata_positions)
    reencoded = (
        bytes(_new_codec(spec).encode(decoded)) if len(decoded) == spec.k else b""
    )
    structural_checks = {
        "exact_codeword_length": len(raw) == spec.n,
        "exact_payload_length": len(decoded) == spec.k,
        "exact_corrected_codeword_length": len(corrected) == spec.n,
        "reported_errata_within_capability": errata_count <= spec.correction_capability,
        "reencoded_matches_corrected_codeword": bool(reencoded) and reencoded == corrected,
    }
    valid = all(structural_checks.values())
    return ReedSolomonDecodeResult(
        **base,
        decoded_bytes=decoded,
        corrected_codeword=corrected,
        corrected_error_count=errata_count,
        decode_success=True,
        valid=valid,
        output_length=len(decoded),
        error_message=None if valid else "reedsolo output failed one or more structural validity checks",
        diagnostics={
            "exact_codeword_length": True,
            "reedsolo_decode_returned": True,
            "reedsolo_version_api": "RSCodec(nsym=..., nsize=255).decode -> (message, corrected_message, errata_positions)",
            "errata_positions": tuple(int(position) for position in errata_positions),
            "structural_checks": structural_checks,
            "shortening": "not supported",
            "padding": "not applied",
        },
    )
