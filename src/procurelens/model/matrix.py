"""Stable preprocessed matrices for ProcureLens detectors.

Collects rows produced by one fitted preprocessor into an immutable rectangular
matrix while preserving row identity, exact feature order, source-row evidence,
and Decimal values. Floating-point conversion is explicit and diagnostic; this
module never changes feature order, refits preprocessing, or scores anomalies.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from hashlib import sha256
import json
import math
import struct
import sys
from types import MappingProxyType
from typing import Any, Iterable, Mapping

from procurelens.model.preprocessor import PreprocessedRow


class MatrixError(ValueError):
    """Raised when model-matrix evidence or numeric conversion is inconsistent."""


@dataclass(frozen=True, slots=True)
class FloatConversionDiagnostic:
    feature_name: str
    value_count: int
    nonzero_roundtrip_error_count: int
    maximum_absolute_roundtrip_error: Decimal

    def __post_init__(self) -> None:
        name = self.feature_name.strip()
        if not name:
            raise MatrixError("feature_name must not be blank")
        object.__setattr__(self, "feature_name", name)
        for field_name in ("value_count", "nonzero_roundtrip_error_count"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise MatrixError(f"{field_name} must be a non-negative integer")
        if self.value_count < 1:
            raise MatrixError("float diagnostic requires at least one value")
        if self.nonzero_roundtrip_error_count > self.value_count:
            raise MatrixError(
                "nonzero_roundtrip_error_count cannot exceed value_count"
            )
        error = self.maximum_absolute_roundtrip_error
        if (
            not isinstance(error, Decimal)
            or not error.is_finite()
            or error < 0
        ):
            raise MatrixError(
                "maximum_absolute_roundtrip_error must be non-negative Decimal"
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "feature_name": self.feature_name,
            "value_count": self.value_count,
            "nonzero_roundtrip_error_count":
                self.nonzero_roundtrip_error_count,
            "maximum_absolute_roundtrip_error":
                str(self.maximum_absolute_roundtrip_error),
        }


@dataclass(frozen=True, slots=True)
class Float64Matrix:
    """Explicit IEEE-754 binary64 view of one Decimal model matrix."""

    source_matrix_sha256: str
    preprocessor_sha256: str
    output_feature_names: tuple[str, ...]
    row_identities: tuple[tuple[str, str], ...]
    values: tuple[tuple[float, ...], ...]
    conversion_diagnostics: tuple[FloatConversionDiagnostic, ...]

    def __post_init__(self) -> None:
        for field_name in ("source_matrix_sha256", "preprocessor_sha256"):
            object.__setattr__(
                self,
                field_name,
                _digest_hex(getattr(self, field_name), field_name),
            )
        names = _feature_names(self.output_feature_names)
        identities = _row_identities(self.row_identities)
        rows = tuple(tuple(row) for row in self.values)
        if len(rows) != len(identities):
            raise MatrixError("float matrix row count differs from identities")
        if any(len(row) != len(names) for row in rows):
            raise MatrixError("float matrix is not rectangular")
        if any(
            not isinstance(value, float) or not math.isfinite(value)
            for row in rows
            for value in row
        ):
            raise MatrixError("float matrix values must be finite Python floats")
        diagnostics = tuple(self.conversion_diagnostics)
        if tuple(item.feature_name for item in diagnostics) != names:
            raise MatrixError(
                "conversion diagnostics must match feature order exactly"
            )
        if any(item.value_count != len(rows) for item in diagnostics):
            raise MatrixError(
                "conversion diagnostics must cover every matrix row"
            )
        object.__setattr__(self, "output_feature_names", names)
        object.__setattr__(self, "row_identities", identities)
        object.__setattr__(self, "values", rows)
        object.__setattr__(self, "conversion_diagnostics", diagnostics)

    @property
    def row_count(self) -> int:
        return len(self.values)

    @property
    def feature_count(self) -> int:
        return len(self.output_feature_names)

    @property
    def evidence_sha256(self) -> str:
        return _digest(
            {
                "source_matrix_sha256": self.source_matrix_sha256,
                "preprocessor_sha256": self.preprocessor_sha256,
                "feature_names": list(self.output_feature_names),
                "row_identities": self.row_identities,
                "values_hex": [
                    [value.hex() for value in row]
                    for row in self.values
                ],
                "conversion_diagnostics": [
                    item.as_dict() for item in self.conversion_diagnostics
                ],
            }
        )


@dataclass(frozen=True, slots=True)
class PreprocessedMatrix:
    """Immutable Decimal matrix built from one fitted preprocessor."""

    preprocessor_sha256: str
    output_feature_names: tuple[str, ...]
    row_identities: tuple[tuple[str, str], ...]
    source_row_evidence_sha256: tuple[str, ...]
    values: tuple[tuple[Decimal, ...], ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "preprocessor_sha256",
            _digest_hex(self.preprocessor_sha256, "preprocessor_sha256"),
        )
        names = _feature_names(self.output_feature_names)
        identities = _row_identities(self.row_identities)
        evidence = tuple(
            _digest_hex(value, "source_row_evidence_sha256")
            for value in self.source_row_evidence_sha256
        )
        rows = tuple(tuple(row) for row in self.values)
        if not rows:
            raise MatrixError("matrix requires at least one row")
        if len(rows) != len(identities) or len(rows) != len(evidence):
            raise MatrixError(
                "matrix row identities/evidence/value counts differ"
            )
        if any(len(row) != len(names) for row in rows):
            raise MatrixError("matrix is not rectangular")
        if any(
            not isinstance(value, Decimal) or not value.is_finite()
            for row in rows
            for value in row
        ):
            raise MatrixError("matrix values must be finite Decimal values")
        object.__setattr__(self, "output_feature_names", names)
        object.__setattr__(self, "row_identities", identities)
        object.__setattr__(self, "source_row_evidence_sha256", evidence)
        object.__setattr__(self, "values", rows)

    @property
    def row_count(self) -> int:
        return len(self.values)

    @property
    def feature_count(self) -> int:
        return len(self.output_feature_names)

    @property
    def row_index(self) -> Mapping[tuple[str, str], int]:
        return MappingProxyType(
            {identity: index for index, identity in enumerate(self.row_identities)}
        )

    @property
    def evidence_sha256(self) -> str:
        return _digest(
            {
                "preprocessor_sha256": self.preprocessor_sha256,
                "feature_names": list(self.output_feature_names),
                "rows": [
                    {
                        "identity": identity,
                        "source_row_evidence_sha256": evidence,
                        "values": [str(value) for value in values],
                    }
                    for identity, evidence, values in zip(
                        self.row_identities,
                        self.source_row_evidence_sha256,
                        self.values,
                    )
                ],
            }
        )

    def to_float64(self) -> Float64Matrix:
        """Convert to IEEE-754 binary64 while measuring Decimal round-trip loss."""

        _require_binary64_python_float()
        float_rows: list[tuple[float, ...]] = []
        error_counts = [0] * self.feature_count
        max_errors = [Decimal(0)] * self.feature_count

        for decimal_row in self.values:
            converted_row: list[float] = []
            for index, value in enumerate(decimal_row):
                converted = float(value)
                if not math.isfinite(converted):
                    raise MatrixError(
                        f"{self.output_feature_names[index]}: "
                        "Decimal value cannot be represented as finite binary64"
                    )
                exact_float_decimal = Decimal.from_float(converted)
                error = abs(exact_float_decimal - value)
                if error != 0:
                    error_counts[index] += 1
                    if error > max_errors[index]:
                        max_errors[index] = error
                converted_row.append(converted)
            float_rows.append(tuple(converted_row))

        diagnostics = tuple(
            FloatConversionDiagnostic(
                feature_name=name,
                value_count=self.row_count,
                nonzero_roundtrip_error_count=error_counts[index],
                maximum_absolute_roundtrip_error=max_errors[index],
            )
            for index, name in enumerate(self.output_feature_names)
        )
        return Float64Matrix(
            source_matrix_sha256=self.evidence_sha256,
            preprocessor_sha256=self.preprocessor_sha256,
            output_feature_names=self.output_feature_names,
            row_identities=self.row_identities,
            values=tuple(float_rows),
            conversion_diagnostics=diagnostics,
        )


def build_preprocessed_matrix(
    rows: Iterable[PreprocessedRow],
) -> PreprocessedMatrix:
    """Build one rectangular matrix without reordering caller-provided rows."""

    if isinstance(rows, (str, bytes)):
        raise MatrixError("rows must be an iterable of PreprocessedRow")
    items = tuple(rows)
    if not items:
        raise MatrixError("at least one preprocessed row is required")
    if any(not isinstance(row, PreprocessedRow) for row in items):
        raise TypeError("all rows must be PreprocessedRow")

    first = items[0]
    preprocessor_sha = first.preprocessor_sha256
    feature_names = first.output_feature_names
    identities: list[tuple[str, str]] = []
    evidence: list[str] = []
    values: list[tuple[Decimal, ...]] = []
    seen: set[tuple[str, str]] = set()

    for row in items:
        if row.preprocessor_sha256 != preprocessor_sha:
            raise MatrixError(
                "rows were produced by different fitted preprocessors"
            )
        if row.output_feature_names != feature_names:
            raise MatrixError(
                "rows have different preprocessed feature order"
            )
        identity = (row.transaction_id, row.award_id)
        if identity in seen:
            raise MatrixError(f"duplicate matrix row identity: {identity!r}")
        seen.add(identity)
        identities.append(identity)
        evidence.append(row.evidence_sha256)
        values.append(row.values)

    return PreprocessedMatrix(
        preprocessor_sha256=preprocessor_sha,
        output_feature_names=feature_names,
        row_identities=tuple(identities),
        source_row_evidence_sha256=tuple(evidence),
        values=tuple(values),
    )


def _feature_names(values: Iterable[str]) -> tuple[str, ...]:
    names = tuple(value.strip() for value in values)
    if not names or any(not name for name in names):
        raise MatrixError("feature names must be non-empty and non-blank")
    if len(names) != len(set(names)):
        raise MatrixError("feature names must be globally unique")
    return names


def _row_identities(
    values: Iterable[tuple[str, str]],
) -> tuple[tuple[str, str], ...]:
    identities: list[tuple[str, str]] = []
    for raw in values:
        if not isinstance(raw, tuple) or len(raw) != 2:
            raise MatrixError(
                "row identities must be (transaction_id, award_id) tuples"
            )
        txid, award_id = raw[0].strip(), raw[1].strip()
        if not txid or not award_id:
            raise MatrixError("row identity values must not be blank")
        identities.append((txid, award_id))
    result = tuple(identities)
    if not result or len(result) != len(set(result)):
        raise MatrixError("row identities must be non-empty and unique")
    return result


def _require_binary64_python_float() -> None:
    if (
        struct.calcsize("d") != 8
        or sys.float_info.radix != 2
        or sys.float_info.mant_dig != 53
        or sys.float_info.max_exp != 1024
    ):
        raise MatrixError(
            "runtime Python float is not the expected IEEE-754 binary64 format"
        )


def _digest_hex(value: str, name: str) -> str:
    digest = value.strip().lower()
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise MatrixError(f"{name} must be a SHA-256 hex digest")
    return digest


def _digest(value: Any) -> str:
    return sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
