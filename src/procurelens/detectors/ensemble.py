"""Calibrated heterogeneous detector ensembles for ProcureLens.

Combines score batches only after each detector has been interpreted against its
own frozen training-score distribution. The ensemble operates on tie-aware
empirical position intervals, not raw detector scores, so detector-specific score
scales cannot silently dominate the combination.

Combination policy is fully explicit. Weighted means require caller-supplied
positive weights that sum exactly to one; maximum and median use no hidden
weights. Per-detector robust tail distances are preserved as evidence but are not
silently mixed into the ensemble score.

No contamination assumption, review threshold, risk score, or misconduct
conclusion lives here.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from hashlib import sha256
import json
from typing import Any

from procurelens.detectors.calibration import (
    CalibratedDetectorScore,
    CalibratedScoreBatch,
)


class EnsembleError(ValueError):
    """Raised when calibrated detector ensemble evidence is inconsistent."""


class EnsembleMethod(str, Enum):
    WEIGHTED_MEAN = "weighted_mean"
    MAXIMUM = "maximum"
    MEDIAN = "median"


@dataclass(frozen=True, slots=True)
class EnsembleMemberSpec:
    """One exact calibrated detector expected by an ensemble specification."""

    member_name: str
    detector_name: str
    calibration_sha256: str
    weight: Decimal | None = None

    def __post_init__(self) -> None:
        for field_name in ("member_name", "detector_name"):
            text = getattr(self, field_name).strip()
            if not text:
                raise EnsembleError(f"{field_name} must not be blank")
            object.__setattr__(self, field_name, text)
        object.__setattr__(
            self,
            "calibration_sha256",
            _digest_hex(self.calibration_sha256, "calibration_sha256"),
        )
        if self.weight is not None and (
            not isinstance(self.weight, Decimal)
            or not self.weight.is_finite()
            or self.weight <= 0
        ):
            raise EnsembleError("member weight must be positive finite Decimal or None")

    def as_dict(self) -> dict[str, Any]:
        return {
            "member_name": self.member_name,
            "detector_name": self.detector_name,
            "calibration_sha256": self.calibration_sha256,
            "weight": None if self.weight is None else str(self.weight),
        }


@dataclass(frozen=True, slots=True)
class EnsembleSpec:
    """Explicit calibrated-score combination contract."""

    name: str
    description: str
    method: EnsembleMethod
    members: tuple[EnsembleMemberSpec, ...]

    def __post_init__(self) -> None:
        for field_name in ("name", "description"):
            text = getattr(self, field_name).strip()
            if not text:
                raise EnsembleError(f"{field_name} must not be blank")
            object.__setattr__(self, field_name, text)
        object.__setattr__(self, "method", EnsembleMethod(self.method))
        members = tuple(self.members)
        if len(members) < 2:
            raise EnsembleError("ensemble requires at least two calibrated detectors")
        if any(not isinstance(member, EnsembleMemberSpec) for member in members):
            raise TypeError("members must be EnsembleMemberSpec")
        member_names = tuple(member.member_name for member in members)
        if len(member_names) != len(set(member_names)):
            raise EnsembleError("ensemble member names must be unique")
        calibrations = tuple(member.calibration_sha256 for member in members)
        if len(calibrations) != len(set(calibrations)):
            raise EnsembleError(
                "ensemble members must reference distinct calibration artifacts"
            )
        object.__setattr__(self, "members", members)

        if self.method is EnsembleMethod.WEIGHTED_MEAN:
            weights = tuple(member.weight for member in members)
            if any(weight is None for weight in weights):
                raise EnsembleError(
                    "weighted_mean requires an explicit positive weight for every member"
                )
            total = sum((weight for weight in weights if weight is not None), Decimal(0))
            if total != Decimal(1):
                raise EnsembleError(
                    "weighted_mean member weights must sum exactly to Decimal(1)"
                )
        elif any(member.weight is not None for member in members):
            raise EnsembleError(
                "maximum/median ensemble members must not carry unused weights"
            )

    @property
    def sha256_hex(self) -> str:
        return _digest(self.as_dict(include_sha=False))

    def as_dict(self, *, include_sha: bool = True) -> dict[str, Any]:
        result = {
            "name": self.name,
            "description": self.description,
            "method": self.method.value,
            "members": [member.as_dict() for member in self.members],
        }
        if include_sha:
            result["sha256"] = self.sha256_hex
        return result


@dataclass(frozen=True, slots=True)
class EnsembleMemberBatchProvenance:
    member_name: str
    detector_name: str
    detector_family: str
    calibration_sha256: str
    source_score_batch_sha256: str
    calibrated_batch_sha256: str
    fitted_model_sha256: str
    training_matrix_sha256: str
    scoring_matrix_sha256: str
    preprocessor_sha256: str

    def __post_init__(self) -> None:
        for field_name in ("member_name", "detector_name", "detector_family"):
            text = getattr(self, field_name).strip()
            if not text:
                raise EnsembleError(f"{field_name} must not be blank")
            object.__setattr__(self, field_name, text)
        for field_name in (
            "calibration_sha256",
            "source_score_batch_sha256",
            "calibrated_batch_sha256",
            "fitted_model_sha256",
            "training_matrix_sha256",
            "scoring_matrix_sha256",
            "preprocessor_sha256",
        ):
            object.__setattr__(
                self,
                field_name,
                _digest_hex(getattr(self, field_name), field_name),
            )

    def as_dict(self) -> dict[str, str]:
        return {
            "member_name": self.member_name,
            "detector_name": self.detector_name,
            "detector_family": self.detector_family,
            "calibration_sha256": self.calibration_sha256,
            "source_score_batch_sha256": self.source_score_batch_sha256,
            "calibrated_batch_sha256": self.calibrated_batch_sha256,
            "fitted_model_sha256": self.fitted_model_sha256,
            "training_matrix_sha256": self.training_matrix_sha256,
            "scoring_matrix_sha256": self.scoring_matrix_sha256,
            "preprocessor_sha256": self.preprocessor_sha256,
        }


@dataclass(frozen=True, slots=True)
class EnsembleMemberScore:
    """One member's calibrated evidence retained inside an ensemble row."""

    member_name: str
    detector_name: str
    calibration_sha256: str
    empirical_lower_fraction: Decimal
    empirical_midpoint_fraction: Decimal
    empirical_upper_fraction: Decimal
    modified_z: Decimal | None
    modified_z_unavailable_reason: str | None
    iqr_distance: Decimal | None
    iqr_distance_unavailable_reason: str | None

    def __post_init__(self) -> None:
        for field_name in ("member_name", "detector_name"):
            text = getattr(self, field_name).strip()
            if not text:
                raise EnsembleError(f"{field_name} must not be blank")
            object.__setattr__(self, field_name, text)
        object.__setattr__(
            self,
            "calibration_sha256",
            _digest_hex(self.calibration_sha256, "calibration_sha256"),
        )
        lower = self.empirical_lower_fraction
        midpoint = self.empirical_midpoint_fraction
        upper = self.empirical_upper_fraction
        for field_name, value in (
            ("empirical_lower_fraction", lower),
            ("empirical_midpoint_fraction", midpoint),
            ("empirical_upper_fraction", upper),
        ):
            _fraction(value, field_name)
        if not lower <= midpoint <= upper:
            raise EnsembleError("member empirical fractions are inconsistent")
        _optional_measure(
            self.modified_z,
            self.modified_z_unavailable_reason,
            "modified_z",
        )
        _optional_measure(
            self.iqr_distance,
            self.iqr_distance_unavailable_reason,
            "iqr_distance",
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "member_name": self.member_name,
            "detector_name": self.detector_name,
            "calibration_sha256": self.calibration_sha256,
            "empirical_lower_fraction": str(self.empirical_lower_fraction),
            "empirical_midpoint_fraction": str(self.empirical_midpoint_fraction),
            "empirical_upper_fraction": str(self.empirical_upper_fraction),
            "modified_z": _decimal_text(self.modified_z),
            "modified_z_unavailable_reason": self.modified_z_unavailable_reason,
            "iqr_distance": _decimal_text(self.iqr_distance),
            "iqr_distance_unavailable_reason": self.iqr_distance_unavailable_reason,
        }


@dataclass(frozen=True, slots=True)
class EnsembleRowScore:
    transaction_id: str
    award_id: str
    ensemble_lower_fraction: Decimal
    ensemble_midpoint_fraction: Decimal
    ensemble_upper_fraction: Decimal
    member_midpoint_min: Decimal
    member_midpoint_max: Decimal
    member_disagreement_span: Decimal
    members: tuple[EnsembleMemberScore, ...]

    def __post_init__(self) -> None:
        for field_name in ("transaction_id", "award_id"):
            text = getattr(self, field_name).strip()
            if not text:
                raise EnsembleError(f"{field_name} must not be blank")
            object.__setattr__(self, field_name, text)
        lower = self.ensemble_lower_fraction
        midpoint = self.ensemble_midpoint_fraction
        upper = self.ensemble_upper_fraction
        minimum = self.member_midpoint_min
        maximum = self.member_midpoint_max
        span = self.member_disagreement_span
        for field_name, value in (
            ("ensemble_lower_fraction", lower),
            ("ensemble_midpoint_fraction", midpoint),
            ("ensemble_upper_fraction", upper),
            ("member_midpoint_min", minimum),
            ("member_midpoint_max", maximum),
            ("member_disagreement_span", span),
        ):
            _fraction(value, field_name)
        if not lower <= midpoint <= upper:
            raise EnsembleError("ensemble empirical interval is inconsistent")
        if minimum > maximum or span != maximum - minimum:
            raise EnsembleError("member midpoint disagreement evidence is inconsistent")

        members = tuple(self.members)
        if len(members) < 2:
            raise EnsembleError("ensemble row requires at least two member scores")
        member_names = tuple(member.member_name for member in members)
        if len(member_names) != len(set(member_names)):
            raise EnsembleError("ensemble row member names must be unique")
        observed_midpoints = tuple(
            member.empirical_midpoint_fraction for member in members
        )
        if min(observed_midpoints) != minimum or max(observed_midpoints) != maximum:
            raise EnsembleError(
                "member midpoint bounds differ from retained member evidence"
            )
        object.__setattr__(self, "members", members)

    @property
    def identity(self) -> tuple[str, str]:
        return self.transaction_id, self.award_id

    def as_dict(self) -> dict[str, Any]:
        return {
            "transaction_id": self.transaction_id,
            "award_id": self.award_id,
            "ensemble_lower_fraction": str(self.ensemble_lower_fraction),
            "ensemble_midpoint_fraction": str(self.ensemble_midpoint_fraction),
            "ensemble_upper_fraction": str(self.ensemble_upper_fraction),
            "member_midpoint_min": str(self.member_midpoint_min),
            "member_midpoint_max": str(self.member_midpoint_max),
            "member_disagreement_span": str(self.member_disagreement_span),
            "members": [member.as_dict() for member in self.members],
        }


@dataclass(frozen=True, slots=True)
class EnsembleScoreBatch:
    spec: EnsembleSpec
    member_provenance: tuple[EnsembleMemberBatchProvenance, ...]
    row_population_sha256: str
    rows: tuple[EnsembleRowScore, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.spec, EnsembleSpec):
            raise TypeError("spec must be EnsembleSpec")
        object.__setattr__(
            self,
            "row_population_sha256",
            _digest_hex(self.row_population_sha256, "row_population_sha256"),
        )
        provenance = tuple(self.member_provenance)
        rows = tuple(self.rows)
        if not rows:
            raise EnsembleError("ensemble score batch requires rows")
        if len(provenance) != len(self.spec.members):
            raise EnsembleError("member provenance count differs from ensemble spec")

        spec_names = tuple(member.member_name for member in self.spec.members)
        if tuple(item.member_name for item in provenance) != spec_names:
            raise EnsembleError(
                "member provenance must follow ensemble specification order"
            )
        expected_calibrations = tuple(
            member.calibration_sha256 for member in self.spec.members
        )
        if tuple(item.calibration_sha256 for item in provenance) != expected_calibrations:
            raise EnsembleError(
                "member provenance calibration fingerprints differ from ensemble spec"
            )
        if tuple(item.detector_name for item in provenance) != tuple(
            member.detector_name for member in self.spec.members
        ):
            raise EnsembleError(
                "member provenance detector names differ from ensemble spec"
            )

        identities = tuple(row.identity for row in rows)
        if len(identities) != len(set(identities)):
            raise EnsembleError("ensemble rows contain duplicate identities")
        if _population_digest(identities) != self.row_population_sha256:
            raise EnsembleError("row_population_sha256 differs from ensemble rows")

        for row in rows:
            if tuple(member.member_name for member in row.members) != spec_names:
                raise EnsembleError(
                    "ensemble row member order differs from ensemble specification"
                )
            _validate_row_combination(self.spec, row)

        object.__setattr__(self, "member_provenance", provenance)
        object.__setattr__(self, "rows", rows)

    @property
    def row_count(self) -> int:
        return len(self.rows)

    @property
    def row_identities(self) -> tuple[tuple[str, str], ...]:
        return tuple(row.identity for row in self.rows)

    @property
    def evidence_sha256(self) -> str:
        return _digest(
            {
                "ensemble_spec_sha256": self.spec.sha256_hex,
                "member_provenance": [
                    item.as_dict() for item in self.member_provenance
                ],
                "row_population_sha256": self.row_population_sha256,
                "rows": [row.as_dict() for row in self.rows],
            }
        )


def combine_calibrated_scores(
    spec: EnsembleSpec,
    batches: Iterable[CalibratedScoreBatch],
) -> EnsembleScoreBatch:
    """Combine exact calibrated detector artifacts over one identical row population."""

    if not isinstance(spec, EnsembleSpec):
        raise TypeError("spec must be EnsembleSpec")
    if isinstance(batches, (str, bytes)):
        raise EnsembleError("batches must be an iterable of CalibratedScoreBatch")
    supplied = tuple(batches)
    if any(not isinstance(batch, CalibratedScoreBatch) for batch in supplied):
        raise TypeError("all batches must be CalibratedScoreBatch")
    if len(supplied) != len(spec.members):
        raise EnsembleError(
            "supplied calibrated batch count differs from ensemble specification"
        )

    by_calibration: dict[str, CalibratedScoreBatch] = {}
    for batch in supplied:
        if batch.calibration_sha256 in by_calibration:
            raise EnsembleError("duplicate calibrated batch fingerprint supplied")
        by_calibration[batch.calibration_sha256] = batch

    ordered: list[CalibratedScoreBatch] = []
    provenance: list[EnsembleMemberBatchProvenance] = []
    for member in spec.members:
        batch = by_calibration.get(member.calibration_sha256)
        if batch is None:
            raise EnsembleError(
                f"missing calibrated batch for member {member.member_name!r}"
            )
        if batch.detector_name != member.detector_name:
            raise EnsembleError(
                f"{member.member_name}: detector name differs from ensemble spec"
            )
        ordered.append(batch)
        provenance.append(
            EnsembleMemberBatchProvenance(
                member_name=member.member_name,
                detector_name=batch.detector_name,
                detector_family=batch.detector_family,
                calibration_sha256=batch.calibration_sha256,
                source_score_batch_sha256=batch.source_score_batch_sha256,
                calibrated_batch_sha256=batch.evidence_sha256,
                fitted_model_sha256=batch.fitted_model_sha256,
                training_matrix_sha256=batch.training_matrix_sha256,
                scoring_matrix_sha256=batch.scoring_matrix_sha256,
                preprocessor_sha256=batch.preprocessor_sha256,
            )
        )

    identities = ordered[0].row_identities
    if any(batch.row_identities != identities for batch in ordered[1:]):
        raise EnsembleError(
            "calibrated detector batches must contain the exact same ordered row population"
        )

    rows: list[EnsembleRowScore] = []
    for row_index, identity in enumerate(identities):
        member_scores: list[EnsembleMemberScore] = []
        lower_values: list[Decimal] = []
        midpoint_values: list[Decimal] = []
        upper_values: list[Decimal] = []

        for member, batch in zip(spec.members, ordered):
            source = batch.scores[row_index]
            retained = _member_score(member, source)
            member_scores.append(retained)
            lower_values.append(retained.empirical_lower_fraction)
            midpoint_values.append(retained.empirical_midpoint_fraction)
            upper_values.append(retained.empirical_upper_fraction)

        lower = _combine(spec, lower_values)
        midpoint = _combine(spec, midpoint_values)
        upper = _combine(spec, upper_values)
        minimum = min(midpoint_values)
        maximum = max(midpoint_values)

        rows.append(
            EnsembleRowScore(
                transaction_id=identity[0],
                award_id=identity[1],
                ensemble_lower_fraction=lower,
                ensemble_midpoint_fraction=midpoint,
                ensemble_upper_fraction=upper,
                member_midpoint_min=minimum,
                member_midpoint_max=maximum,
                member_disagreement_span=maximum - minimum,
                members=tuple(member_scores),
            )
        )

    return EnsembleScoreBatch(
        spec=spec,
        member_provenance=tuple(provenance),
        row_population_sha256=_population_digest(identities),
        rows=tuple(rows),
    )


def _member_score(
    member: EnsembleMemberSpec,
    source: CalibratedDetectorScore,
) -> EnsembleMemberScore:
    return EnsembleMemberScore(
        member_name=member.member_name,
        detector_name=member.detector_name,
        calibration_sha256=member.calibration_sha256,
        empirical_lower_fraction=source.empirical_lower_fraction,
        empirical_midpoint_fraction=source.empirical_midpoint_fraction,
        empirical_upper_fraction=source.empirical_upper_fraction,
        modified_z=source.modified_z,
        modified_z_unavailable_reason=source.modified_z_unavailable_reason,
        iqr_distance=source.iqr_distance,
        iqr_distance_unavailable_reason=source.iqr_distance_unavailable_reason,
    )


def _combine(spec: EnsembleSpec, values: list[Decimal]) -> Decimal:
    if len(values) != len(spec.members):
        raise EnsembleError("combination value count differs from ensemble spec")
    if spec.method is EnsembleMethod.WEIGHTED_MEAN:
        total = Decimal(0)
        for member, value in zip(spec.members, values):
            if member.weight is None:
                raise AssertionError("validated weighted member missing weight")
            total += member.weight * value
        _fraction(total, "weighted ensemble result")
        return total
    if spec.method is EnsembleMethod.MAXIMUM:
        return max(values)
    if spec.method is EnsembleMethod.MEDIAN:
        ordered = sorted(values)
        middle = len(ordered) // 2
        if len(ordered) % 2:
            return ordered[middle]
        return (ordered[middle - 1] + ordered[middle]) / Decimal(2)
    raise AssertionError(f"unsupported ensemble method: {spec.method!r}")


def _validate_row_combination(spec: EnsembleSpec, row: EnsembleRowScore) -> None:
    lower = _combine(
        spec, [member.empirical_lower_fraction for member in row.members]
    )
    midpoint = _combine(
        spec, [member.empirical_midpoint_fraction for member in row.members]
    )
    upper = _combine(
        spec, [member.empirical_upper_fraction for member in row.members]
    )
    if (
        row.ensemble_lower_fraction != lower
        or row.ensemble_midpoint_fraction != midpoint
        or row.ensemble_upper_fraction != upper
    ):
        raise EnsembleError("ensemble row scores differ from specification combination")


def _population_digest(identities: Iterable[tuple[str, str]]) -> str:
    normalized: list[tuple[str, str]] = []
    for identity in identities:
        if not isinstance(identity, tuple) or len(identity) != 2:
            raise EnsembleError(
                "row identities must be (transaction_id, award_id) tuples"
            )
        txid, award_id = identity[0].strip(), identity[1].strip()
        if not txid or not award_id:
            raise EnsembleError("row identity values must not be blank")
        normalized.append((txid, award_id))
    if not normalized:
        raise EnsembleError("row population must not be empty")
    if len(normalized) != len(set(normalized)):
        raise EnsembleError("row population contains duplicate identities")
    return _digest(normalized)


def _optional_measure(
    value: Decimal | None,
    reason: str | None,
    name: str,
) -> None:
    if value is None:
        text = None if reason is None else reason.strip()
        if not text:
            raise EnsembleError(f"{name}: unavailable value requires a reason")
        return
    if not isinstance(value, Decimal) or not value.is_finite():
        raise EnsembleError(f"{name}: value must be finite Decimal or None")
    if reason is not None:
        raise EnsembleError(f"{name}: available value cannot carry unavailable reason")


def _fraction(value: Decimal, name: str) -> None:
    if (
        not isinstance(value, Decimal)
        or not value.is_finite()
        or value < 0
        or value > 1
    ):
        raise EnsembleError(f"{name} must be a finite Decimal in [0, 1]")


def _decimal_text(value: Decimal | None) -> str | None:
    return None if value is None else str(value)


def _digest_hex(value: str, name: str) -> str:
    digest = value.strip().lower()
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise EnsembleError(f"{name} must be a SHA-256 hex digest")
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
