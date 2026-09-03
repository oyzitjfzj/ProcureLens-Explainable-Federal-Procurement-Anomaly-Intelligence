"""Investigator-facing anomaly evidence binding for ProcureLens.

Binds canonical feature-row provenance, calibrated ensemble evidence, and optional
run-to-run stability for the exact same scored population. The result is a review
handoff contract: it preserves what the system observed and how uncertain/stable
the anomaly ranking was without converting those facts into a fraud conclusion,
review threshold, or 0-100 policy score.

Optional stability evidence is accepted only when the current ensemble artifact
is one of the analyzed runs and the row population matches exactly.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from decimal import Decimal
from hashlib import sha256
import json
from typing import Any

from procurelens.detectors.ensemble import EnsembleScoreBatch
from procurelens.detectors.stability import StabilityReport
from procurelens.model.feature_row import CandidateFeatureRow


class ReviewEvidenceError(ValueError):
    """Raised when review-evidence provenance is inconsistent."""


@dataclass(frozen=True, slots=True)
class ReviewEvidenceRow:
    transaction_id: str
    award_id: str
    feature_row_evidence_sha256: str
    feature_catalog_sha256: str
    available_feature_count: int
    missing_feature_count: int

    ensemble_batch_sha256: str
    ensemble_spec_sha256: str
    ensemble_lower_fraction: Decimal
    ensemble_midpoint_fraction: Decimal
    ensemble_upper_fraction: Decimal
    detector_disagreement_span: Decimal

    stability_report_sha256: str | None
    stability_run_count: int | None
    stability_minimum_position: Decimal | None
    stability_median_position: Decimal | None
    stability_maximum_position: Decimal | None
    stability_position_span: Decimal | None
    stability_median_absolute_deviation: Decimal | None

    def __post_init__(self) -> None:
        for field_name in ("transaction_id", "award_id"):
            text = getattr(self, field_name).strip()
            if not text:
                raise ReviewEvidenceError(f"{field_name} must not be blank")
            object.__setattr__(self, field_name, text)

        for field_name in (
            "feature_row_evidence_sha256",
            "feature_catalog_sha256",
            "ensemble_batch_sha256",
            "ensemble_spec_sha256",
        ):
            object.__setattr__(
                self,
                field_name,
                _digest_hex(getattr(self, field_name), field_name),
            )

        for field_name in ("available_feature_count", "missing_feature_count"):
            value = getattr(self, field_name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
            ):
                raise ReviewEvidenceError(
                    f"{field_name} must be a non-negative integer"
                )
        if self.available_feature_count + self.missing_feature_count < 1:
            raise ReviewEvidenceError(
                "review evidence row requires at least one candidate feature"
            )

        lower = self.ensemble_lower_fraction
        midpoint = self.ensemble_midpoint_fraction
        upper = self.ensemble_upper_fraction
        disagreement = self.detector_disagreement_span
        for field_name, value in (
            ("ensemble_lower_fraction", lower),
            ("ensemble_midpoint_fraction", midpoint),
            ("ensemble_upper_fraction", upper),
            ("detector_disagreement_span", disagreement),
        ):
            _fraction(value, field_name)
        if not lower <= midpoint <= upper:
            raise ReviewEvidenceError(
                "ensemble empirical interval is inconsistent"
            )

        stability_values = (
            self.stability_minimum_position,
            self.stability_median_position,
            self.stability_maximum_position,
            self.stability_position_span,
            self.stability_median_absolute_deviation,
        )
        if self.stability_report_sha256 is None:
            if self.stability_run_count is not None or any(
                value is not None for value in stability_values
            ):
                raise ReviewEvidenceError(
                    "row without stability report cannot carry stability measurements"
                )
        else:
            object.__setattr__(
                self,
                "stability_report_sha256",
                _digest_hex(
                    self.stability_report_sha256,
                    "stability_report_sha256",
                ),
            )
            if (
                isinstance(self.stability_run_count, bool)
                or not isinstance(self.stability_run_count, int)
                or self.stability_run_count < 2
            ):
                raise ReviewEvidenceError(
                    "stability_run_count must be an integer of at least two"
                )
            if any(value is None for value in stability_values):
                raise ReviewEvidenceError(
                    "stability report requires complete row stability measurements"
                )
            minimum, median, maximum, span, mad = stability_values
            assert minimum is not None
            assert median is not None
            assert maximum is not None
            assert span is not None
            assert mad is not None
            for field_name, value in (
                ("stability_minimum_position", minimum),
                ("stability_median_position", median),
                ("stability_maximum_position", maximum),
                ("stability_position_span", span),
                ("stability_median_absolute_deviation", mad),
            ):
                _fraction(value, field_name)
            if not minimum <= median <= maximum:
                raise ReviewEvidenceError(
                    "stability position bounds are inconsistent"
                )
            if span != maximum - minimum:
                raise ReviewEvidenceError(
                    "stability position span is inconsistent"
                )

    @property
    def identity(self) -> tuple[str, str]:
        return self.transaction_id, self.award_id

    @property
    def stability_available(self) -> bool:
        return self.stability_report_sha256 is not None

    @property
    def evidence_sha256(self) -> str:
        return _digest(self.as_dict(include_sha=False))

    def as_dict(self, *, include_sha: bool = True) -> dict[str, Any]:
        result = {
            "transaction_id": self.transaction_id,
            "award_id": self.award_id,
            "feature_row_evidence_sha256": self.feature_row_evidence_sha256,
            "feature_catalog_sha256": self.feature_catalog_sha256,
            "available_feature_count": self.available_feature_count,
            "missing_feature_count": self.missing_feature_count,
            "ensemble_batch_sha256": self.ensemble_batch_sha256,
            "ensemble_spec_sha256": self.ensemble_spec_sha256,
            "ensemble_lower_fraction": str(self.ensemble_lower_fraction),
            "ensemble_midpoint_fraction": str(self.ensemble_midpoint_fraction),
            "ensemble_upper_fraction": str(self.ensemble_upper_fraction),
            "detector_disagreement_span": str(
                self.detector_disagreement_span
            ),
            "stability_report_sha256": self.stability_report_sha256,
            "stability_run_count": self.stability_run_count,
            "stability_minimum_position": _decimal_text(
                self.stability_minimum_position
            ),
            "stability_median_position": _decimal_text(
                self.stability_median_position
            ),
            "stability_maximum_position": _decimal_text(
                self.stability_maximum_position
            ),
            "stability_position_span": _decimal_text(
                self.stability_position_span
            ),
            "stability_median_absolute_deviation": _decimal_text(
                self.stability_median_absolute_deviation
            ),
        }
        if include_sha:
            result["evidence_sha256"] = self.evidence_sha256
        return result


@dataclass(frozen=True, slots=True)
class ReviewEvidenceBatch:
    ensemble_batch_sha256: str
    ensemble_spec_sha256: str
    feature_catalog_sha256: str
    row_population_sha256: str
    stability_report_sha256: str | None
    rows: tuple[ReviewEvidenceRow, ...]

    def __post_init__(self) -> None:
        for field_name in (
            "ensemble_batch_sha256",
            "ensemble_spec_sha256",
            "feature_catalog_sha256",
            "row_population_sha256",
        ):
            object.__setattr__(
                self,
                field_name,
                _digest_hex(getattr(self, field_name), field_name),
            )
        if self.stability_report_sha256 is not None:
            object.__setattr__(
                self,
                "stability_report_sha256",
                _digest_hex(
                    self.stability_report_sha256,
                    "stability_report_sha256",
                ),
            )

        rows = tuple(self.rows)
        if not rows:
            raise ReviewEvidenceError(
                "review evidence batch requires at least one row"
            )
        identities = tuple(row.identity for row in rows)
        if len(identities) != len(set(identities)):
            raise ReviewEvidenceError(
                "review evidence rows contain duplicate identities"
            )
        if _population_digest(identities) != self.row_population_sha256:
            raise ReviewEvidenceError(
                "row_population_sha256 differs from review evidence rows"
            )
        if any(
            row.ensemble_batch_sha256 != self.ensemble_batch_sha256
            or row.ensemble_spec_sha256 != self.ensemble_spec_sha256
            or row.feature_catalog_sha256 != self.feature_catalog_sha256
            or row.stability_report_sha256 != self.stability_report_sha256
            for row in rows
        ):
            raise ReviewEvidenceError(
                "review evidence row provenance differs from batch provenance"
            )
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
                "ensemble_batch_sha256": self.ensemble_batch_sha256,
                "ensemble_spec_sha256": self.ensemble_spec_sha256,
                "feature_catalog_sha256": self.feature_catalog_sha256,
                "row_population_sha256": self.row_population_sha256,
                "stability_report_sha256": self.stability_report_sha256,
                "rows": [row.evidence_sha256 for row in self.rows],
            }
        )


def build_review_evidence_batch(
    feature_rows: Iterable[CandidateFeatureRow],
    ensemble: EnsembleScoreBatch,
    *,
    stability: StabilityReport | None = None,
) -> ReviewEvidenceBatch:
    """Bind features, ensemble uncertainty, and optional matching stability evidence."""

    if isinstance(feature_rows, (str, bytes)):
        raise ReviewEvidenceError(
            "feature_rows must be an iterable of CandidateFeatureRow"
        )
    features = tuple(feature_rows)
    if not features:
        raise ReviewEvidenceError(
            "at least one candidate feature row is required"
        )
    if any(not isinstance(row, CandidateFeatureRow) for row in features):
        raise TypeError("all feature_rows must be CandidateFeatureRow")
    if not isinstance(ensemble, EnsembleScoreBatch):
        raise TypeError("ensemble must be EnsembleScoreBatch")
    if stability is not None and not isinstance(stability, StabilityReport):
        raise TypeError("stability must be StabilityReport or None")

    feature_identities = tuple(
        (row.transaction_id, row.award_id) for row in features
    )
    if len(feature_identities) != len(set(feature_identities)):
        raise ReviewEvidenceError(
            "candidate feature rows contain duplicate identities"
        )
    if feature_identities != ensemble.row_identities:
        raise ReviewEvidenceError(
            "feature rows must match the exact ordered ensemble row population"
        )

    catalog_hashes = {row.catalog_sha256 for row in features}
    if len(catalog_hashes) != 1:
        raise ReviewEvidenceError(
            "candidate feature rows use different catalog fingerprints"
        )
    catalog_sha = next(iter(catalog_hashes))

    stability_by_identity: dict[
        tuple[str, str], Any
    ] = {}
    stability_sha: str | None = None
    stability_run_count: int | None = None
    if stability is not None:
        if tuple(row.identity for row in stability.rows) != ensemble.row_identities:
            raise ReviewEvidenceError(
                "stability report row population differs from ensemble population"
            )
        if ensemble.evidence_sha256 not in {
            run.source_evidence_sha256 for run in stability.runs
        }:
            raise ReviewEvidenceError(
                "stability report does not include the current ensemble artifact"
            )
        stability_sha = stability.evidence_sha256
        stability_run_count = len(stability.runs)
        stability_by_identity = {row.identity: row for row in stability.rows}

    ensemble_sha = ensemble.evidence_sha256
    spec_sha = ensemble.spec.sha256_hex
    rows: list[ReviewEvidenceRow] = []

    for feature_row, ensemble_row in zip(features, ensemble.rows):
        identity = (feature_row.transaction_id, feature_row.award_id)
        if ensemble_row.identity != identity:
            raise ReviewEvidenceError(
                "feature and ensemble row identities differ"
            )
        stable_row = stability_by_identity.get(identity)
        rows.append(
            ReviewEvidenceRow(
                transaction_id=identity[0],
                award_id=identity[1],
                feature_row_evidence_sha256=feature_row.evidence_sha256,
                feature_catalog_sha256=feature_row.catalog_sha256,
                available_feature_count=feature_row.available_count,
                missing_feature_count=feature_row.missing_count,
                ensemble_batch_sha256=ensemble_sha,
                ensemble_spec_sha256=spec_sha,
                ensemble_lower_fraction=ensemble_row.ensemble_lower_fraction,
                ensemble_midpoint_fraction=(
                    ensemble_row.ensemble_midpoint_fraction
                ),
                ensemble_upper_fraction=ensemble_row.ensemble_upper_fraction,
                detector_disagreement_span=(
                    ensemble_row.member_disagreement_span
                ),
                stability_report_sha256=stability_sha,
                stability_run_count=stability_run_count,
                stability_minimum_position=(
                    None if stable_row is None
                    else stable_row.minimum_position
                ),
                stability_median_position=(
                    None if stable_row is None
                    else stable_row.median_position
                ),
                stability_maximum_position=(
                    None if stable_row is None
                    else stable_row.maximum_position
                ),
                stability_position_span=(
                    None if stable_row is None
                    else stable_row.position_span
                ),
                stability_median_absolute_deviation=(
                    None if stable_row is None
                    else stable_row.median_absolute_deviation
                ),
            )
        )

    return ReviewEvidenceBatch(
        ensemble_batch_sha256=ensemble_sha,
        ensemble_spec_sha256=spec_sha,
        feature_catalog_sha256=catalog_sha,
        row_population_sha256=_population_digest(feature_identities),
        stability_report_sha256=stability_sha,
        rows=tuple(rows),
    )


def _population_digest(
    identities: Iterable[tuple[str, str]],
) -> str:
    normalized: list[tuple[str, str]] = []
    for identity in identities:
        if not isinstance(identity, tuple) or len(identity) != 2:
            raise ReviewEvidenceError(
                "row identities must be (transaction_id, award_id) tuples"
            )
        txid, award_id = identity[0].strip(), identity[1].strip()
        if not txid or not award_id:
            raise ReviewEvidenceError(
                "row identity values must not be blank"
            )
        normalized.append((txid, award_id))
    result = tuple(normalized)
    if not result or len(result) != len(set(result)):
        raise ReviewEvidenceError(
            "row identities must be non-empty and unique"
        )
    return _digest(result)


def _fraction(value: Decimal, name: str) -> None:
    if (
        not isinstance(value, Decimal)
        or not value.is_finite()
        or value < 0
        or value > 1
    ):
        raise ReviewEvidenceError(
            f"{name} must be finite Decimal in [0, 1]"
        )


def _decimal_text(value: Decimal | None) -> str | None:
    return None if value is None else str(value)


def _digest_hex(value: str, name: str) -> str:
    digest = value.strip().lower()
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ReviewEvidenceError(
            f"{name} must be a SHA-256 hex digest"
        )
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
