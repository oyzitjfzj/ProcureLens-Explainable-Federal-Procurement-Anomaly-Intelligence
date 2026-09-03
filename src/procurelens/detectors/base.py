"""Shared anomaly-detector evidence contracts for ProcureLens.

Detector adapters preserve their native numeric score while exposing a single
monotonic orientation where larger values mean "more anomalous". No score-range
normalization, contamination threshold, review flag, risk score, or misconduct
conclusion lives here.

Every score batch pins algorithm/config/model and matrix provenance so later
ensembles cannot silently mix scores from incompatible detector runs.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
import json
import math
from typing import Any, Iterable


class DetectorContractError(ValueError):
    """Raised when detector score evidence or provenance is inconsistent."""


class ScoreOrientation(str, Enum):
    """Native score direction relative to abnormality."""

    LOWER_IS_MORE_ANOMALOUS = "lower_is_more_anomalous"
    HIGHER_IS_MORE_ANOMALOUS = "higher_is_more_anomalous"


@dataclass(frozen=True, slots=True)
class DetectorScore:
    """One row's native detector score plus common anomaly orientation."""

    transaction_id: str
    award_id: str
    raw_score: float
    anomaly_score: float

    def __post_init__(self) -> None:
        for field_name in ("transaction_id", "award_id"):
            text = getattr(self, field_name).strip()
            if not text:
                raise DetectorContractError(f"{field_name} must not be blank")
            object.__setattr__(self, field_name, text)
        for field_name in ("raw_score", "anomaly_score"):
            value = getattr(self, field_name)
            if not isinstance(value, float) or not math.isfinite(value):
                raise DetectorContractError(
                    f"{field_name} must be a finite float"
                )

    @property
    def identity(self) -> tuple[str, str]:
        return self.transaction_id, self.award_id

    def as_dict(self) -> dict[str, Any]:
        return {
            "transaction_id": self.transaction_id,
            "award_id": self.award_id,
            "raw_score_hex": self.raw_score.hex(),
            "anomaly_score_hex": self.anomaly_score.hex(),
        }


@dataclass(frozen=True, slots=True)
class DetectorScoreBatch:
    """Immutable scored population from one fitted detector artifact."""

    detector_name: str
    detector_family: str
    implementation_name: str
    implementation_version: str
    score_orientation: ScoreOrientation
    config_sha256: str
    fitted_model_sha256: str
    training_matrix_sha256: str
    scoring_matrix_sha256: str
    preprocessor_sha256: str
    output_feature_names: tuple[str, ...]
    scores: tuple[DetectorScore, ...]

    def __post_init__(self) -> None:
        for field_name in (
            "detector_name",
            "detector_family",
            "implementation_name",
            "implementation_version",
        ):
            text = getattr(self, field_name).strip()
            if not text:
                raise DetectorContractError(f"{field_name} must not be blank")
            object.__setattr__(self, field_name, text)
        object.__setattr__(
            self, "score_orientation", ScoreOrientation(self.score_orientation)
        )
        for field_name in (
            "config_sha256",
            "fitted_model_sha256",
            "training_matrix_sha256",
            "scoring_matrix_sha256",
            "preprocessor_sha256",
        ):
            object.__setattr__(
                self,
                field_name,
                _validate_digest(getattr(self, field_name), field_name),
            )

        names = tuple(name.strip() for name in self.output_feature_names)
        if not names or any(not name for name in names):
            raise DetectorContractError(
                "output_feature_names must be non-empty and non-blank"
            )
        if len(names) != len(set(names)):
            raise DetectorContractError(
                "output_feature_names must be globally unique"
            )
        object.__setattr__(self, "output_feature_names", names)

        scores = tuple(self.scores)
        if not scores:
            raise DetectorContractError(
                "detector score batch requires at least one score"
            )
        identities = tuple(score.identity for score in scores)
        if len(identities) != len(set(identities)):
            raise DetectorContractError(
                "detector score batch contains duplicate row identities"
            )
        for score in scores:
            expected = orient_score(score.raw_score, self.score_orientation)
            if score.anomaly_score != expected:
                raise DetectorContractError(
                    "anomaly_score is inconsistent with raw score orientation"
                )
        object.__setattr__(self, "scores", scores)

    @property
    def row_count(self) -> int:
        return len(self.scores)

    @property
    def row_identities(self) -> tuple[tuple[str, str], ...]:
        return tuple(score.identity for score in self.scores)

    @property
    def evidence_sha256(self) -> str:
        return _digest(
            {
                "detector_name": self.detector_name,
                "detector_family": self.detector_family,
                "implementation_name": self.implementation_name,
                "implementation_version": self.implementation_version,
                "score_orientation": self.score_orientation.value,
                "config_sha256": self.config_sha256,
                "fitted_model_sha256": self.fitted_model_sha256,
                "training_matrix_sha256": self.training_matrix_sha256,
                "scoring_matrix_sha256": self.scoring_matrix_sha256,
                "preprocessor_sha256": self.preprocessor_sha256,
                "output_feature_names": list(self.output_feature_names),
                "scores": [score.as_dict() for score in self.scores],
            }
        )


def orient_score(
    raw_score: float,
    orientation: ScoreOrientation,
) -> float:
    """Return a monotonic score where larger always means more anomalous."""

    if not isinstance(raw_score, float) or not math.isfinite(raw_score):
        raise DetectorContractError("raw_score must be a finite float")
    orientation = ScoreOrientation(orientation)
    return (
        -raw_score
        if orientation is ScoreOrientation.LOWER_IS_MORE_ANOMALOUS
        else raw_score
    )


def build_detector_scores(
    row_identities: Iterable[tuple[str, str]],
    raw_scores: Iterable[float],
    *,
    orientation: ScoreOrientation,
) -> tuple[DetectorScore, ...]:
    """Bind native scores to row identities without thresholding them."""

    identities = tuple(row_identities)
    values = tuple(raw_scores)
    if len(identities) != len(values):
        raise DetectorContractError(
            "row identity and detector-score lengths differ"
        )
    if not identities:
        raise DetectorContractError("at least one detector score is required")

    result: list[DetectorScore] = []
    seen: set[tuple[str, str]] = set()
    for identity, raw in zip(identities, values):
        if not isinstance(identity, tuple) or len(identity) != 2:
            raise DetectorContractError(
                "row identities must be (transaction_id, award_id) tuples"
            )
        txid, award_id = identity[0].strip(), identity[1].strip()
        if not txid or not award_id:
            raise DetectorContractError(
                "row identity values must not be blank"
            )
        normalized = (txid, award_id)
        if normalized in seen:
            raise DetectorContractError(
                f"duplicate detector row identity: {normalized!r}"
            )
        seen.add(normalized)
        if not isinstance(raw, float) or not math.isfinite(raw):
            raise DetectorContractError(
                "detector raw scores must be finite floats"
            )
        result.append(
            DetectorScore(
                transaction_id=txid,
                award_id=award_id,
                raw_score=raw,
                anomaly_score=orient_score(raw, orientation),
            )
        )
    return tuple(result)


def _validate_digest(value: str, name: str) -> str:
    digest = value.strip().lower()
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise DetectorContractError(f"{name} must be a SHA-256 hex digest")
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
