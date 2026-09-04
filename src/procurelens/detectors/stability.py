"""Run-to-run stability evidence for ProcureLens anomaly rankings.

Measures whether repeated calibrated detector or ensemble runs produce consistent
relative anomaly positions over the exact same row population. Stability is
descriptive evidence only: this module does not define an acceptable correlation,
minimum number of runs, top-k cutoff, review threshold, or risk decision.

Pairwise rank agreement uses tie-aware average ranks. Per-row evidence preserves
the observed score range and median absolute deviation across runs so instability
can be inspected without collapsing it into one pass/fail label.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, localcontext
from enum import Enum
from hashlib import sha256
import json
from typing import Any, Iterable

from procurelens.detectors.calibration import CalibratedScoreBatch
from procurelens.detectors.ensemble import EnsembleScoreBatch


class StabilityError(ValueError):
    """Raised when repeated-run stability evidence is inconsistent."""


DECIMAL_WORKING_PRECISION = 50


class StabilitySourceKind(str, Enum):
    CALIBRATED_DETECTOR = "calibrated_detector"
    ENSEMBLE = "ensemble"


@dataclass(frozen=True, slots=True)
class StabilityRun:
    run_name: str
    source_kind: StabilitySourceKind
    source_evidence_sha256: str
    row_identities: tuple[tuple[str, str], ...]
    anomaly_positions: tuple[Decimal, ...]

    def __post_init__(self) -> None:
        name = self.run_name.strip()
        if not name:
            raise StabilityError("run_name must not be blank")
        object.__setattr__(self, "run_name", name)
        object.__setattr__(self, "source_kind", StabilitySourceKind(self.source_kind))
        object.__setattr__(
            self,
            "source_evidence_sha256",
            _digest_hex(self.source_evidence_sha256, "source_evidence_sha256"),
        )
        identities = _identities(self.row_identities)
        positions = tuple(self.anomaly_positions)
        if len(positions) != len(identities):
            raise StabilityError("run position count differs from row population")
        if any(
            not isinstance(value, Decimal) or not value.is_finite()
            for value in positions
        ):
            raise StabilityError("run positions must be finite Decimal values")
        for value in positions:
            _fraction(value, "anomaly position")
        object.__setattr__(self, "row_identities", identities)
        object.__setattr__(self, "anomaly_positions", positions)

    @property
    def sha256_hex(self) -> str:
        return _digest(
            {
                "run_name": self.run_name,
                "source_kind": self.source_kind.value,
                "source_evidence_sha256": self.source_evidence_sha256,
                "row_identities": self.row_identities,
                "anomaly_positions": [str(value) for value in self.anomaly_positions],
            }
        )


@dataclass(frozen=True, slots=True)
class PairwiseRankAgreement:
    left_run_name: str
    right_run_name: str
    spearman_rho: Decimal | None
    unavailable_reason: str | None

    def __post_init__(self) -> None:
        left, right = self.left_run_name.strip(), self.right_run_name.strip()
        if not left or not right or left == right:
            raise StabilityError("rank agreement requires two distinct run names")
        object.__setattr__(self, "left_run_name", left)
        object.__setattr__(self, "right_run_name", right)
        if self.spearman_rho is None:
            reason = (
                None
                if self.unavailable_reason is None
                else self.unavailable_reason.strip()
            )
            if not reason:
                raise StabilityError(
                    "unavailable rank agreement requires a reason"
                )
            object.__setattr__(self, "unavailable_reason", reason)
        else:
            if (
                not isinstance(self.spearman_rho, Decimal)
                or not self.spearman_rho.is_finite()
                or self.spearman_rho < Decimal(-1)
                or self.spearman_rho > Decimal(1)
            ):
                raise StabilityError(
                    "spearman_rho must be finite Decimal in [-1, 1]"
                )
            if self.unavailable_reason is not None:
                raise StabilityError(
                    "available rank agreement cannot carry unavailable reason"
                )

    def as_dict(self) -> dict[str, Any]:
        return {
            "left_run_name": self.left_run_name,
            "right_run_name": self.right_run_name,
            "spearman_rho": (
                None if self.spearman_rho is None else str(self.spearman_rho)
            ),
            "unavailable_reason": self.unavailable_reason,
        }


@dataclass(frozen=True, slots=True)
class RowStabilityEvidence:
    transaction_id: str
    award_id: str
    minimum_position: Decimal
    median_position: Decimal
    maximum_position: Decimal
    position_span: Decimal
    median_absolute_deviation: Decimal
    run_positions: tuple[tuple[str, Decimal], ...]

    def __post_init__(self) -> None:
        for field_name in ("transaction_id", "award_id"):
            text = getattr(self, field_name).strip()
            if not text:
                raise StabilityError(f"{field_name} must not be blank")
            object.__setattr__(self, field_name, text)

        for field_name in (
            "minimum_position",
            "median_position",
            "maximum_position",
            "position_span",
            "median_absolute_deviation",
        ):
            value = getattr(self, field_name)
            _fraction(value, field_name)
        if not (
            self.minimum_position
            <= self.median_position
            <= self.maximum_position
        ):
            raise StabilityError("row stability bounds are inconsistent")
        if self.position_span != self.maximum_position - self.minimum_position:
            raise StabilityError("row position span is inconsistent")

        positions = tuple(self.run_positions)
        if len(positions) < 2:
            raise StabilityError(
                "row stability evidence requires at least two runs"
            )
        names = tuple(name.strip() for name, _ in positions)
        if any(not name for name in names) or len(names) != len(set(names)):
            raise StabilityError(
                "row stability run names must be non-blank and unique"
            )
        values = tuple(value for _, value in positions)
        for value in values:
            _fraction(value, "run position")
        if (
            min(values) != self.minimum_position
            or max(values) != self.maximum_position
            or _median(values) != self.median_position
            or _median(
                tuple(abs(value - self.median_position) for value in values)
            )
            != self.median_absolute_deviation
        ):
            raise StabilityError(
                "row stability summary differs from retained run positions"
            )
        object.__setattr__(
            self,
            "run_positions",
            tuple(zip(names, values)),
        )

    @property
    def identity(self) -> tuple[str, str]:
        return self.transaction_id, self.award_id

    def as_dict(self) -> dict[str, Any]:
        return {
            "transaction_id": self.transaction_id,
            "award_id": self.award_id,
            "minimum_position": str(self.minimum_position),
            "median_position": str(self.median_position),
            "maximum_position": str(self.maximum_position),
            "position_span": str(self.position_span),
            "median_absolute_deviation": str(
                self.median_absolute_deviation
            ),
            "run_positions": [
                (name, str(value)) for name, value in self.run_positions
            ],
        }


@dataclass(frozen=True, slots=True)
class StabilityReport:
    runs: tuple[StabilityRun, ...]
    row_population_sha256: str
    pairwise_rank_agreements: tuple[PairwiseRankAgreement, ...]
    rows: tuple[RowStabilityEvidence, ...]

    def __post_init__(self) -> None:
        runs = tuple(self.runs)
        if len(runs) < 2:
            raise StabilityError("stability report requires at least two runs")
        names = tuple(run.run_name for run in runs)
        if len(names) != len(set(names)):
            raise StabilityError("stability run names must be unique")
        source_hashes = tuple(run.source_evidence_sha256 for run in runs)
        if len(source_hashes) != len(set(source_hashes)):
            raise StabilityError(
                "stability runs must reference distinct source artifacts"
            )
        identities = runs[0].row_identities
        if any(run.row_identities != identities for run in runs[1:]):
            raise StabilityError(
                "stability runs must contain the exact same ordered row population"
            )
        object.__setattr__(
            self,
            "row_population_sha256",
            _digest_hex(self.row_population_sha256, "row_population_sha256"),
        )
        if _population_digest(identities) != self.row_population_sha256:
            raise StabilityError(
                "row_population_sha256 differs from stability runs"
            )

        expected_pairs = len(runs) * (len(runs) - 1) // 2
        pairs = tuple(self.pairwise_rank_agreements)
        if len(pairs) != expected_pairs:
            raise StabilityError("pairwise rank-agreement count is incomplete")
        expected_names = {
            (runs[left].run_name, runs[right].run_name)
            for left in range(len(runs))
            for right in range(left + 1, len(runs))
        }
        observed_names = {
            (item.left_run_name, item.right_run_name) for item in pairs
        }
        if observed_names != expected_names:
            raise StabilityError(
                "pairwise rank agreements do not cover every run pair exactly"
            )

        rows = tuple(self.rows)
        if tuple(row.identity for row in rows) != identities:
            raise StabilityError(
                "row stability evidence order differs from run population"
            )
        if any(
            tuple(name for name, _ in row.run_positions) != names
            for row in rows
        ):
            raise StabilityError(
                "row stability evidence does not preserve run order"
            )

        object.__setattr__(self, "runs", runs)
        object.__setattr__(self, "pairwise_rank_agreements", pairs)
        object.__setattr__(self, "rows", rows)

    @property
    def evidence_sha256(self) -> str:
        return _digest(
            {
                "runs": [
                    {
                        "run_name": run.run_name,
                        "source_kind": run.source_kind.value,
                        "source_evidence_sha256": run.source_evidence_sha256,
                        "run_sha256": run.sha256_hex,
                    }
                    for run in self.runs
                ],
                "row_population_sha256": self.row_population_sha256,
                "pairwise_rank_agreements": [
                    item.as_dict() for item in self.pairwise_rank_agreements
                ],
                "rows": [row.as_dict() for row in self.rows],
            }
        )


def stability_run_from_calibrated(
    run_name: str,
    batch: CalibratedScoreBatch,
) -> StabilityRun:
    if not isinstance(batch, CalibratedScoreBatch):
        raise TypeError("batch must be CalibratedScoreBatch")
    return StabilityRun(
        run_name=run_name,
        source_kind=StabilitySourceKind.CALIBRATED_DETECTOR,
        source_evidence_sha256=batch.evidence_sha256,
        row_identities=batch.row_identities,
        anomaly_positions=tuple(
            score.empirical_midpoint_fraction for score in batch.scores
        ),
    )


def stability_run_from_ensemble(
    run_name: str,
    batch: EnsembleScoreBatch,
) -> StabilityRun:
    if not isinstance(batch, EnsembleScoreBatch):
        raise TypeError("batch must be EnsembleScoreBatch")
    return StabilityRun(
        run_name=run_name,
        source_kind=StabilitySourceKind.ENSEMBLE,
        source_evidence_sha256=batch.evidence_sha256,
        row_identities=batch.row_identities,
        anomaly_positions=tuple(
            row.ensemble_midpoint_fraction for row in batch.rows
        ),
    )


def analyze_stability(runs: Iterable[StabilityRun]) -> StabilityReport:
    if isinstance(runs, (str, bytes)):
        raise StabilityError("runs must be an iterable of StabilityRun")
    items = tuple(runs)
    if len(items) < 2:
        raise StabilityError("at least two stability runs are required")
    if any(not isinstance(run, StabilityRun) for run in items):
        raise TypeError("all runs must be StabilityRun")

    names = tuple(run.run_name for run in items)
    if len(names) != len(set(names)):
        raise StabilityError("stability run names must be unique")
    source_hashes = tuple(run.source_evidence_sha256 for run in items)
    if len(source_hashes) != len(set(source_hashes)):
        raise StabilityError(
            "stability analysis cannot include duplicate source artifacts"
        )
    identities = items[0].row_identities
    if any(run.row_identities != identities for run in items[1:]):
        raise StabilityError(
            "stability runs must contain the exact same ordered row population"
        )

    pairs: list[PairwiseRankAgreement] = []
    for left in range(len(items)):
        for right in range(left + 1, len(items)):
            rho, reason = _spearman(
                items[left].anomaly_positions,
                items[right].anomaly_positions,
            )
            pairs.append(
                PairwiseRankAgreement(
                    left_run_name=items[left].run_name,
                    right_run_name=items[right].run_name,
                    spearman_rho=rho,
                    unavailable_reason=reason,
                )
            )

    row_evidence: list[RowStabilityEvidence] = []
    for index, identity in enumerate(identities):
        positions = tuple(
            (run.run_name, run.anomaly_positions[index]) for run in items
        )
        values = tuple(value for _, value in positions)
        median = _median(values)
        row_evidence.append(
            RowStabilityEvidence(
                transaction_id=identity[0],
                award_id=identity[1],
                minimum_position=min(values),
                median_position=median,
                maximum_position=max(values),
                position_span=max(values) - min(values),
                median_absolute_deviation=_median(
                    tuple(abs(value - median) for value in values)
                ),
                run_positions=positions,
            )
        )

    return StabilityReport(
        runs=items,
        row_population_sha256=_population_digest(identities),
        pairwise_rank_agreements=tuple(pairs),
        rows=tuple(row_evidence),
    )


def _spearman(
    left: tuple[Decimal, ...],
    right: tuple[Decimal, ...],
) -> tuple[Decimal | None, str | None]:
    if len(left) != len(right) or len(left) < 2:
        raise StabilityError(
            "rank-correlation inputs must have equal length of at least two"
        )
    left_ranks = _average_ranks(left)
    right_ranks = _average_ranks(right)
    left_mean = sum(left_ranks, Decimal(0)) / Decimal(len(left_ranks))
    right_mean = sum(right_ranks, Decimal(0)) / Decimal(len(right_ranks))

    numerator = sum(
        (
            (left_value - left_mean) * (right_value - right_mean)
            for left_value, right_value in zip(left_ranks, right_ranks)
        ),
        Decimal(0),
    )
    left_ss = sum(
        ((value - left_mean) ** 2 for value in left_ranks), Decimal(0)
    )
    right_ss = sum(
        ((value - right_mean) ** 2 for value in right_ranks), Decimal(0)
    )
    if left_ss == 0 or right_ss == 0:
        return None, "constant_rank_distribution"

    with localcontext() as context:
        context.prec = DECIMAL_WORKING_PRECISION
        denominator = (left_ss * right_ss).sqrt()
        rho = numerator / denominator
    if rho > Decimal(1):
        rho = Decimal(1)
    elif rho < Decimal(-1):
        rho = Decimal(-1)
    return rho, None


def _average_ranks(values: tuple[Decimal, ...]) -> tuple[Decimal, ...]:
    if len(values) < 2:
        raise StabilityError("average ranks require at least two values")
    if any(
        not isinstance(value, Decimal) or not value.is_finite()
        for value in values
    ):
        raise StabilityError("rank values must be finite Decimal values")

    indexed = sorted((value, index) for index, value in enumerate(values))
    ranks = [Decimal(0)] * len(values)
    start = 0
    while start < len(indexed):
        end = start + 1
        while end < len(indexed) and indexed[end][0] == indexed[start][0]:
            end += 1
        average_rank = (
            Decimal(start + 1) + Decimal(end)
        ) / Decimal(2)
        for _, original_index in indexed[start:end]:
            ranks[original_index] = average_rank
        start = end
    return tuple(ranks)


def _median(values: tuple[Decimal, ...]) -> Decimal:
    if not values:
        raise StabilityError("median requires values")
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    with localcontext() as context:
        context.prec = DECIMAL_WORKING_PRECISION
        return (ordered[middle - 1] + ordered[middle]) / Decimal(2)


def _identities(
    values: Iterable[tuple[str, str]],
) -> tuple[tuple[str, str], ...]:
    identities: list[tuple[str, str]] = []
    for value in values:
        if not isinstance(value, tuple) or len(value) != 2:
            raise StabilityError(
                "row identities must be (transaction_id, award_id) tuples"
            )
        txid, award_id = value[0].strip(), value[1].strip()
        if not txid or not award_id:
            raise StabilityError("row identity values must not be blank")
        identities.append((txid, award_id))
    result = tuple(identities)
    if not result or len(result) != len(set(result)):
        raise StabilityError("row identities must be non-empty and unique")
    return result


def _population_digest(identities: Iterable[tuple[str, str]]) -> str:
    return _digest(_identities(identities))


def _fraction(value: Decimal, name: str) -> None:
    if (
        not isinstance(value, Decimal)
        or not value.is_finite()
        or value < 0
        or value > 1
    ):
        raise StabilityError(f"{name} must be finite Decimal in [0, 1]")


def _digest_hex(value: str, name: str) -> str:
    digest = value.strip().lower()
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise StabilityError(f"{name} must be a SHA-256 hex digest")
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
