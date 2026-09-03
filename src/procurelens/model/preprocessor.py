"""Fitted deterministic preprocessing for ProcureLens detector inputs.

Fits only on caller-provided training CandidateFeatureRow objects and reuses the
frozen state for later rows. Missing-value policies and numeric transforms come
entirely from a resolved preprocessing specification; this module never infers
policy from feature names or evidence families.

Original missingness indicators are emitted before imputed values are transformed.
No test/inference row contributes to fitted medians, means, variances, or robust
quantiles.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, localcontext
from hashlib import sha256
import json
from types import MappingProxyType
from typing import Any, Iterable, Mapping

from procurelens.model.feature_row import CandidateFeatureRow
from procurelens.model.preprocessing_spec import (
    MissingValueStrategy,
    NumericTransform,
    PreprocessingSpec,
    ResolvedPreprocessingSpec,
)
from procurelens.statistics.robust import (
    QuantileMethod,
    quantile_sorted,
    sorted_decimals,
)


class PreprocessorError(ValueError):
    """Raised when fitting or applying preprocessing would be inconsistent."""


DECIMAL_WORKING_PRECISION = 50
_HALF = Decimal("0.5")


@dataclass(frozen=True, slots=True)
class FittedFeatureState:
    feature_name: str
    missing_strategy: MissingValueStrategy
    add_missing_indicator: bool
    transform: NumericTransform
    observed_training_count: int
    missing_training_count: int
    impute_value: Decimal | None
    center: Decimal | None
    scale: Decimal | None
    robust_quantile_low: Decimal | None = None
    robust_quantile_high: Decimal | None = None

    def __post_init__(self) -> None:
        name = self.feature_name.strip()
        if not name:
            raise PreprocessorError("feature_name must not be blank")
        object.__setattr__(self, "feature_name", name)
        object.__setattr__(
            self, "missing_strategy", MissingValueStrategy(self.missing_strategy)
        )
        object.__setattr__(self, "transform", NumericTransform(self.transform))
        if not isinstance(self.add_missing_indicator, bool):
            raise PreprocessorError("add_missing_indicator must be bool")

        for field_name in ("observed_training_count", "missing_training_count"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise PreprocessorError(
                    f"{field_name} must be a non-negative integer"
                )
        if self.observed_training_count + self.missing_training_count < 1:
            raise PreprocessorError("fitted state requires at least one training row")

        for field_name in ("impute_value", "center", "scale"):
            value = getattr(self, field_name)
            if value is not None and (
                not isinstance(value, Decimal) or not value.is_finite()
            ):
                raise PreprocessorError(
                    f"{field_name} must be finite Decimal or None"
                )

        if self.missing_strategy is MissingValueStrategy.REQUIRE_PRESENT:
            if self.missing_training_count or self.impute_value is not None:
                raise PreprocessorError(
                    "require-present feature cannot have training missingness/imputation"
                )
        elif self.impute_value is None:
            raise PreprocessorError(
                "imputing missing strategy requires fitted impute_value"
            )

        low, high = self.robust_quantile_low, self.robust_quantile_high
        if self.transform is NumericTransform.IDENTITY:
            if self.center is not None or self.scale is not None:
                raise PreprocessorError(
                    "identity transform cannot carry center/scale"
                )
            if low is not None or high is not None:
                raise PreprocessorError(
                    "identity transform cannot carry robust quantiles"
                )
        elif self.transform is NumericTransform.STANDARD_SCALE:
            if self.center is None or self.scale is None or self.scale <= 0:
                raise PreprocessorError(
                    "standard scaling requires positive fitted scale and center"
                )
            if low is not None or high is not None:
                raise PreprocessorError(
                    "standard scaling cannot carry robust quantiles"
                )
        else:
            if (
                self.center is None
                or self.scale is None
                or self.scale <= 0
                or low is None
                or high is None
            ):
                raise PreprocessorError(
                    "robust scaling requires center, positive scale, and quantiles"
                )
            _probability(low, "robust_quantile_low")
            _probability(high, "robust_quantile_high")
            if low >= high:
                raise PreprocessorError(
                    "robust_quantile_low must be below robust_quantile_high"
                )

    @property
    def training_count(self) -> int:
        return self.observed_training_count + self.missing_training_count

    def as_dict(self) -> dict[str, Any]:
        return {
            "feature_name": self.feature_name,
            "missing_strategy": self.missing_strategy.value,
            "add_missing_indicator": self.add_missing_indicator,
            "transform": self.transform.value,
            "observed_training_count": self.observed_training_count,
            "missing_training_count": self.missing_training_count,
            "impute_value": _text(self.impute_value),
            "center": _text(self.center),
            "scale": _text(self.scale),
            "robust_quantile_low": _text(self.robust_quantile_low),
            "robust_quantile_high": _text(self.robust_quantile_high),
        }


@dataclass(frozen=True, slots=True)
class FittedPreprocessor:
    """Frozen train-only preprocessing state."""

    spec: PreprocessingSpec
    resolved_spec_sha256: str
    training_population_sha256: str
    training_row_count: int
    states: tuple[FittedFeatureState, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.spec, PreprocessingSpec):
            raise TypeError("spec must be PreprocessingSpec")
        object.__setattr__(
            self,
            "resolved_spec_sha256",
            _digest_hex(self.resolved_spec_sha256, "resolved_spec_sha256"),
        )
        object.__setattr__(
            self,
            "training_population_sha256",
            _digest_hex(
                self.training_population_sha256, "training_population_sha256"
            ),
        )
        if (
            isinstance(self.training_row_count, bool)
            or not isinstance(self.training_row_count, int)
            or self.training_row_count < 1
        ):
            raise PreprocessorError("training_row_count must be positive integer")
        states = tuple(self.states)
        if tuple(state.feature_name for state in states) != self.spec.feature_names:
            raise PreprocessorError(
                "fitted feature states must match preprocessing rule order"
            )
        if any(state.training_count != self.training_row_count for state in states):
            raise PreprocessorError(
                "every fitted feature state must cover the full training population"
            )
        object.__setattr__(self, "states", states)

    @property
    def output_feature_names(self) -> tuple[str, ...]:
        return self.spec.output_feature_names

    @property
    def sha256_hex(self) -> str:
        return _digest(
            {
                "preprocessing_spec_sha256": self.spec.sha256_hex,
                "resolved_spec_sha256": self.resolved_spec_sha256,
                "training_population_sha256": self.training_population_sha256,
                "training_row_count": self.training_row_count,
                "states": [state.as_dict() for state in self.states],
                "output_feature_names": list(self.output_feature_names),
            }
        )


@dataclass(frozen=True, slots=True)
class PreprocessedRow:
    transaction_id: str
    award_id: str
    source_row_evidence_sha256: str
    preprocessor_sha256: str
    output_feature_names: tuple[str, ...]
    values: tuple[Decimal, ...]

    def __post_init__(self) -> None:
        for field_name in ("transaction_id", "award_id"):
            text = getattr(self, field_name).strip()
            if not text:
                raise PreprocessorError(f"{field_name} must not be blank")
            object.__setattr__(self, field_name, text)
        for field_name in ("source_row_evidence_sha256", "preprocessor_sha256"):
            object.__setattr__(
                self,
                field_name,
                _digest_hex(getattr(self, field_name), field_name),
            )
        names = tuple(name.strip() for name in self.output_feature_names)
        if (
            not names
            or any(not name for name in names)
            or len(names) != len(set(names))
        ):
            raise PreprocessorError(
                "output_feature_names must be non-empty and unique"
            )
        values = tuple(self.values)
        if len(names) != len(values):
            raise PreprocessorError("output names/value lengths differ")
        if any(
            not isinstance(value, Decimal) or not value.is_finite()
            for value in values
        ):
            raise PreprocessorError(
                "preprocessed values must be finite Decimal values"
            )
        object.__setattr__(self, "output_feature_names", names)
        object.__setattr__(self, "values", values)

    @property
    def as_mapping(self) -> Mapping[str, Decimal]:
        return MappingProxyType(dict(zip(self.output_feature_names, self.values)))

    @property
    def evidence_sha256(self) -> str:
        return _digest(
            {
                "transaction_id": self.transaction_id,
                "award_id": self.award_id,
                "source_row_evidence_sha256": self.source_row_evidence_sha256,
                "preprocessor_sha256": self.preprocessor_sha256,
                "values": [
                    (name, str(value))
                    for name, value in zip(self.output_feature_names, self.values)
                ],
            }
        )


def fit_preprocessor(
    rows: Iterable[CandidateFeatureRow],
    resolved_spec: ResolvedPreprocessingSpec,
) -> FittedPreprocessor:
    """Fit imputation/scaling state from training rows only."""

    if not isinstance(resolved_spec, ResolvedPreprocessingSpec):
        raise TypeError("resolved_spec must be ResolvedPreprocessingSpec")
    training = _validate_training_rows(rows, resolved_spec)
    states: list[FittedFeatureState] = []

    with localcontext() as context:
        context.prec = DECIMAL_WORKING_PRECISION
        for rule in resolved_spec.spec.rules:
            raw = tuple(row.get(rule.feature_name).value for row in training)
            observed = tuple(value for value in raw if value is not None)
            missing_count = len(raw) - len(observed)

            if rule.missing_strategy is MissingValueStrategy.REQUIRE_PRESENT:
                if missing_count:
                    raise PreprocessorError(
                        f"{rule.feature_name}: require_present but "
                        f"{missing_count} training rows are missing"
                    )
                impute_value = None
                filled = observed
            elif rule.missing_strategy is MissingValueStrategy.TRAIN_MEDIAN:
                if not observed:
                    raise PreprocessorError(
                        f"{rule.feature_name}: cannot fit median with no observed values"
                    )
                ordered = sorted_decimals(observed)
                impute_value = quantile_sorted(
                    ordered, _HALF, method=QuantileMethod.LINEAR_TYPE7
                )
                filled = tuple(
                    impute_value if value is None else value for value in raw
                )
            else:
                impute_value = rule.constant_fill_value
                if impute_value is None:
                    raise PreprocessorError(
                        f"{rule.feature_name}: constant imputation has no fill value"
                    )
                filled = tuple(
                    impute_value if value is None else value for value in raw
                )

            center, scale = _fit_transform_state(rule.transform, filled, rule)
            states.append(
                FittedFeatureState(
                    feature_name=rule.feature_name,
                    missing_strategy=rule.missing_strategy,
                    add_missing_indicator=rule.add_missing_indicator,
                    transform=rule.transform,
                    observed_training_count=len(observed),
                    missing_training_count=missing_count,
                    impute_value=impute_value,
                    center=center,
                    scale=scale,
                    robust_quantile_low=rule.robust_quantile_low,
                    robust_quantile_high=rule.robust_quantile_high,
                )
            )

    return FittedPreprocessor(
        spec=resolved_spec.spec,
        resolved_spec_sha256=resolved_spec.sha256_hex,
        training_population_sha256=_training_population_digest(training),
        training_row_count=len(training),
        states=tuple(states),
    )


def transform_row(
    row: CandidateFeatureRow,
    fitted: FittedPreprocessor,
) -> PreprocessedRow:
    """Apply frozen preprocessing state without fitting on the target row."""

    if not isinstance(row, CandidateFeatureRow):
        raise TypeError("row must be CandidateFeatureRow")
    if not isinstance(fitted, FittedPreprocessor):
        raise TypeError("fitted must be FittedPreprocessor")
    if row.catalog_sha256 != fitted.spec.catalog_sha256:
        raise PreprocessorError(
            "row catalog fingerprint differs from fitted preprocessing contract"
        )

    output: list[Decimal] = []
    with localcontext() as context:
        context.prec = DECIMAL_WORKING_PRECISION
        for state in fitted.states:
            candidate = row.get(state.feature_name)
            was_missing = candidate.value is None

            if was_missing:
                if state.missing_strategy is MissingValueStrategy.REQUIRE_PRESENT:
                    raise PreprocessorError(
                        f"{state.feature_name}: missing value violates require_present"
                    )
                if state.impute_value is None:
                    raise PreprocessorError(
                        f"{state.feature_name}: fitted imputation value is absent"
                    )
                value = state.impute_value
            else:
                value = candidate.value
                if value is None:
                    raise AssertionError("unreachable missing candidate")

            output.append(_apply_transform(value, state))
            if state.add_missing_indicator:
                output.append(Decimal(1) if was_missing else Decimal(0))

    return PreprocessedRow(
        transaction_id=row.transaction_id,
        award_id=row.award_id,
        source_row_evidence_sha256=row.evidence_sha256,
        preprocessor_sha256=fitted.sha256_hex,
        output_feature_names=fitted.output_feature_names,
        values=tuple(output),
    )


def transform_rows(
    rows: Iterable[CandidateFeatureRow],
    fitted: FittedPreprocessor,
) -> tuple[PreprocessedRow, ...]:
    """Apply one frozen preprocessor to rows in caller-provided order."""

    return tuple(transform_row(row, fitted) for row in rows)


def _fit_transform_state(
    transform: NumericTransform,
    values: tuple[Decimal, ...],
    rule: Any,
) -> tuple[Decimal | None, Decimal | None]:
    if not values:
        raise PreprocessorError("cannot fit transform on empty values")
    if transform is NumericTransform.IDENTITY:
        return None, None

    count = Decimal(len(values))
    if transform is NumericTransform.STANDARD_SCALE:
        center = sum(values, Decimal(0)) / count
        variance = sum(
            ((value - center) * (value - center) for value in values),
            Decimal(0),
        ) / count
        if variance == 0:
            raise PreprocessorError(
                f"{rule.feature_name}: standard scaling has zero training variance"
            )
        scale = variance.sqrt()
        if scale <= 0:
            raise PreprocessorError(
                f"{rule.feature_name}: standard scaling produced invalid scale"
            )
        return center, scale

    ordered = sorted_decimals(values)
    low = rule.robust_quantile_low
    high = rule.robust_quantile_high
    if low is None or high is None:
        raise PreprocessorError(
            f"{rule.feature_name}: robust scaling quantiles are absent"
        )
    center = quantile_sorted(
        ordered, _HALF, method=QuantileMethod.LINEAR_TYPE7
    )
    lower = quantile_sorted(
        ordered, low, method=QuantileMethod.LINEAR_TYPE7
    )
    upper = quantile_sorted(
        ordered, high, method=QuantileMethod.LINEAR_TYPE7
    )
    scale = upper - lower
    if scale == 0:
        raise PreprocessorError(
            f"{rule.feature_name}: robust scaling has zero training quantile range"
        )
    if scale < 0:
        raise PreprocessorError(
            f"{rule.feature_name}: robust scaling produced negative scale"
        )
    return center, scale


def _apply_transform(
    value: Decimal,
    state: FittedFeatureState,
) -> Decimal:
    if state.transform is NumericTransform.IDENTITY:
        return value
    if state.center is None or state.scale is None or state.scale <= 0:
        raise PreprocessorError(
            f"{state.feature_name}: fitted transform state is incomplete"
        )
    return (value - state.center) / state.scale


def _validate_training_rows(
    rows: Iterable[CandidateFeatureRow],
    resolved_spec: ResolvedPreprocessingSpec,
) -> tuple[CandidateFeatureRow, ...]:
    if isinstance(rows, (str, bytes)):
        raise PreprocessorError("rows must be an iterable of CandidateFeatureRow")
    training = tuple(rows)
    if not training:
        raise PreprocessorError("at least one training row is required")

    seen: set[tuple[str, str]] = set()
    for row in training:
        if not isinstance(row, CandidateFeatureRow):
            raise TypeError("all training rows must be CandidateFeatureRow")
        if row.catalog_sha256 != resolved_spec.spec.catalog_sha256:
            raise PreprocessorError(
                "training row catalog fingerprint differs from preprocessing spec"
            )
        identity = (row.transaction_id, row.award_id)
        if identity in seen:
            raise PreprocessorError(
                f"duplicate training row identity: {identity!r}"
            )
        seen.add(identity)
    return training


def _training_population_digest(
    rows: tuple[CandidateFeatureRow, ...],
) -> str:
    return _digest(
        sorted(
            (
                row.transaction_id,
                row.award_id,
                row.evidence_sha256,
            )
            for row in rows
        )
    )


def _probability(value: Decimal, name: str) -> None:
    if not isinstance(value, Decimal) or not value.is_finite():
        raise PreprocessorError(f"{name} must be finite Decimal")
    if value < Decimal(0) or value > Decimal(1):
        raise PreprocessorError(f"{name} must be between 0 and 1")


def _digest_hex(value: str, name: str) -> str:
    digest = value.strip().lower()
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise PreprocessorError(f"{name} must be a SHA-256 hex digest")
    return digest


def _text(value: Decimal | None) -> str | None:
    return None if value is None else str(value)


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
