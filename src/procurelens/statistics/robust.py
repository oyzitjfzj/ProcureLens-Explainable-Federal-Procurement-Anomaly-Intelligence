"""Small, deterministic robust-statistics primitives for ProcureLens.

The functions here describe distributions; they do not decide what is risky or
anomalous. No procurement threshold lives in this module.

Decimal inputs are kept as Decimal through quantiles and robust distances so
money-derived features are reproducible and do not silently inherit binary
floating-point rounding. Callers decide how these descriptive measurements are
used later.
"""
from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR
from enum import Enum


class RobustStatisticsError(ValueError):
    """Raised when robust statistics receive invalid or inconsistent input."""


class QuantileMethod(str, Enum):
    """Explicit quantile definitions supported by ProcureLens."""

    LINEAR_TYPE7 = "linear_type7"
    NEAREST_RANK = "nearest_rank"


# Phi^-1(0.75). It is part of the standard modified-z normalization under a
# normal reference distribution, not an anomaly cutoff or risk threshold.
NORMAL_MAD_FACTOR = Decimal("0.6744897501960817")


@dataclass(frozen=True, slots=True)
class EmpiricalPosition:
    """Tie-aware position of a target relative to an observed reference set."""

    less: int
    equal: int
    greater: int
    lower_fraction: Decimal
    midpoint_fraction: Decimal
    upper_fraction: Decimal

    def __post_init__(self) -> None:
        counts = (self.less, self.equal, self.greater)
        if any(isinstance(value, bool) or not isinstance(value, int) for value in counts):
            raise RobustStatisticsError("empirical-position counts must be integers")
        if any(value < 0 for value in counts) or self.total < 1:
            raise RobustStatisticsError("empirical-position counts are inconsistent")
        if not (
            Decimal(0) <= self.lower_fraction <= self.midpoint_fraction
            <= self.upper_fraction <= Decimal(1)
        ):
            raise RobustStatisticsError("empirical-position fractions are inconsistent")

    @property
    def total(self) -> int:
        return self.less + self.equal + self.greater


@dataclass(frozen=True, slots=True)
class RobustSummary:
    """Robust location and spread summary for one non-empty Decimal sample."""

    count: int
    minimum: Decimal
    first_quartile: Decimal
    median: Decimal
    third_quartile: Decimal
    maximum: Decimal
    interquartile_range: Decimal
    median_absolute_deviation: Decimal
    quantile_method: QuantileMethod

    def __post_init__(self) -> None:
        if isinstance(self.count, bool) or not isinstance(self.count, int) or self.count < 1:
            raise RobustStatisticsError("summary count must be a positive integer")
        values = (
            self.minimum,
            self.first_quartile,
            self.median,
            self.third_quartile,
            self.maximum,
            self.interquartile_range,
            self.median_absolute_deviation,
        )
        if any(not isinstance(value, Decimal) or not value.is_finite() for value in values):
            raise RobustStatisticsError("summary values must be finite Decimal values")
        if not (
            self.minimum <= self.first_quartile <= self.median
            <= self.third_quartile <= self.maximum
        ):
            raise RobustStatisticsError("summary ordering is inconsistent")
        if self.interquartile_range != self.third_quartile - self.first_quartile:
            raise RobustStatisticsError("interquartile range is inconsistent")
        if self.median_absolute_deviation < 0:
            raise RobustStatisticsError("median absolute deviation must not be negative")
        object.__setattr__(self, "quantile_method", QuantileMethod(self.quantile_method))


_ONE_QUARTER = Decimal("0.25")
_ONE_HALF = Decimal("0.5")
_THREE_QUARTERS = Decimal("0.75")


def sorted_decimals(values: Iterable[Decimal]) -> tuple[Decimal, ...]:
    """Validate finite Decimal observations and return a deterministic sort."""

    if isinstance(values, (str, bytes)):
        raise RobustStatisticsError("values must be an iterable of Decimal observations")
    cleaned: list[Decimal] = []
    for value in values:
        if not isinstance(value, Decimal) or not value.is_finite():
            raise RobustStatisticsError("all observations must be finite Decimal values")
        cleaned.append(value)
    if not cleaned:
        raise RobustStatisticsError("at least one observation is required")
    return tuple(sorted(cleaned))


def summarize(
    values: Iterable[Decimal],
    *,
    method: QuantileMethod = QuantileMethod.LINEAR_TYPE7,
) -> RobustSummary:
    """Validate, sort, and robustly summarize a non-empty sample."""

    return summarize_sorted(sorted_decimals(values), method=method)


def summarize_sorted(
    values: Sequence[Decimal],
    *,
    method: QuantileMethod = QuantileMethod.LINEAR_TYPE7,
) -> RobustSummary:
    """Summarize an already sorted non-empty finite Decimal sequence."""

    ordered = _validate_sorted(values)
    method = QuantileMethod(method)
    q1 = quantile_sorted(ordered, _ONE_QUARTER, method=method)
    median = quantile_sorted(ordered, _ONE_HALF, method=method)
    q3 = quantile_sorted(ordered, _THREE_QUARTERS, method=method)
    deviations = tuple(sorted(abs(value - median) for value in ordered))
    mad = quantile_sorted(deviations, _ONE_HALF, method=method)
    return RobustSummary(
        count=len(ordered),
        minimum=ordered[0],
        first_quartile=q1,
        median=median,
        third_quartile=q3,
        maximum=ordered[-1],
        interquartile_range=q3 - q1,
        median_absolute_deviation=mad,
        quantile_method=method,
    )


def quantile_sorted(
    values: Sequence[Decimal],
    probability: Decimal,
    *,
    method: QuantileMethod = QuantileMethod.LINEAR_TYPE7,
) -> Decimal:
    """Compute an explicit quantile definition over sorted Decimal values."""

    ordered = _validate_sorted(values)
    if not isinstance(probability, Decimal) or not probability.is_finite():
        raise RobustStatisticsError("quantile probability must be a finite Decimal")
    if not (Decimal(0) <= probability <= Decimal(1)):
        raise RobustStatisticsError("quantile probability must be between 0 and 1")
    method = QuantileMethod(method)
    if len(ordered) == 1:
        return ordered[0]

    if method is QuantileMethod.LINEAR_TYPE7:
        position = Decimal(len(ordered) - 1) * probability
        lower = int(position.to_integral_value(rounding=ROUND_FLOOR))
        upper = min(lower + 1, len(ordered) - 1)
        fraction = position - Decimal(lower)
        return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction

    if method is QuantileMethod.NEAREST_RANK:
        if probability == 0:
            return ordered[0]
        rank = int(
            (probability * Decimal(len(ordered))).to_integral_value(
                rounding=ROUND_CEILING
            )
        )
        return ordered[max(1, rank) - 1]

    raise RobustStatisticsError(f"unsupported quantile method: {method!r}")


def empirical_position_sorted(
    values: Sequence[Decimal],
    target: Decimal,
) -> EmpiricalPosition:
    """Return lower/mid/upper empirical fractions without hiding ties."""

    ordered = _validate_sorted(values)
    if not isinstance(target, Decimal) or not target.is_finite():
        raise RobustStatisticsError("target must be a finite Decimal")
    left = bisect_left(ordered, target)
    right = bisect_right(ordered, target)
    less = left
    equal = right - left
    greater = len(ordered) - right
    total = Decimal(len(ordered))
    return EmpiricalPosition(
        less=less,
        equal=equal,
        greater=greater,
        lower_fraction=Decimal(less) / total,
        midpoint_fraction=(Decimal(less) + Decimal(equal) / 2) / total,
        upper_fraction=Decimal(less + equal) / total,
    )


def modified_z(target: Decimal, summary: RobustSummary) -> Decimal | None:
    """Return the standard MAD-based modified z value, or None if MAD is zero."""

    _validate_target(target)
    if not isinstance(summary, RobustSummary):
        raise TypeError("summary must be RobustSummary")
    mad = summary.median_absolute_deviation
    if mad == 0:
        return None
    return NORMAL_MAD_FACTOR * (target - summary.median) / mad


def iqr_distance(target: Decimal, summary: RobustSummary) -> Decimal | None:
    """Return median-centered distance in IQR units, or None if IQR is zero."""

    _validate_target(target)
    if not isinstance(summary, RobustSummary):
        raise TypeError("summary must be RobustSummary")
    spread = summary.interquartile_range
    if spread == 0:
        return None
    return (target - summary.median) / spread


def remove_one_sorted(
    values: Sequence[Decimal],
    target: Decimal,
) -> tuple[Decimal, ...]:
    """Remove exactly one matching observation from a sorted reference sample."""

    ordered = _validate_sorted(values)
    _validate_target(target)
    index = bisect_left(ordered, target)
    if index >= len(ordered) or ordered[index] != target:
        raise RobustStatisticsError("target observation is absent from reference sample")
    return ordered[:index] + ordered[index + 1 :]


def decimal_identity(value: Decimal) -> str:
    """Stable numeric identity: equal Decimal values produce equal text."""

    _validate_target(value)
    if value.is_zero():
        return "0"
    sign, digits_tuple, exponent = value.as_tuple()
    digits = list(digits_tuple)
    while digits and digits[-1] == 0:
        digits.pop()
        exponent += 1
    return f"{sign}:{''.join(str(digit) for digit in digits)}:{exponent}"


def _validate_target(value: Decimal) -> None:
    if not isinstance(value, Decimal) or not value.is_finite():
        raise RobustStatisticsError("value must be a finite Decimal")


def _validate_sorted(values: Sequence[Decimal]) -> tuple[Decimal, ...]:
    if isinstance(values, (str, bytes)):
        raise RobustStatisticsError("values must be a sorted Decimal sequence")
    ordered = tuple(values)
    if not ordered:
        raise RobustStatisticsError("at least one observation is required")
    if any(not isinstance(value, Decimal) or not value.is_finite() for value in ordered):
        raise RobustStatisticsError("all observations must be finite Decimal values")
    if any(ordered[index] > ordered[index + 1] for index in range(len(ordered) - 1)):
        raise RobustStatisticsError("values must be sorted in non-decreasing order")
    return ordered
