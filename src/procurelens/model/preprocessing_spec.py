"""Declarative preprocessing contracts for ProcureLens model inputs.

A preprocessing specification pins how an already selected ordered feature set
may be converted into numeric detector inputs. It separates model policy from
feature evidence: no preprocessing rule is inferred from a feature name,
family, or source.

Missing values remain explicit until a fitted preprocessing implementation
applies the caller-selected policy. Any statistic needed for imputation or
scaling must be learned from training data only; this module stores policy and
stable fingerprints, not fitted statistics.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from hashlib import sha256
import json
from typing import Any

from procurelens.model.feature_spec import ResolvedFeatureSelection


class PreprocessingSpecError(ValueError):
    """Raised when a preprocessing policy contract is inconsistent."""


class MissingValueStrategy(str, Enum):
    """How a selected feature may handle a missing numeric measurement."""

    REQUIRE_PRESENT = "require_present"
    TRAIN_MEDIAN = "train_median"
    CONSTANT = "constant"


class NumericTransform(str, Enum):
    """Supported numeric transforms whose fitted state is learned downstream."""

    IDENTITY = "identity"
    STANDARD_SCALE = "standard_scale"
    ROBUST_SCALE = "robust_scale"


@dataclass(frozen=True, slots=True)
class FeaturePreprocessingRule:
    """Explicit preprocessing policy for one selected canonical feature."""

    feature_name: str
    missing_strategy: MissingValueStrategy
    add_missing_indicator: bool
    transform: NumericTransform
    constant_fill_value: Decimal | None = None
    robust_quantile_low: Decimal | None = None
    robust_quantile_high: Decimal | None = None

    def __post_init__(self) -> None:
        name = self.feature_name.strip()
        if not name:
            raise PreprocessingSpecError("feature_name must not be blank")
        object.__setattr__(self, "feature_name", name)
        object.__setattr__(
            self, "missing_strategy", MissingValueStrategy(self.missing_strategy)
        )
        object.__setattr__(self, "transform", NumericTransform(self.transform))

        if not isinstance(self.add_missing_indicator, bool):
            raise PreprocessingSpecError("add_missing_indicator must be bool")

        fill = self.constant_fill_value
        if fill is not None and (
            not isinstance(fill, Decimal) or not fill.is_finite()
        ):
            raise PreprocessingSpecError(
                "constant_fill_value must be finite Decimal or None"
            )
        if self.missing_strategy is MissingValueStrategy.CONSTANT:
            if fill is None:
                raise PreprocessingSpecError(
                    "constant missing strategy requires constant_fill_value"
                )
        elif fill is not None:
            raise PreprocessingSpecError(
                "constant_fill_value is only valid with constant missing strategy"
            )

        low, high = self.robust_quantile_low, self.robust_quantile_high
        if self.transform is NumericTransform.ROBUST_SCALE:
            if low is None or high is None:
                raise PreprocessingSpecError(
                    "robust scaling requires explicit low/high quantiles"
                )
            _validate_probability(low, "robust_quantile_low")
            _validate_probability(high, "robust_quantile_high")
            if low >= high:
                raise PreprocessingSpecError(
                    "robust_quantile_low must be below robust_quantile_high"
                )
        elif low is not None or high is not None:
            raise PreprocessingSpecError(
                "robust quantiles are only valid with robust scaling"
            )

    @property
    def fit_required(self) -> bool:
        return (
            self.missing_strategy is MissingValueStrategy.TRAIN_MEDIAN
            or self.transform
            in (NumericTransform.STANDARD_SCALE, NumericTransform.ROBUST_SCALE)
        )

    @property
    def output_value_name(self) -> str:
        return self.feature_name

    @property
    def missing_indicator_name(self) -> str | None:
        if not self.add_missing_indicator:
            return None
        return f"{self.feature_name}__missing"

    def as_dict(self) -> dict[str, Any]:
        return {
            "feature_name": self.feature_name,
            "missing_strategy": self.missing_strategy.value,
            "add_missing_indicator": self.add_missing_indicator,
            "transform": self.transform.value,
            "constant_fill_value": (
                None
                if self.constant_fill_value is None
                else str(self.constant_fill_value)
            ),
            "robust_quantile_low": (
                None
                if self.robust_quantile_low is None
                else str(self.robust_quantile_low)
            ),
            "robust_quantile_high": (
                None
                if self.robust_quantile_high is None
                else str(self.robust_quantile_high)
            ),
        }


@dataclass(frozen=True, slots=True)
class PreprocessingSpec:
    """Ordered preprocessing contract pinned to one resolved feature selection."""

    name: str
    description: str
    selection_sha256: str
    catalog_sha256: str
    rules: tuple[FeaturePreprocessingRule, ...]

    def __post_init__(self) -> None:
        for field_name in ("name", "description"):
            text = getattr(self, field_name).strip()
            if not text:
                raise PreprocessingSpecError(f"{field_name} must not be blank")
            object.__setattr__(self, field_name, text)

        object.__setattr__(
            self,
            "selection_sha256",
            _validate_digest(self.selection_sha256, "selection_sha256"),
        )
        object.__setattr__(
            self,
            "catalog_sha256",
            _validate_digest(self.catalog_sha256, "catalog_sha256"),
        )

        rules = tuple(self.rules)
        if not rules:
            raise PreprocessingSpecError(
                "preprocessing spec requires at least one feature rule"
            )
        names = tuple(rule.feature_name for rule in rules)
        if len(names) != len(set(names)):
            raise PreprocessingSpecError(
                "preprocessing spec contains duplicate feature rules"
            )
        object.__setattr__(self, "rules", rules)

        output_names = self.output_feature_names
        if len(output_names) != len(set(output_names)):
            raise PreprocessingSpecError(
                "preprocessing output feature names collide"
            )

    @property
    def feature_names(self) -> tuple[str, ...]:
        return tuple(rule.feature_name for rule in self.rules)

    @property
    def output_feature_names(self) -> tuple[str, ...]:
        result: list[str] = []
        selected = set(self.feature_names)
        for rule in self.rules:
            result.append(rule.output_value_name)
            indicator = rule.missing_indicator_name
            if indicator is not None:
                if indicator in selected:
                    raise PreprocessingSpecError(
                        "missing-indicator name collides with selected feature name: "
                        f"{indicator}"
                    )
                result.append(indicator)
        return tuple(result)

    @property
    def input_feature_count(self) -> int:
        return len(self.rules)

    @property
    def output_feature_count(self) -> int:
        return len(self.output_feature_names)

    @property
    def fit_required(self) -> bool:
        return any(rule.fit_required for rule in self.rules)

    @property
    def sha256_hex(self) -> str:
        return _digest(self.as_dict(include_sha=False))

    def as_dict(self, *, include_sha: bool = True) -> dict[str, Any]:
        result = {
            "name": self.name,
            "description": self.description,
            "selection_sha256": self.selection_sha256,
            "catalog_sha256": self.catalog_sha256,
            "input_feature_count": self.input_feature_count,
            "output_feature_count": self.output_feature_count,
            "fit_required": self.fit_required,
            "rules": [rule.as_dict() for rule in self.rules],
            "output_feature_names": list(self.output_feature_names),
        }
        if include_sha:
            result["sha256"] = self.sha256_hex
        return result


@dataclass(frozen=True, slots=True)
class ResolvedPreprocessingSpec:
    """Preprocessing contract cross-checked against its feature selection."""

    selection: ResolvedFeatureSelection
    spec: PreprocessingSpec

    def __post_init__(self) -> None:
        if not isinstance(self.selection, ResolvedFeatureSelection):
            raise TypeError("selection must be ResolvedFeatureSelection")
        if not isinstance(self.spec, PreprocessingSpec):
            raise TypeError("spec must be PreprocessingSpec")

        if self.spec.catalog_sha256 != self.selection.spec.catalog_sha256:
            raise PreprocessingSpecError(
                "preprocessing catalog fingerprint differs from feature selection"
            )
        if self.spec.selection_sha256 != self.selection.sha256_hex:
            raise PreprocessingSpecError(
                "preprocessing selection fingerprint differs from resolved selection"
            )
        if self.spec.feature_names != self.selection.feature_names:
            raise PreprocessingSpecError(
                "preprocessing rules must exactly match selected feature order"
            )

    @property
    def sha256_hex(self) -> str:
        return _digest(
            {
                "resolved_selection_sha256": self.selection.sha256_hex,
                "preprocessing_spec_sha256": self.spec.sha256_hex,
            }
        )


def resolve_preprocessing_spec(
    selection: ResolvedFeatureSelection,
    spec: PreprocessingSpec,
) -> ResolvedPreprocessingSpec:
    """Cross-check one explicit preprocessing policy against an ordered selection."""

    return ResolvedPreprocessingSpec(selection=selection, spec=spec)


def _validate_probability(value: Decimal, name: str) -> None:
    if not isinstance(value, Decimal) or not value.is_finite():
        raise PreprocessingSpecError(f"{name} must be a finite Decimal")
    if value < Decimal(0) or value > Decimal(1):
        raise PreprocessingSpecError(f"{name} must be between 0 and 1")


def _validate_digest(value: str, name: str) -> str:
    digest = value.strip().lower()
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise PreprocessingSpecError(f"{name} must be a SHA-256 hex digest")
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
