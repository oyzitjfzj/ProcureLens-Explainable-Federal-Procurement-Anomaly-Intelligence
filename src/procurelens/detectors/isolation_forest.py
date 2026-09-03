"""Auditable scikit-learn Isolation Forest adapter for ProcureLens.

Fits a standard IsolationForest as one detector member, not as a final risk
classifier. The adapter uses score_samples (native direction: lower is more
abnormal), fixes contamination to "auto" because review thresholds belong to a
later policy layer, and explicitly converts the Decimal matrix to float32 before
calling scikit-learn.

Configuration, numeric conversion, fitted tree structure, training matrix, and
scored matrix are fingerprinted. No binary outlier decision or risk score is
produced here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from hashlib import sha256
import json
import math
from typing import Any

import numpy as np
import sklearn
from sklearn.ensemble import IsolationForest

from procurelens.detectors.base import (
    DetectorScoreBatch,
    ScoreOrientation,
    build_detector_scores,
)
from procurelens.model.matrix import PreprocessedMatrix


class IsolationForestAdapterError(ValueError):
    """Raised when Isolation Forest configuration/evidence is inconsistent."""


@dataclass(frozen=True, slots=True)
class IsolationForestConfig:
    """Explicit reproducible parameters that affect the fitted forest."""

    n_estimators: int
    max_samples: str | int | Decimal
    max_features: int | Decimal
    bootstrap: bool
    random_state: int
    n_jobs: int | None = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.n_estimators, bool)
            or not isinstance(self.n_estimators, int)
            or self.n_estimators < 1
        ):
            raise IsolationForestAdapterError(
                "n_estimators must be a positive integer"
            )

        max_samples = self.max_samples
        if isinstance(max_samples, str):
            if max_samples != "auto":
                raise IsolationForestAdapterError(
                    'max_samples string must be exactly "auto"'
                )
        elif isinstance(max_samples, bool):
            raise IsolationForestAdapterError(
                "max_samples must be auto, positive int, or Decimal fraction"
            )
        elif isinstance(max_samples, int):
            if max_samples < 1:
                raise IsolationForestAdapterError(
                    "integer max_samples must be positive"
                )
        elif isinstance(max_samples, Decimal):
            _fraction(max_samples, "max_samples")
        else:
            raise IsolationForestAdapterError(
                "max_samples must be auto, positive int, or Decimal fraction"
            )

        max_features = self.max_features
        if isinstance(max_features, bool):
            raise IsolationForestAdapterError(
                "max_features must be positive int or Decimal fraction"
            )
        if isinstance(max_features, int):
            if max_features < 1:
                raise IsolationForestAdapterError(
                    "integer max_features must be positive"
                )
        elif isinstance(max_features, Decimal):
            _fraction(max_features, "max_features")
        else:
            raise IsolationForestAdapterError(
                "max_features must be positive int or Decimal fraction"
            )

        if not isinstance(self.bootstrap, bool):
            raise IsolationForestAdapterError("bootstrap must be bool")
        if (
            isinstance(self.random_state, bool)
            or not isinstance(self.random_state, int)
            or not 0 <= self.random_state < 2**32
        ):
            raise IsolationForestAdapterError(
                "random_state must be an integer in [0, 2**32)"
            )
        if self.n_jobs is not None and (
            isinstance(self.n_jobs, bool)
            or not isinstance(self.n_jobs, int)
            or self.n_jobs == 0
        ):
            raise IsolationForestAdapterError(
                "n_jobs must be None or a non-zero integer"
            )

    @property
    def sha256_hex(self) -> str:
        return _digest(
            {
                "n_estimators": self.n_estimators,
                "max_samples": _parameter_text(self.max_samples),
                "max_features": _parameter_text(self.max_features),
                "bootstrap": self.bootstrap,
                "random_state": self.random_state,
                "n_jobs": self.n_jobs,
                "contamination": "auto",
                "warm_start": False,
                "verbose": 0,
            }
        )


@dataclass(frozen=True, slots=True)
class Float32ConversionDiagnostic:
    feature_name: str
    value_count: int
    nonzero_conversion_error_count: int
    maximum_absolute_conversion_error: Decimal

    def __post_init__(self) -> None:
        name = self.feature_name.strip()
        if not name:
            raise IsolationForestAdapterError("feature_name must not be blank")
        object.__setattr__(self, "feature_name", name)
        for field_name in ("value_count", "nonzero_conversion_error_count"):
            value = getattr(self, field_name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
            ):
                raise IsolationForestAdapterError(
                    f"{field_name} must be a non-negative integer"
                )
        if self.value_count < 1:
            raise IsolationForestAdapterError(
                "float32 conversion diagnostic requires values"
            )
        if self.nonzero_conversion_error_count > self.value_count:
            raise IsolationForestAdapterError(
                "conversion error count cannot exceed value count"
            )
        error = self.maximum_absolute_conversion_error
        if (
            not isinstance(error, Decimal)
            or not error.is_finite()
            or error < 0
        ):
            raise IsolationForestAdapterError(
                "maximum conversion error must be non-negative Decimal"
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "feature_name": self.feature_name,
            "value_count": self.value_count,
            "nonzero_conversion_error_count":
                self.nonzero_conversion_error_count,
            "maximum_absolute_conversion_error":
                str(self.maximum_absolute_conversion_error),
        }


@dataclass(frozen=True, slots=True)
class IsolationForestInput:
    """Exact float32 matrix delivered to scikit-learn."""

    source_matrix_sha256: str
    preprocessor_sha256: str
    output_feature_names: tuple[str, ...]
    row_identities: tuple[tuple[str, str], ...]
    values: np.ndarray = field(repr=False, compare=False)
    conversion_diagnostics: tuple[Float32ConversionDiagnostic, ...]

    def __post_init__(self) -> None:
        for field_name in ("source_matrix_sha256", "preprocessor_sha256"):
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
            raise IsolationForestAdapterError(
                "output_feature_names must be non-empty and unique"
            )
        identities = tuple(self.row_identities)
        if (
            not identities
            or len(identities) != len(set(identities))
            or any(
                not isinstance(item, tuple)
                or len(item) != 2
                or not item[0].strip()
                or not item[1].strip()
                for item in identities
            )
        ):
            raise IsolationForestAdapterError(
                "row_identities must be unique non-blank pairs"
            )
        matrix = self.values
        if (
            not isinstance(matrix, np.ndarray)
            or matrix.dtype != np.dtype("float32")
            or matrix.ndim != 2
            or matrix.shape != (len(identities), len(names))
            or not np.isfinite(matrix).all()
        ):
            raise IsolationForestAdapterError(
                "IsolationForestInput values must be finite float32 2-D matrix"
            )
        diagnostics = tuple(self.conversion_diagnostics)
        if tuple(item.feature_name for item in diagnostics) != names:
            raise IsolationForestAdapterError(
                "conversion diagnostics must match feature order"
            )
        if any(item.value_count != len(identities) for item in diagnostics):
            raise IsolationForestAdapterError(
                "conversion diagnostics must cover every row"
            )
        object.__setattr__(self, "output_feature_names", names)
        object.__setattr__(self, "row_identities", identities)
        object.__setattr__(self, "conversion_diagnostics", diagnostics)

    @property
    def row_count(self) -> int:
        return self.values.shape[0]

    @property
    def feature_count(self) -> int:
        return self.values.shape[1]

    @property
    def sha256_hex(self) -> str:
        canonical = np.asarray(self.values, dtype="<f4", order="C")
        hasher = sha256()
        _hash_json(
            hasher,
            {
                "source_matrix_sha256": self.source_matrix_sha256,
                "preprocessor_sha256": self.preprocessor_sha256,
                "feature_names": list(self.output_feature_names),
                "row_identities": self.row_identities,
                "shape": list(canonical.shape),
                "conversion_diagnostics": [
                    item.as_dict() for item in self.conversion_diagnostics
                ],
            },
        )
        hasher.update(canonical.tobytes(order="C"))
        return hasher.hexdigest()


@dataclass(frozen=True, slots=True)
class FittedIsolationForest:
    """Fitted sklearn IsolationForest plus auditable immutable metadata."""

    config: IsolationForestConfig
    sklearn_version: str
    numpy_version: str
    training_input_sha256: str
    training_source_matrix_sha256: str
    preprocessor_sha256: str
    output_feature_names: tuple[str, ...]
    actual_max_samples: int
    fitted_model_sha256: str
    estimator: Any = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.config, IsolationForestConfig):
            raise TypeError("config must be IsolationForestConfig")
        for field_name in ("sklearn_version", "numpy_version"):
            text = getattr(self, field_name).strip()
            if not text:
                raise IsolationForestAdapterError(
                    f"{field_name} must not be blank"
                )
            object.__setattr__(self, field_name, text)
        for field_name in (
            "training_input_sha256",
            "training_source_matrix_sha256",
            "preprocessor_sha256",
            "fitted_model_sha256",
        ):
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
            raise IsolationForestAdapterError(
                "output_feature_names must be non-empty and unique"
            )
        object.__setattr__(self, "output_feature_names", names)
        if (
            isinstance(self.actual_max_samples, bool)
            or not isinstance(self.actual_max_samples, int)
            or self.actual_max_samples < 1
        ):
            raise IsolationForestAdapterError(
                "actual_max_samples must be positive integer"
            )
        if not isinstance(self.estimator, IsolationForest):
            raise TypeError("estimator must be sklearn IsolationForest")
        if getattr(self.estimator, "n_features_in_", None) != len(names):
            raise IsolationForestAdapterError(
                "fitted estimator feature count differs from metadata"
            )
        if getattr(self.estimator, "max_samples_", None) != self.actual_max_samples:
            raise IsolationForestAdapterError(
                "fitted estimator max_samples differs from metadata"
            )

    @property
    def config_sha256(self) -> str:
        return self.config.sha256_hex


def fit_isolation_forest(
    matrix: PreprocessedMatrix,
    config: IsolationForestConfig,
) -> FittedIsolationForest:
    """Fit standard IsolationForest on the explicitly converted training matrix."""

    if not isinstance(matrix, PreprocessedMatrix):
        raise TypeError("matrix must be PreprocessedMatrix")
    if not isinstance(config, IsolationForestConfig):
        raise TypeError("config must be IsolationForestConfig")
    if matrix.row_count < 2:
        raise IsolationForestAdapterError(
            "Isolation Forest training requires at least two rows"
        )

    converted = _to_float32(matrix)
    max_samples = _resolve_parameter(config.max_samples)
    max_features = _resolve_parameter(config.max_features)

    if isinstance(max_samples, int) and max_samples > converted.row_count:
        raise IsolationForestAdapterError(
            "integer max_samples exceeds training row count"
        )
    if isinstance(max_samples, float):
        resolved_samples = int(max_samples * converted.row_count)
        if resolved_samples < 1:
            raise IsolationForestAdapterError(
                "fractional max_samples resolves to fewer than one training row"
            )
    if isinstance(max_features, int) and max_features > converted.feature_count:
        raise IsolationForestAdapterError(
            "integer max_features exceeds feature count"
        )
    if isinstance(max_features, float):
        resolved_features = max(1, int(max_features * converted.feature_count))
        if resolved_features < 1:
            raise IsolationForestAdapterError(
                "fractional max_features resolves to no features"
            )

    estimator = IsolationForest(
        n_estimators=config.n_estimators,
        max_samples=max_samples,
        contamination="auto",
        max_features=max_features,
        bootstrap=config.bootstrap,
        n_jobs=config.n_jobs,
        random_state=config.random_state,
        verbose=0,
        warm_start=False,
    )
    estimator.fit(converted.values)
    model_sha = _fingerprint_model(
        estimator=estimator,
        config=config,
        training_input_sha256=converted.sha256_hex,
        feature_names=converted.output_feature_names,
    )
    return FittedIsolationForest(
        config=config,
        sklearn_version=sklearn.__version__,
        numpy_version=np.__version__,
        training_input_sha256=converted.sha256_hex,
        training_source_matrix_sha256=matrix.evidence_sha256,
        preprocessor_sha256=matrix.preprocessor_sha256,
        output_feature_names=matrix.output_feature_names,
        actual_max_samples=int(estimator.max_samples_),
        fitted_model_sha256=model_sha,
        estimator=estimator,
    )


def score_isolation_forest(
    fitted: FittedIsolationForest,
    matrix: PreprocessedMatrix,
) -> DetectorScoreBatch:
    """Score rows with native score_samples; lower native scores are more abnormal."""

    if not isinstance(fitted, FittedIsolationForest):
        raise TypeError("fitted must be FittedIsolationForest")
    if not isinstance(matrix, PreprocessedMatrix):
        raise TypeError("matrix must be PreprocessedMatrix")
    if matrix.preprocessor_sha256 != fitted.preprocessor_sha256:
        raise IsolationForestAdapterError(
            "scoring matrix preprocessor differs from fitted detector"
        )
    if matrix.output_feature_names != fitted.output_feature_names:
        raise IsolationForestAdapterError(
            "scoring matrix feature order differs from fitted detector"
        )

    converted = _to_float32(matrix)
    raw = fitted.estimator.score_samples(converted.values)
    raw_scores = tuple(float(value) for value in np.asarray(raw, dtype=np.float64))
    scores = build_detector_scores(
        converted.row_identities,
        raw_scores,
        orientation=ScoreOrientation.LOWER_IS_MORE_ANOMALOUS,
    )
    return DetectorScoreBatch(
        detector_name="sklearn_isolation_forest",
        detector_family="isolation_forest",
        implementation_name="scikit-learn",
        implementation_version=fitted.sklearn_version,
        score_orientation=ScoreOrientation.LOWER_IS_MORE_ANOMALOUS,
        config_sha256=fitted.config_sha256,
        fitted_model_sha256=fitted.fitted_model_sha256,
        training_matrix_sha256=fitted.training_input_sha256,
        scoring_matrix_sha256=converted.sha256_hex,
        preprocessor_sha256=fitted.preprocessor_sha256,
        output_feature_names=fitted.output_feature_names,
        scores=scores,
    )


def _to_float32(matrix: PreprocessedMatrix) -> IsolationForestInput:
    values = np.empty((matrix.row_count, matrix.feature_count), dtype=np.float32)
    error_counts = [0] * matrix.feature_count
    max_errors = [Decimal(0)] * matrix.feature_count

    for row_index, row in enumerate(matrix.values):
        for feature_index, value in enumerate(row):
            with np.errstate(over="ignore", invalid="ignore"):
                converted = np.float32(float(value))
            if not np.isfinite(converted):
                raise IsolationForestAdapterError(
                    f"{matrix.output_feature_names[feature_index]}: "
                    "value cannot be represented as finite float32"
                )
            roundtrip = Decimal.from_float(float(converted))
            error = abs(roundtrip - value)
            if error != 0:
                error_counts[feature_index] += 1
                if error > max_errors[feature_index]:
                    max_errors[feature_index] = error
            values[row_index, feature_index] = converted

    diagnostics = tuple(
        Float32ConversionDiagnostic(
            feature_name=name,
            value_count=matrix.row_count,
            nonzero_conversion_error_count=error_counts[index],
            maximum_absolute_conversion_error=max_errors[index],
        )
        for index, name in enumerate(matrix.output_feature_names)
    )
    return IsolationForestInput(
        source_matrix_sha256=matrix.evidence_sha256,
        preprocessor_sha256=matrix.preprocessor_sha256,
        output_feature_names=matrix.output_feature_names,
        row_identities=matrix.row_identities,
        values=values,
        conversion_diagnostics=diagnostics,
    )


def _fingerprint_model(
    *,
    estimator: IsolationForest,
    config: IsolationForestConfig,
    training_input_sha256: str,
    feature_names: tuple[str, ...],
) -> str:
    hasher = sha256()
    _hash_json(
        hasher,
        {
            "algorithm": "sklearn_isolation_forest",
            "sklearn_version": sklearn.__version__,
            "numpy_version": np.__version__,
            "config_sha256": config.sha256_hex,
            "training_input_sha256": training_input_sha256,
            "feature_names": list(feature_names),
            "n_features_in": int(estimator.n_features_in_),
            "max_samples": int(estimator.max_samples_),
            "offset_hex": float(estimator.offset_).hex(),
            "tree_count": len(estimator.estimators_),
        },
    )

    for index, tree_estimator in enumerate(estimator.estimators_):
        tree = tree_estimator.tree_
        features = np.asarray(estimator.estimators_features_[index], dtype="<i8")
        _hash_json(
            hasher,
            {
                "tree_index": index,
                "random_state": int(tree_estimator.random_state),
                "node_count": int(tree.node_count),
                "max_depth": int(tree.max_depth),
            },
        )
        hasher.update(features.tobytes(order="C"))
        for array, dtype in (
            (tree.children_left, "<i8"),
            (tree.children_right, "<i8"),
            (tree.feature, "<i8"),
            (tree.threshold, "<f8"),
            (tree.n_node_samples, "<i8"),
            (tree.weighted_n_node_samples, "<f8"),
        ):
            canonical = np.asarray(array, dtype=dtype, order="C")
            hasher.update(canonical.tobytes(order="C"))
    return hasher.hexdigest()


def _resolve_parameter(value: str | int | Decimal) -> str | int | float:
    if isinstance(value, Decimal):
        converted = float(value)
        if not math.isfinite(converted):
            raise IsolationForestAdapterError(
                "fractional parameter cannot be represented as finite float"
            )
        return converted
    return value


def _fraction(value: Decimal, name: str) -> None:
    if not value.is_finite() or value <= 0 or value > 1:
        raise IsolationForestAdapterError(
            f"{name} Decimal fraction must be in (0, 1]"
        )


def _parameter_text(value: str | int | Decimal) -> str | int:
    return str(value) if isinstance(value, Decimal) else value


def _digest_hex(value: str, name: str) -> str:
    digest = value.strip().lower()
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise IsolationForestAdapterError(
            f"{name} must be a SHA-256 hex digest"
        )
    return digest


def _hash_json(hasher: Any, value: Any) -> None:
    hasher.update(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    )


def _digest(value: Any) -> str:
    hasher = sha256()
    _hash_json(hasher, value)
    return hasher.hexdigest()
