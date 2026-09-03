"""Explicit ordered model-feature selection contracts for ProcureLens.

A feature specification pins an ordered subset of the unified candidate catalog
without choosing preprocessing, imputation, detector hyperparameters, anomaly
thresholds, or risk policy. Catalog fingerprints make stale selections fail
loudly when upstream feature contracts change.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Any

from procurelens.features.catalog import (
    FeatureCatalog,
    FeatureCatalogEntry,
    FeatureCatalogError,
    FeatureSource,
    feature_catalog,
)


class FeatureSpecError(ValueError):
    """Raised when a model feature-selection contract is inconsistent."""


@dataclass(frozen=True, slots=True)
class FeatureSelectionSpec:
    name: str
    description: str
    catalog_sha256: str
    feature_names: tuple[str, ...]

    def __post_init__(self) -> None:
        for field_name in ("name", "description"):
            text = getattr(self, field_name).strip()
            if not text:
                raise FeatureSpecError(f"{field_name} must not be blank")
            object.__setattr__(self, field_name, text)
        object.__setattr__(
            self,
            "catalog_sha256",
            _validate_digest(self.catalog_sha256, "catalog_sha256"),
        )

        names = tuple(
            _validate_feature_name(item)
            for item in self.feature_names
        )
        if not names:
            raise FeatureSpecError(
                "feature selection must contain at least one feature"
            )
        if len(names) != len(set(names)):
            duplicates = sorted(
                {name for name in names if names.count(name) > 1}
            )
            raise FeatureSpecError(
                "feature selection contains duplicates: "
                + ", ".join(duplicates)
            )
        object.__setattr__(self, "feature_names", names)

    @property
    def feature_count(self) -> int:
        return len(self.feature_names)

    @property
    def sha256_hex(self) -> str:
        """Fingerprint includes column order because order is model input state."""
        return _digest(
            {
                "name": self.name,
                "description": self.description,
                "catalog_sha256": self.catalog_sha256,
                "feature_names": self.feature_names,
            }
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "catalog_sha256": self.catalog_sha256,
            "feature_count": self.feature_count,
            "feature_names": list(self.feature_names),
            "spec_sha256": self.sha256_hex,
        }


@dataclass(frozen=True, slots=True)
class ResolvedFeatureSelection:
    spec: FeatureSelectionSpec
    entries: tuple[FeatureCatalogEntry, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.spec, FeatureSelectionSpec):
            raise TypeError("spec must be FeatureSelectionSpec")
        entries = tuple(self.entries)
        if len(entries) != self.spec.feature_count:
            raise FeatureSpecError(
                "resolved entry count differs from feature specification"
            )
        observed = tuple(entry.name for entry in entries)
        if observed != self.spec.feature_names:
            raise FeatureSpecError(
                "resolved entries must exactly preserve specified feature order"
            )
        object.__setattr__(self, "entries", entries)

    @property
    def feature_names(self) -> tuple[str, ...]:
        return self.spec.feature_names

    @property
    def feature_count(self) -> int:
        return self.spec.feature_count

    @property
    def sources(self) -> tuple[FeatureSource, ...]:
        seen: set[FeatureSource] = set()
        ordered: list[FeatureSource] = []
        for entry in self.entries:
            if entry.source not in seen:
                seen.add(entry.source)
                ordered.append(entry.source)
        return tuple(ordered)

    @property
    def source_feature_counts(self) -> dict[str, int]:
        result: dict[str, int] = {}
        for entry in self.entries:
            result[entry.source.value] = (
                result.get(entry.source.value, 0) + 1
            )
        return result

    @property
    def sha256_hex(self) -> str:
        return _digest(
            {
                "spec_sha256": self.spec.sha256_hex,
                "entries": [
                    {
                        "name": entry.name,
                        "entry_sha256": entry.sha256_hex,
                    }
                    for entry in self.entries
                ],
            }
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "spec": self.spec.as_dict(),
            "resolved_sha256": self.sha256_hex,
            "sources": [source.value for source in self.sources],
            "source_feature_counts": self.source_feature_counts,
            "entries": [entry.as_dict() for entry in self.entries],
        }


def make_feature_selection_spec(
    *,
    name: str,
    description: str,
    feature_names: Iterable[str],
    catalog: FeatureCatalog | None = None,
) -> FeatureSelectionSpec:
    """Create a spec pinned to the current supplied catalog."""

    catalog = feature_catalog() if catalog is None else catalog
    if not isinstance(catalog, FeatureCatalog):
        raise TypeError("catalog must be FeatureCatalog")
    if isinstance(feature_names, (str, bytes)):
        raise FeatureSpecError("feature_names must be an iterable")
    names = tuple(feature_names)
    try:
        catalog.select(names)
    except FeatureCatalogError as exc:
        raise FeatureSpecError(str(exc)) from exc
    return FeatureSelectionSpec(
        name=name,
        description=description,
        catalog_sha256=catalog.sha256_hex,
        feature_names=names,
    )


def resolve_feature_selection(
    spec: FeatureSelectionSpec,
    *,
    catalog: FeatureCatalog | None = None,
) -> ResolvedFeatureSelection:
    """Resolve and cross-check an ordered selection against its pinned catalog."""

    if not isinstance(spec, FeatureSelectionSpec):
        raise TypeError("spec must be FeatureSelectionSpec")
    catalog = feature_catalog() if catalog is None else catalog
    if not isinstance(catalog, FeatureCatalog):
        raise TypeError("catalog must be FeatureCatalog")
    if spec.catalog_sha256 != catalog.sha256_hex:
        raise FeatureSpecError(
            "feature specification catalog fingerprint differs from current catalog"
        )
    try:
        entries = catalog.select(spec.feature_names)
    except FeatureCatalogError as exc:
        raise FeatureSpecError(str(exc)) from exc
    return ResolvedFeatureSelection(spec=spec, entries=entries)


def _validate_feature_name(value: str) -> str:
    if not isinstance(value, str):
        raise FeatureSpecError("feature names must be strings")
    text = value.strip()
    if not text:
        raise FeatureSpecError("feature names must not be blank")
    return text


def _validate_digest(value: str, name: str) -> str:
    digest = value.strip().lower()
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise FeatureSpecError(f"{name} must be a SHA-256 hex digest")
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
