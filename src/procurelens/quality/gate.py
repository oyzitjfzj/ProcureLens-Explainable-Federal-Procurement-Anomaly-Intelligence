"""Configurable analysis-readiness gates for ProcureLens data quality.

`profile.py` measures the data. This module decides whether those measurements
are sufficient for a *specific* analysis.

There is deliberately no universal quality score and no baked-in "80% is good"
rule. Callers define named requirements for each analysis. A blocking failure
stops only that analysis; warnings can allow it to run with visible limitations.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from enum import Enum
from types import MappingProxyType
from typing import Any, TypeAlias

from procurelens.quality.profile import Coverage, NamedCount, QualityProfile


class QualityGateConfigurationError(ValueError):
    """Raised when an analysis gate is ambiguous or internally inconsistent."""


class QualityGateEvaluationError(RuntimeError):
    """Raised when a configured metric cannot be resolved from a profile."""


class GateSeverity(str, Enum):
    WARNING = "warning"
    BLOCK = "block"


class GateStatus(str, Enum):
    READY = "ready"
    DEGRADED = "degraded"
    BLOCKED = "blocked"


class CoverageKind(str, Enum):
    FIELD = "field"
    CONTEXT = "context"
    COMPLETE_ANALYSIS_CONTEXT = "complete_analysis_context"


class BreakdownKind(str, Enum):
    VENDOR_IDENTITY = "vendor_identity"
    CATEGORY_IDENTITY = "category_identity"
    AGENCY_IDENTITY = "agency_identity"
    COMPETITION_CORE = "competition_core"
    OBLIGATION_SIGN = "obligation_sign"


class CardinalityKind(str, Enum):
    SOURCE = "source"
    SOURCE_SCHEMA = "source_schema"
    ACTION_FISCAL_YEAR = "action_fiscal_year"


_BREAKDOWN_MODES: Mapping[BreakdownKind, frozenset[str]] = MappingProxyType({
    BreakdownKind.VENDOR_IDENTITY: frozenset({"uei", "legacy_identifier", "name_only", "missing"}),
    BreakdownKind.CATEGORY_IDENTITY: frozenset({"psc_and_naics", "psc_only", "naics_only", "missing"}),
    BreakdownKind.AGENCY_IDENTITY: frozenset({"code_and_name", "code_only", "name_only", "missing"}),
    BreakdownKind.COMPETITION_CORE: frozenset({"extent_and_offers", "extent_only", "offers_only", "missing"}),
    BreakdownKind.OBLIGATION_SIGN: frozenset({"positive", "negative", "zero"}),
})


def _decimal_rate(value: Decimal | float | int | str, name: str) -> Decimal:
    if isinstance(value, bool):
        raise QualityGateConfigurationError(f"{name} must be a number from 0 to 1")
    try:
        parsed = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise QualityGateConfigurationError(f"{name} must be a finite number from 0 to 1") from exc
    if not parsed.is_finite() or not (Decimal(0) <= parsed <= Decimal(1)):
        raise QualityGateConfigurationError(f"{name} must be between 0 and 1")
    return parsed


def _clean_identifier(value: str | None, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise QualityGateConfigurationError(f"{name} must be non-blank text")
    return value.strip()


def _clean_text_tuple(values: Sequence[str], name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise QualityGateConfigurationError(f"{name} must be a sequence of strings")
    cleaned: list[str] = []
    seen: set[str] = set()
    for raw in values:
        value = _clean_identifier(raw, name)
        if value not in seen:
            cleaned.append(value)
            seen.add(value)
    if not cleaned:
        raise QualityGateConfigurationError(f"{name} must contain at least one value")
    return tuple(cleaned)


def _validate_nonnegative_int(value: int | None, name: str) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise QualityGateConfigurationError(f"{name} must be a non-negative integer or None")


def _rate(present: int, total: int) -> Decimal | None:
    return None if total == 0 else Decimal(present) / Decimal(total)


def _rate_text(value: Decimal | None) -> str | None:
    if value is None:
        return None
    return format(value.quantize(Decimal("0.000001")), "f").rstrip("0").rstrip(".")


def _pct(value: Decimal | None) -> str:
    if value is None:
        return "no rows available"
    return f"{(value * Decimal(100)).quantize(Decimal('0.1'))}%"


@dataclass(frozen=True, slots=True)
class CoverageMetric:
    kind: CoverageKind
    name: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", CoverageKind(self.kind))
        if self.kind is CoverageKind.COMPLETE_ANALYSIS_CONTEXT:
            if self.name is not None:
                raise QualityGateConfigurationError("complete-analysis-context coverage does not take a name")
            return
        object.__setattr__(self, "name", _clean_identifier(self.name, "coverage metric name"))

    @property
    def label(self) -> str:
        return "complete_analysis_context" if self.kind is CoverageKind.COMPLETE_ANALYSIS_CONTEXT else f"{self.kind.value}:{self.name}"

    def resolve(self, profile: QualityProfile) -> Coverage:
        if self.kind is CoverageKind.COMPLETE_ANALYSIS_CONTEXT:
            return profile.complete_analysis_context_coverage
        source = profile.field_coverage if self.kind is CoverageKind.FIELD else profile.context_coverage
        assert self.name is not None
        try:
            return source[self.name]
        except KeyError as exc:
            raise QualityGateEvaluationError(f"profile does not expose configured coverage metric {self.label!r}") from exc


@dataclass(frozen=True, slots=True)
class GateCheck:
    requirement_id: str
    severity: GateSeverity
    passed: bool
    metric: str
    expectation: str
    observed: str
    message: str
    evidence: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "requirement_id", _clean_identifier(self.requirement_id, "requirement_id"))
        object.__setattr__(self, "severity", GateSeverity(self.severity))
        object.__setattr__(self, "metric", _clean_identifier(self.metric, "metric"))
        object.__setattr__(self, "expectation", _clean_identifier(self.expectation, "expectation"))
        object.__setattr__(self, "observed", _clean_identifier(self.observed, "observed"))
        object.__setattr__(self, "message", _clean_identifier(self.message, "message"))
        object.__setattr__(self, "evidence", MappingProxyType(dict(self.evidence)))

    def as_dict(self) -> dict[str, Any]:
        return {
            "requirement_id": self.requirement_id,
            "severity": self.severity.value,
            "passed": self.passed,
            "metric": self.metric,
            "expectation": self.expectation,
            "observed": self.observed,
            "message": self.message,
            "evidence": dict(self.evidence),
        }


@dataclass(frozen=True, slots=True)
class PopulationRequirement:
    requirement_id: str
    minimum_transactions: int
    severity: GateSeverity = GateSeverity.BLOCK
    description: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "requirement_id", _clean_identifier(self.requirement_id, "requirement_id"))
        _validate_nonnegative_int(self.minimum_transactions, "minimum_transactions")
        object.__setattr__(self, "severity", GateSeverity(self.severity))
        if self.description is not None:
            object.__setattr__(self, "description", _clean_identifier(self.description, "description"))

    def evaluate(self, profile: QualityProfile) -> GateCheck:
        actual = profile.total_transactions
        passed = actual >= self.minimum_transactions
        return GateCheck(
            self.requirement_id, self.severity, passed, "population_size",
            f"at least {self.minimum_transactions} transactions", f"{actual} transactions",
            self.description or ("Enough transactions are available for this analysis." if passed else "This analysis does not have enough transactions."),
            {"actual_transactions": actual, "minimum_transactions": self.minimum_transactions},
        )


@dataclass(frozen=True, slots=True)
class CoverageRequirement:
    requirement_id: str
    metric: CoverageMetric
    minimum_rate: Decimal | float | int | str | None = None
    minimum_present: int | None = None
    severity: GateSeverity = GateSeverity.BLOCK
    description: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "requirement_id", _clean_identifier(self.requirement_id, "requirement_id"))
        if not isinstance(self.metric, CoverageMetric):
            raise QualityGateConfigurationError("metric must be CoverageMetric")
        if self.minimum_rate is None and self.minimum_present is None:
            raise QualityGateConfigurationError("coverage requirement needs minimum_rate and/or minimum_present")
        if self.minimum_rate is not None:
            object.__setattr__(self, "minimum_rate", _decimal_rate(self.minimum_rate, "minimum_rate"))
        _validate_nonnegative_int(self.minimum_present, "minimum_present")
        object.__setattr__(self, "severity", GateSeverity(self.severity))
        if self.description is not None:
            object.__setattr__(self, "description", _clean_identifier(self.description, "description"))

    def evaluate(self, profile: QualityProfile) -> GateCheck:
        coverage = self.metric.resolve(profile)
        actual_rate = _rate(coverage.present, coverage.total)
        rate_ok = self.minimum_rate is None or (actual_rate is not None and actual_rate >= self.minimum_rate)
        count_ok = self.minimum_present is None or coverage.present >= self.minimum_present
        passed = rate_ok and count_ok
        clauses: list[str] = []
        if self.minimum_rate is not None:
            clauses.append(f"coverage at least {_pct(self.minimum_rate)}")
        if self.minimum_present is not None:
            clauses.append(f"at least {self.minimum_present} present rows")
        observed = f"{coverage.present}/{coverage.total} present" + (f" ({_pct(actual_rate)})" if actual_rate is not None else " (no population)")
        return GateCheck(
            self.requirement_id, self.severity, passed, self.metric.label, " and ".join(clauses), observed,
            self.description or ("Required data coverage is sufficient for this analysis." if passed else "Required data coverage is not sufficient for this analysis."),
            {
                "present": coverage.present, "missing": coverage.missing, "total": coverage.total,
                "actual_rate": _rate_text(actual_rate),
                "minimum_rate": _rate_text(self.minimum_rate) if isinstance(self.minimum_rate, Decimal) else None,
                "minimum_present": self.minimum_present,
            },
        )


@dataclass(frozen=True, slots=True)
class BreakdownShareRequirement:
    requirement_id: str
    breakdown: BreakdownKind
    modes: tuple[str, ...]
    minimum_rate: Decimal | float | int | str | None = None
    maximum_rate: Decimal | float | int | str | None = None
    minimum_count: int | None = None
    maximum_count: int | None = None
    severity: GateSeverity = GateSeverity.BLOCK
    description: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "requirement_id", _clean_identifier(self.requirement_id, "requirement_id"))
        object.__setattr__(self, "breakdown", BreakdownKind(self.breakdown))
        object.__setattr__(self, "modes", _clean_text_tuple(self.modes, "modes"))
        unknown = set(self.modes) - _BREAKDOWN_MODES[self.breakdown]
        if unknown:
            raise QualityGateConfigurationError(f"unknown modes for {self.breakdown.value}: {', '.join(sorted(unknown))}")
        if all(v is None for v in (self.minimum_rate, self.maximum_rate, self.minimum_count, self.maximum_count)):
            raise QualityGateConfigurationError("breakdown-share requirement needs at least one bound")
        if self.minimum_rate is not None:
            object.__setattr__(self, "minimum_rate", _decimal_rate(self.minimum_rate, "minimum_rate"))
        if self.maximum_rate is not None:
            object.__setattr__(self, "maximum_rate", _decimal_rate(self.maximum_rate, "maximum_rate"))
        if isinstance(self.minimum_rate, Decimal) and isinstance(self.maximum_rate, Decimal) and self.minimum_rate > self.maximum_rate:
            raise QualityGateConfigurationError("minimum_rate cannot exceed maximum_rate")
        _validate_nonnegative_int(self.minimum_count, "minimum_count")
        _validate_nonnegative_int(self.maximum_count, "maximum_count")
        if self.minimum_count is not None and self.maximum_count is not None and self.minimum_count > self.maximum_count:
            raise QualityGateConfigurationError("minimum_count cannot exceed maximum_count")
        object.__setattr__(self, "severity", GateSeverity(self.severity))
        if self.description is not None:
            object.__setattr__(self, "description", _clean_identifier(self.description, "description"))

    def evaluate(self, profile: QualityProfile) -> GateCheck:
        items = _breakdown(profile, self.breakdown)
        counts = {item.name: item.count for item in items}
        selected = sum(counts.get(mode, 0) for mode in self.modes)
        total = profile.total_transactions
        actual_rate = _rate(selected, total)
        passed = True
        if self.minimum_rate is not None:
            passed = passed and actual_rate is not None and actual_rate >= self.minimum_rate
        if self.maximum_rate is not None:
            passed = passed and actual_rate is not None and actual_rate <= self.maximum_rate
        if self.minimum_count is not None:
            passed = passed and selected >= self.minimum_count
        if self.maximum_count is not None:
            passed = passed and selected <= self.maximum_count
        expectations: list[str] = []
        if self.minimum_rate is not None: expectations.append(f"share at least {_pct(self.minimum_rate)}")
        if self.maximum_rate is not None: expectations.append(f"share at most {_pct(self.maximum_rate)}")
        if self.minimum_count is not None: expectations.append(f"count at least {self.minimum_count}")
        if self.maximum_count is not None: expectations.append(f"count at most {self.maximum_count}")
        observed = f"{selected}/{total} rows in {', '.join(self.modes)}" + (f" ({_pct(actual_rate)})" if actual_rate is not None else " (no population)")
        return GateCheck(
            self.requirement_id, self.severity, passed, f"breakdown:{self.breakdown.value}", " and ".join(expectations), observed,
            self.description or ("The requested data mix is sufficient for this analysis." if passed else "The requested data mix is not sufficient for this analysis."),
            {"selected_modes": list(self.modes), "selected_count": selected, "total": total, "actual_rate": _rate_text(actual_rate), "available_modes": counts},
        )


@dataclass(frozen=True, slots=True)
class CardinalityRequirement:
    requirement_id: str
    metric: CardinalityKind
    minimum_distinct: int | None = None
    maximum_distinct: int | None = None
    severity: GateSeverity = GateSeverity.BLOCK
    description: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "requirement_id", _clean_identifier(self.requirement_id, "requirement_id"))
        object.__setattr__(self, "metric", CardinalityKind(self.metric))
        if self.minimum_distinct is None and self.maximum_distinct is None:
            raise QualityGateConfigurationError("cardinality requirement needs a minimum and/or maximum")
        _validate_nonnegative_int(self.minimum_distinct, "minimum_distinct")
        _validate_nonnegative_int(self.maximum_distinct, "maximum_distinct")
        if self.minimum_distinct is not None and self.maximum_distinct is not None and self.minimum_distinct > self.maximum_distinct:
            raise QualityGateConfigurationError("minimum_distinct cannot exceed maximum_distinct")
        object.__setattr__(self, "severity", GateSeverity(self.severity))
        if self.description is not None:
            object.__setattr__(self, "description", _clean_identifier(self.description, "description"))

    def evaluate(self, profile: QualityProfile) -> GateCheck:
        items = _cardinality_items(profile, self.metric)
        actual = len(items)
        passed = True
        expectations: list[str] = []
        if self.minimum_distinct is not None:
            passed = passed and actual >= self.minimum_distinct
            expectations.append(f"at least {self.minimum_distinct} distinct")
        if self.maximum_distinct is not None:
            passed = passed and actual <= self.maximum_distinct
            expectations.append(f"at most {self.maximum_distinct} distinct")
        return GateCheck(
            self.requirement_id, self.severity, passed, f"cardinality:{self.metric.value}", " and ".join(expectations), f"{actual} distinct",
            self.description or ("The data variety is within the configured range." if passed else "The data variety is outside the configured range."),
            {"distinct_count": actual, "values": [{"name": item.name, "count": item.count} for item in items]},
        )


QualityRequirement: TypeAlias = PopulationRequirement | CoverageRequirement | BreakdownShareRequirement | CardinalityRequirement


@dataclass(frozen=True, slots=True)
class QualityGateSpec:
    analysis_name: str
    requirements: tuple[QualityRequirement, ...]
    description: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "analysis_name", _clean_identifier(self.analysis_name, "analysis_name"))
        if not self.requirements:
            raise QualityGateConfigurationError("an analysis gate must contain at least one requirement")
        cleaned = tuple(self.requirements)
        supported = (PopulationRequirement, CoverageRequirement, BreakdownShareRequirement, CardinalityRequirement)
        if any(not isinstance(item, supported) for item in cleaned):
            raise QualityGateConfigurationError("requirements contain an unsupported requirement type")
        ids = [item.requirement_id for item in cleaned]
        if len(ids) != len(set(ids)):
            raise QualityGateConfigurationError("requirement_id values must be unique inside an analysis gate")
        object.__setattr__(self, "requirements", cleaned)
        if self.description is not None:
            object.__setattr__(self, "description", _clean_identifier(self.description, "description"))


@dataclass(frozen=True, slots=True)
class QualityGateReport:
    analysis_name: str
    status: GateStatus
    checks: tuple[GateCheck, ...]
    total_transactions: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "analysis_name", _clean_identifier(self.analysis_name, "analysis_name"))
        object.__setattr__(self, "status", GateStatus(self.status))
        _validate_nonnegative_int(self.total_transactions, "total_transactions")
        if not self.checks:
            raise QualityGateEvaluationError("gate report must contain checks")
        expected = _status_from_checks(self.checks)
        if self.status is not expected:
            raise QualityGateEvaluationError("gate report status is inconsistent with its checks")

    @property
    def allowed(self) -> bool:
        return self.status is not GateStatus.BLOCKED

    @property
    def fully_ready(self) -> bool:
        return self.status is GateStatus.READY

    @property
    def blocked_checks(self) -> tuple[GateCheck, ...]:
        return tuple(check for check in self.checks if not check.passed and check.severity is GateSeverity.BLOCK)

    @property
    def warnings(self) -> tuple[GateCheck, ...]:
        return tuple(check for check in self.checks if not check.passed and check.severity is GateSeverity.WARNING)

    def as_dict(self) -> dict[str, Any]:
        return {
            "analysis_name": self.analysis_name, "status": self.status.value,
            "allowed": self.allowed, "fully_ready": self.fully_ready,
            "total_transactions": self.total_transactions,
            "checks": [check.as_dict() for check in self.checks],
        }


class QualityGate:
    __slots__ = ("spec",)

    def __init__(self, spec: QualityGateSpec) -> None:
        if not isinstance(spec, QualityGateSpec):
            raise TypeError("spec must be QualityGateSpec")
        self.spec = spec

    def evaluate(self, profile: QualityProfile) -> QualityGateReport:
        if not isinstance(profile, QualityProfile):
            raise TypeError("profile must be QualityProfile")
        checks = tuple(requirement.evaluate(profile) for requirement in self.spec.requirements)
        return QualityGateReport(self.spec.analysis_name, _status_from_checks(checks), checks, profile.total_transactions)


class QualityGateRegistry:
    __slots__ = ("_specs",)

    def __init__(self, specs: Iterable[QualityGateSpec]) -> None:
        by_name: dict[str, QualityGateSpec] = {}
        for spec in specs:
            if not isinstance(spec, QualityGateSpec):
                raise QualityGateConfigurationError("registry accepts QualityGateSpec objects only")
            key = spec.analysis_name.casefold()
            if key in by_name:
                raise QualityGateConfigurationError(f"duplicate analysis gate name: {spec.analysis_name}")
            by_name[key] = spec
        if not by_name:
            raise QualityGateConfigurationError("quality-gate registry must contain at least one gate")
        self._specs = MappingProxyType(by_name)

    @property
    def specs(self) -> tuple[QualityGateSpec, ...]:
        return tuple(self._specs.values())

    def get(self, analysis_name: str) -> QualityGateSpec:
        key = _clean_identifier(analysis_name, "analysis_name").casefold()
        try:
            return self._specs[key]
        except KeyError as exc:
            raise QualityGateConfigurationError(f"unknown analysis gate: {analysis_name!r}") from exc

    def evaluate(self, analysis_name: str, profile: QualityProfile) -> QualityGateReport:
        return QualityGate(self.get(analysis_name)).evaluate(profile)

    def evaluate_all(self, profile: QualityProfile) -> tuple[QualityGateReport, ...]:
        return tuple(QualityGate(spec).evaluate(profile) for spec in self.specs)


def evaluate_quality_gate(profile: QualityProfile, spec: QualityGateSpec) -> QualityGateReport:
    return QualityGate(spec).evaluate(profile)


def _status_from_checks(checks: Sequence[GateCheck]) -> GateStatus:
    if any(not check.passed and check.severity is GateSeverity.BLOCK for check in checks):
        return GateStatus.BLOCKED
    if any(not check.passed for check in checks):
        return GateStatus.DEGRADED
    return GateStatus.READY


def _breakdown(profile: QualityProfile, kind: BreakdownKind) -> tuple[NamedCount, ...]:
    if kind is BreakdownKind.VENDOR_IDENTITY: return profile.vendor_identity_modes
    if kind is BreakdownKind.CATEGORY_IDENTITY: return profile.category_identity_modes
    if kind is BreakdownKind.AGENCY_IDENTITY: return profile.agency_identity_modes
    if kind is BreakdownKind.COMPETITION_CORE: return profile.competition_core_modes
    if kind is BreakdownKind.OBLIGATION_SIGN: return profile.obligation_signs
    raise QualityGateEvaluationError(f"unsupported breakdown metric: {kind!r}")


def _cardinality_items(profile: QualityProfile, kind: CardinalityKind) -> tuple[NamedCount, ...]:
    if kind is CardinalityKind.SOURCE: return profile.source_names
    if kind is CardinalityKind.SOURCE_SCHEMA: return profile.source_schemas
    if kind is CardinalityKind.ACTION_FISCAL_YEAR: return profile.action_fiscal_years
    raise QualityGateEvaluationError(f"unsupported cardinality metric: {kind!r}")
