"""Auditable live-USAspending acquisition-to-analysis orchestration for ProcureLens.

This module connects the already-separated USAspending control plane, verified
artifact transfer, explicit schema loader, quality gate, and top-level ProcureLens
analysis runner without weakening any of those boundaries.

The analytical population remains caller-owned. The runner never invents filters,
silently truncates an over-limit population, changes quality requirements, or
mutates feature/model policy. A quality-gate block stops analysis after preserving
all acquisition/loading/readiness evidence.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from hashlib import sha256
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any

from procurelens.domain.transaction import ProcurementTransaction
from procurelens.pipeline.config import ModelReviewPlan
from procurelens.pipeline.feature_config import FeatureBuildPlan
from procurelens.pipeline.run import ProcureLensAnalysisRun, run_procurelens_analysis
from procurelens.quality.gate import (
    GateStatus,
    QualityGateReport,
    QualityGateSpec,
    evaluate_quality_gate,
)
from procurelens.quality.profile import QualityProfile, profile_transactions
from procurelens.sources.usaspending.artifact import (
    ArtifactReceipt,
    USAspendingArtifactStore,
)
from procurelens.sources.usaspending.client import (
    DownloadCount,
    DownloadJob,
    DownloadStatus,
    USAspendingClient,
)
from procurelens.sources.usaspending.loader import (
    LoadPlan,
    LoadReport,
    USAspendingDatasetLoader,
)


class LiveUSAspendingError(RuntimeError):
    """Base error for the bounded live-USAspending orchestration boundary."""


class USAspendingPopulationLimitError(LiveUSAspendingError):
    """Raised before download creation when USAspending reports a limit breach."""

    def __init__(self, count: DownloadCount) -> None:
        super().__init__(
            "requested USAspending transaction population exceeds the current "
            "server download limit; narrow or deliberately partition the filters"
        )
        self.count = count


class USAspendingDownloadFailed(LiveUSAspendingError):
    """Raised when an asynchronous USAspending download job terminates as failed."""

    def __init__(self, status: DownloadStatus) -> None:
        message = status.message or "USAspending download job failed"
        super().__init__(message)
        self.status = status


class USAspendingPreparationError(LiveUSAspendingError):
    """Raised when a source artifact cannot become a complete canonical dataset."""


class USAspendingQualityGateBlocked(LiveUSAspendingError):
    """Fail-closed analysis stop retaining the complete prepared-dataset evidence."""

    def __init__(self, prepared: "PreparedUSAspendingDataset") -> None:
        reason = prepared.analysis_block_reason or "quality gate does not allow analysis"
        super().__init__(reason)
        self.prepared = prepared


class USAspendingDownloadFormat(str, Enum):
    CSV = "csv"
    TSV = "tsv"
    PSTXT = "pstxt"


@dataclass(frozen=True, slots=True)
class DownloadWaitPolicy:
    """Explicit operational polling policy; not an anomaly/model threshold."""

    timeout_seconds: float = 900.0
    initial_poll_seconds: float = 1.0
    max_poll_seconds: float = 15.0
    poll_multiplier: float = 1.5
    jitter_ratio: float = 0.1

    def __post_init__(self) -> None:
        for name in ("timeout_seconds", "initial_poll_seconds", "max_poll_seconds"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.max_poll_seconds < self.initial_poll_seconds:
            raise ValueError("max_poll_seconds must be >= initial_poll_seconds")
        if (
            isinstance(self.poll_multiplier, bool)
            or not isinstance(self.poll_multiplier, (int, float))
            or self.poll_multiplier < 1
        ):
            raise ValueError("poll_multiplier must be >= 1")
        if (
            isinstance(self.jitter_ratio, bool)
            or not isinstance(self.jitter_ratio, (int, float))
            or not 0 <= self.jitter_ratio <= 1
        ):
            raise ValueError("jitter_ratio must be between 0 and 1")

    @property
    def sha256_hex(self) -> str:
        return _digest(self.as_dict())

    def as_dict(self) -> dict[str, Any]:
        return {
            "timeout_seconds": self.timeout_seconds,
            "initial_poll_seconds": self.initial_poll_seconds,
            "max_poll_seconds": self.max_poll_seconds,
            "poll_multiplier": self.poll_multiplier,
            "jitter_ratio": self.jitter_ratio,
        }


@dataclass(frozen=True, slots=True)
class LiveUSAspendingPlan:
    """Complete explicit policy for one live USAspending-to-analysis run."""

    name: str
    description: str
    filters: Mapping[str, Any]
    quality_gate: QualityGateSpec
    feature_plan: FeatureBuildPlan
    model_plan: ModelReviewPlan

    columns: tuple[str, ...] | None = None
    file_format: USAspendingDownloadFormat = USAspendingDownloadFormat.CSV
    wait_policy: DownloadWaitPolicy = DownloadWaitPolicy()
    member_names: tuple[str, ...] | None = None
    profile_name: str | None = None
    allow_degraded_quality: bool = False

    def __post_init__(self) -> None:
        for field_name in ("name", "description"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must not be blank")
            object.__setattr__(self, field_name, value.strip())

        if not isinstance(self.filters, Mapping):
            raise TypeError("filters must be a mapping")
        frozen_filters = _freeze_json(self.filters, "filters")
        if not isinstance(frozen_filters, Mapping):
            raise AssertionError("frozen filters must remain a mapping")
        object.__setattr__(self, "filters", frozen_filters)

        if not isinstance(self.quality_gate, QualityGateSpec):
            raise TypeError("quality_gate must be QualityGateSpec")
        if not isinstance(self.feature_plan, FeatureBuildPlan):
            raise TypeError("feature_plan must be FeatureBuildPlan")
        if not isinstance(self.model_plan, ModelReviewPlan):
            raise TypeError("model_plan must be ModelReviewPlan")
        if (
            self.model_plan.feature_selection.spec.catalog_sha256
            != self.feature_plan.feature_catalog_sha256
        ):
            raise ValueError(
                "model feature-selection catalog differs from feature-build catalog"
            )

        object.__setattr__(self, "file_format", USAspendingDownloadFormat(self.file_format))
        if not isinstance(self.wait_policy, DownloadWaitPolicy):
            raise TypeError("wait_policy must be DownloadWaitPolicy")
        object.__setattr__(self, "columns", _optional_names(self.columns, "columns"))
        object.__setattr__(
            self,
            "member_names",
            _optional_names(self.member_names, "member_names"),
        )
        profile = self.profile_name
        if profile is not None:
            if not isinstance(profile, str):
                raise TypeError("profile_name must be text or None")
            profile = profile.strip() or None
        object.__setattr__(self, "profile_name", profile)
        if not isinstance(self.allow_degraded_quality, bool):
            raise TypeError("allow_degraded_quality must be bool")

    @property
    def filters_payload(self) -> dict[str, Any]:
        payload = _jsonable(self.filters)
        assert isinstance(payload, dict)
        return payload

    @property
    def sha256_hex(self) -> str:
        return _digest(self.as_dict(include_sha=False))

    def as_dict(self, *, include_sha: bool = True) -> dict[str, Any]:
        result = {
            "name": self.name,
            "description": self.description,
            "filters": self.filters_payload,
            "columns": None if self.columns is None else list(self.columns),
            "file_format": self.file_format.value,
            "wait_policy": self.wait_policy.as_dict(),
            "member_names": (
                None if self.member_names is None else list(self.member_names)
            ),
            "profile_name": self.profile_name,
            "allow_degraded_quality": self.allow_degraded_quality,
            "quality_gate": _canonical(self.quality_gate),
            "feature_plan_sha256": self.feature_plan.sha256_hex,
            "model_plan_sha256": self.model_plan.sha256_hex,
        }
        if include_sha:
            result["sha256"] = self.sha256_hex
        return result


@dataclass(frozen=True, slots=True)
class PreparedUSAspendingDataset:
    """Verified canonical population plus acquisition/loading/readiness evidence."""

    plan: LiveUSAspendingPlan
    count: DownloadCount
    job: DownloadJob
    status: DownloadStatus
    artifact: ArtifactReceipt
    load_plan: LoadPlan
    load_report: LoadReport
    quality_profile: QualityProfile
    quality_gate: QualityGateReport
    transactions: tuple[ProcurementTransaction, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.plan, LiveUSAspendingPlan):
            raise TypeError("plan must be LiveUSAspendingPlan")
        if not isinstance(self.count, DownloadCount):
            raise TypeError("count must be DownloadCount")
        if not isinstance(self.job, DownloadJob):
            raise TypeError("job must be DownloadJob")
        if not isinstance(self.status, DownloadStatus):
            raise TypeError("status must be DownloadStatus")
        if self.status.status != "finished":
            raise USAspendingPreparationError("prepared dataset requires finished download")
        if self.status.file_name != self.job.file_name:
            raise USAspendingPreparationError("download job/status file names differ")
        if not isinstance(self.artifact, ArtifactReceipt):
            raise TypeError("artifact must be ArtifactReceipt")
        if self.artifact.file_name != self.job.file_name:
            raise USAspendingPreparationError("artifact file name differs from download job")
        if (
            self.artifact.request_fingerprint_sha256
            != self.job.request_fingerprint.sha256_hex
        ):
            raise USAspendingPreparationError(
                "artifact request fingerprint differs from download job"
            )
        if not isinstance(self.load_plan, LoadPlan):
            raise TypeError("load_plan must be LoadPlan")
        if self.load_plan.artifact_sha256 != self.artifact.sha256_hex:
            raise USAspendingPreparationError("load plan differs from verified artifact")
        if not isinstance(self.load_report, LoadReport):
            raise TypeError("load_report must be LoadReport")
        if not self.load_report.complete:
            raise USAspendingPreparationError("loader session did not fully exhaust")
        if self.load_report.plan_sha256 != self.load_plan.plan_sha256:
            raise USAspendingPreparationError("load report differs from load plan")
        if not isinstance(self.quality_profile, QualityProfile):
            raise TypeError("quality_profile must be QualityProfile")
        if not isinstance(self.quality_gate, QualityGateReport):
            raise TypeError("quality_gate must be QualityGateReport")
        if self.quality_gate.analysis_name != self.plan.quality_gate.analysis_name:
            raise USAspendingPreparationError("quality report differs from configured gate")

        transactions = tuple(self.transactions)
        if any(not isinstance(item, ProcurementTransaction) for item in transactions):
            raise TypeError("transactions must contain ProcurementTransaction values")
        identities = tuple((item.transaction_id, item.award_id) for item in transactions)
        if len(identities) != len(set(identities)):
            raise USAspendingPreparationError(
                "prepared canonical population contains duplicate identities"
            )
        if self.load_report.transactions_emitted != len(transactions):
            raise USAspendingPreparationError(
                "loader emitted count differs from retained canonical population"
            )
        if self.quality_profile.total_transactions != len(transactions):
            raise USAspendingPreparationError(
                "quality population differs from canonical population"
            )
        if self.quality_gate.total_transactions != len(transactions):
            raise USAspendingPreparationError(
                "quality gate population differs from canonical population"
            )
        object.__setattr__(self, "transactions", transactions)

    @property
    def transaction_count(self) -> int:
        return len(self.transactions)

    @property
    def analysis_allowed(self) -> bool:
        if not self.transactions:
            return False
        if self.quality_gate.status is GateStatus.BLOCKED:
            return False
        if self.quality_gate.status is GateStatus.DEGRADED:
            return self.plan.allow_degraded_quality
        return True

    @property
    def analysis_block_reason(self) -> str | None:
        if not self.transactions:
            return "canonical population is empty after complete loading"
        if self.quality_gate.status is GateStatus.BLOCKED:
            return "configured data-quality gate is BLOCKED"
        if (
            self.quality_gate.status is GateStatus.DEGRADED
            and not self.plan.allow_degraded_quality
        ):
            return "configured data-quality gate is DEGRADED and plan disallows degraded analysis"
        return None

    @property
    def transaction_population_sha256(self) -> str:
        return _digest([_canonical(item) for item in self.transactions])

    @property
    def evidence_sha256(self) -> str:
        return _digest(self.as_dict(include_sha=False))

    def as_dict(self, *, include_sha: bool = True) -> dict[str, Any]:
        result = {
            "plan_sha256": self.plan.sha256_hex,
            "count": _download_count_dict(self.count),
            "job": _download_job_dict(self.job),
            "status": _download_status_dict(self.status),
            "artifact": _artifact_dict(self.artifact),
            "load_plan": _load_plan_dict(self.load_plan),
            "load_report": _canonical(self.load_report),
            "quality_profile": self.quality_profile.as_dict(),
            "quality_gate": self.quality_gate.as_dict(),
            "transaction_count": self.transaction_count,
            "transaction_population_sha256": self.transaction_population_sha256,
            "analysis_allowed": self.analysis_allowed,
            "analysis_block_reason": self.analysis_block_reason,
        }
        if include_sha:
            result["evidence_sha256"] = self.evidence_sha256
        return result


@dataclass(frozen=True, slots=True)
class LiveUSAspendingAnalysisRun:
    """One completed live-source preparation linked to one ProcureLens analysis."""

    prepared: PreparedUSAspendingDataset
    analysis: ProcureLensAnalysisRun

    def __post_init__(self) -> None:
        if not isinstance(self.prepared, PreparedUSAspendingDataset):
            raise TypeError("prepared must be PreparedUSAspendingDataset")
        if not self.prepared.analysis_allowed:
            raise USAspendingPreparationError(
                "analysis artifact cannot be attached to a blocked prepared dataset"
            )
        if not isinstance(self.analysis, ProcureLensAnalysisRun):
            raise TypeError("analysis must be ProcureLensAnalysisRun")
        if self.analysis.run_name != self.prepared.plan.name:
            raise USAspendingPreparationError(
                "analysis run name differs from live-USAspending plan"
            )
        if self.analysis.feature_plan.sha256_hex != self.prepared.plan.feature_plan.sha256_hex:
            raise USAspendingPreparationError("analysis feature plan differs from live plan")
        if self.analysis.model_plan.sha256_hex != self.prepared.plan.model_plan.sha256_hex:
            raise USAspendingPreparationError("analysis model plan differs from live plan")

    @property
    def evidence_sha256(self) -> str:
        return _digest(
            {
                "prepared_dataset_sha256": self.prepared.evidence_sha256,
                "analysis_sha256": self.analysis.evidence_sha256,
            }
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "prepared_dataset_sha256": self.prepared.evidence_sha256,
            "analysis_sha256": self.analysis.evidence_sha256,
            "evidence_sha256": self.evidence_sha256,
        }


def prepare_live_usaspending_dataset(
    plan: LiveUSAspendingPlan,
    working_directory: str | Path,
    *,
    client: USAspendingClient | None = None,
    artifact_store: USAspendingArtifactStore | None = None,
    loader: USAspendingDatasetLoader | None = None,
) -> PreparedUSAspendingDataset:
    """Acquire, verify, canonically load, and assess one explicit live population."""

    if not isinstance(plan, LiveUSAspendingPlan):
        raise TypeError("plan must be LiveUSAspendingPlan")
    root = Path(working_directory)
    if root.exists() and not root.is_dir():
        raise USAspendingPreparationError("working_directory exists but is not a directory")
    root.mkdir(parents=True, exist_ok=True)

    client = USAspendingClient() if client is None else client
    artifact_store = USAspendingArtifactStore() if artifact_store is None else artifact_store
    loader = USAspendingDatasetLoader() if loader is None else loader
    if not isinstance(client, USAspendingClient):
        raise TypeError("client must be USAspendingClient")
    if not isinstance(artifact_store, USAspendingArtifactStore):
        raise TypeError("artifact_store must be USAspendingArtifactStore")
    if not isinstance(loader, USAspendingDatasetLoader):
        raise TypeError("loader must be USAspendingDatasetLoader")

    count = client.count_transactions(plan.filters_payload, spending_level="transactions")
    if count.spending_level != "transactions":
        raise USAspendingPreparationError(
            "USAspending count response unexpectedly changed spending_level"
        )
    if count.transaction_rows_gt_limit or count.rows_gt_limit:
        raise USAspendingPopulationLimitError(count)
    if count.calculated_transaction_count < 1:
        raise USAspendingPreparationError("USAspending count preflight returned no transactions")

    job = client.start_search_download(
        plan.filters_payload,
        columns=plan.columns,
        spending_levels=("transactions",),
        file_format=plan.file_format.value,
        limit=None,
    )
    wait = plan.wait_policy
    status = client.wait_for_download(
        job,
        timeout_seconds=wait.timeout_seconds,
        initial_poll_seconds=wait.initial_poll_seconds,
        max_poll_seconds=wait.max_poll_seconds,
        poll_multiplier=wait.poll_multiplier,
        jitter_ratio=wait.jitter_ratio,
    )
    if status.status == "failed":
        raise USAspendingDownloadFailed(status)
    if status.status != "finished":
        raise USAspendingPreparationError(
            f"wait_for_download returned non-terminal status {status.status!r}"
        )

    artifact = artifact_store.materialize_finished_job(
        job,
        status,
        root / "source",
        overwrite=False,
    )
    load_plan = loader.plan(
        artifact,
        member_names=plan.member_names,
        profile_name=plan.profile_name,
    )
    session = loader.open_session(artifact, plan=load_plan)
    transactions = tuple(item.transaction for item in session.iter_transactions())
    load_report = session.report
    if not load_report.complete:
        raise USAspendingPreparationError("loader session was not fully exhausted")

    profile = profile_transactions(transactions)
    gate = evaluate_quality_gate(profile, plan.quality_gate)
    return PreparedUSAspendingDataset(
        plan=plan,
        count=count,
        job=job,
        status=status,
        artifact=artifact,
        load_plan=load_plan,
        load_report=load_report,
        quality_profile=profile,
        quality_gate=gate,
        transactions=transactions,
    )


def run_live_usaspending_analysis(
    plan: LiveUSAspendingPlan,
    working_directory: str | Path,
    *,
    client: USAspendingClient | None = None,
    artifact_store: USAspendingArtifactStore | None = None,
    loader: USAspendingDatasetLoader | None = None,
    software_components: Mapping[str, str] | Iterable[Any] = (),
    analysis_runner: Callable[..., ProcureLensAnalysisRun] = run_procurelens_analysis,
) -> LiveUSAspendingAnalysisRun:
    """Run a prepared live USAspending population only when its explicit gate allows it."""

    if not callable(analysis_runner):
        raise TypeError("analysis_runner must be callable")
    prepared = prepare_live_usaspending_dataset(
        plan,
        working_directory,
        client=client,
        artifact_store=artifact_store,
        loader=loader,
    )
    if not prepared.analysis_allowed:
        raise USAspendingQualityGateBlocked(prepared)

    analysis = analysis_runner(
        reference_transactions=prepared.transactions,
        scoring_transactions=prepared.transactions,
        feature_plan=plan.feature_plan,
        model_plan=plan.model_plan,
        run_name=plan.name,
        source_revision=prepared.artifact.sha256_hex,
        software_components=software_components,
    )
    if not isinstance(analysis, ProcureLensAnalysisRun):
        raise USAspendingPreparationError(
            "analysis_runner returned an unsupported analysis artifact"
        )
    return LiveUSAspendingAnalysisRun(prepared=prepared, analysis=analysis)


def _optional_names(
    values: Sequence[str] | None,
    name: str,
) -> tuple[str, ...] | None:
    if values is None:
        return None
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of strings or None")
    result: list[str] = []
    seen: set[str] = set()
    for raw in values:
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError(f"{name} must contain non-blank strings")
        value = raw.strip()
        if value in seen:
            raise ValueError(f"{name} must not contain duplicates")
        seen.add(value)
        result.append(value)
    if not result:
        raise ValueError(f"{name} must be None or contain at least one name")
    return tuple(result)


def _freeze_json(value: Any, name: str) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError(f"{name} must contain only finite JSON numbers")
        return value
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise ValueError(f"{name} must contain only finite numbers")
        # USAspending filter payloads are JSON. Preserve exact integral Decimals as
        # ints; reject non-integral Decimal values instead of silently rounding.
        integral = value.to_integral_value()
        if value != integral:
            raise TypeError(f"{name} contains Decimal not directly representable as JSON number")
        return int(integral)
    if isinstance(value, Mapping):
        frozen: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise TypeError(f"{name} mapping keys must be non-empty strings")
            frozen[key] = _freeze_json(item, name)
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item, name) for item in value)
    raise TypeError(f"{name} contains unsupported JSON value: {type(value).__name__}")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    return value


def _download_count_dict(value: DownloadCount) -> dict[str, Any]:
    return {
        "calculated_transaction_count": value.calculated_transaction_count,
        "maximum_transaction_limit": value.maximum_transaction_limit,
        "transaction_rows_gt_limit": value.transaction_rows_gt_limit,
        "calculated_count": value.calculated_count,
        "spending_level": value.spending_level,
        "maximum_limit": value.maximum_limit,
        "rows_gt_limit": value.rows_gt_limit,
        "messages": list(value.messages),
        "request_fingerprint_sha256": value.request_fingerprint.sha256_hex,
        "request_canonical_json": value.request_fingerprint.canonical_json,
    }


def _download_job_dict(value: DownloadJob) -> dict[str, Any]:
    return {
        "status_url": value.status_url,
        "file_name": value.file_name,
        "file_url": value.file_url,
        "download_request": _canonical(value.download_request),
        "request_fingerprint_sha256": value.request_fingerprint.sha256_hex,
        "request_canonical_json": value.request_fingerprint.canonical_json,
    }


def _download_status_dict(value: DownloadStatus) -> dict[str, Any]:
    return {
        "status": value.status,
        "file_name": value.file_name,
        "file_url": value.file_url,
        "message": value.message,
        "total_rows": value.total_rows,
        "total_columns": value.total_columns,
        "total_size": value.total_size,
        "seconds_elapsed": value.seconds_elapsed,
        "checked_at": value.checked_at.isoformat(),
    }


def _artifact_dict(value: ArtifactReceipt) -> dict[str, Any]:
    return {
        "source_url": value.source_url,
        "final_url": value.final_url,
        "file_name": value.file_name,
        "size_bytes": value.size_bytes,
        "sha256": value.sha256_hex,
        "downloaded_at": value.downloaded_at.isoformat(),
        "resumed_from_bytes": value.resumed_from_bytes,
        "etag": value.etag,
        "last_modified": value.last_modified,
        "content_type": value.content_type,
        "request_fingerprint_sha256": value.request_fingerprint_sha256,
        "archive_members": [_canonical(item) for item in value.archive_members],
        "total_uncompressed_bytes": value.total_uncompressed_bytes,
    }


def _load_plan_dict(value: LoadPlan) -> dict[str, Any]:
    return {
        "artifact_sha256": value.artifact_sha256,
        "artifact_size_bytes": value.artifact_size_bytes,
        "request_fingerprint_sha256": value.request_fingerprint_sha256,
        "member_names": list(value.member_names),
        "ignored_members": list(value.ignored_members),
        "members": [
            {
                "member_name": item.member.member_name,
                "observed_schema_sha256": item.member.schema_sha256,
                "profile_name": item.profile.name,
                "profile_sha256": item.profile.sha256_hex,
                "compatibility": _canonical(item.compatibility),
            }
            for item in value.members
        ],
        "plan_sha256": value.plan_sha256,
    }


def _canonical(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError("cannot fingerprint non-finite float")
        return value
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise ValueError("cannot fingerprint non-finite Decimal")
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {
            str(key): _canonical(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list, frozenset, set)):
        items = [_canonical(item) for item in value]
        if isinstance(value, (set, frozenset)):
            items.sort(key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":")))
        return items
    if is_dataclass(value):
        return {item.name: _canonical(getattr(value, item.name)) for item in fields(value)}
    raise TypeError(f"unsupported fingerprint value: {type(value).__name__}")


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
