from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from hashlib import sha256
from io import StringIO
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pytest

import procurelens.pipeline.usaspending_live as live
from procurelens.detectors.calibration import ScoreCalibrationSpec
from procurelens.detectors.ensemble import EnsembleMethod
from procurelens.detectors.isolation_forest import IsolationForestConfig
from procurelens.export.writer import CsvSafetyMode, ExportFormat, ExportSerializationSpec
from procurelens.features.amount_reference import AmountBasis, AmountReferencePolicy
from procurelens.features.award_change_context import AwardChangeContextSupportSpec
from procurelens.features.award_change_reference import (
    AwardChangeReferencePolicy,
    federal_award_change_reference_plan,
)
from procurelens.features.catalog import feature_catalog
from procurelens.features.competition_context import CompetitionContextSupportSpec
from procurelens.features.competition_reference import (
    CompetitionReferencePolicy,
    federal_competition_context_plan,
)
from procurelens.features.peer_groups import federal_contract_amount_peer_plan
from procurelens.features.vendor_identity import VendorIdentityScope
from procurelens.features.vendor_market import VendorMarketPolicy, federal_vendor_market_plan
from procurelens.features.vendor_market_context import VendorMarketSupportSpec
from procurelens.model.feature_spec import make_feature_selection_spec, resolve_feature_selection
from procurelens.model.preprocessing_spec import (
    FeaturePreprocessingRule,
    MissingValueStrategy,
    NumericTransform,
    PreprocessingSpec,
    resolve_preprocessing_spec,
)
from procurelens.pipeline.config import (
    IsolationForestRunPlan,
    ModelReviewPlan,
    TwoDetectorEnsemblePlan,
)
from procurelens.pipeline.feature_config import FeatureBuildPlan
from procurelens.quality.gate import (
    GateSeverity,
    PopulationRequirement,
    QualityGateSpec,
)
from procurelens.review.explanation import ExplanationSpec
from procurelens.review.policy import ReviewPolicySpec, ReviewSelectionMethod
from procurelens.sources.usaspending.artifact import (
    ArchiveMember,
    ArtifactReceipt,
    USAspendingArtifactStore,
)
from procurelens.sources.usaspending.client import (
    DownloadCount,
    DownloadJob,
    DownloadStatus,
    RequestFingerprint,
    USAspendingClient,
)
from procurelens.sources.usaspending.loader import USAspendingDatasetLoader
from procurelens.statistics.robust import QuantileMethod


_HEADERS = (
    "contract_award_unique_key",
    "contract_transaction_unique_key",
    "award_id_piid",
    "modification_number",
    "recipient_name",
    "recipient_uei",
    "action_date",
    "federal_action_obligation",
    "total_dollars_obligated",
    "awarding_agency_code",
    "awarding_agency_name",
    "awarding_sub_agency_code",
    "awarding_sub_agency_name",
    "naics_code",
    "product_or_service_code",
    "extent_competed_code",
    "extent_competed",
    "number_of_offers_received",
    "solicitation_procedures_code",
    "solicitation_procedures",
    "other_than_full_and_open_competition_code",
    "other_than_full_and_open_competition",
)


class _FakeClient(USAspendingClient):
    def __init__(
        self,
        *,
        count: DownloadCount,
        job: DownloadJob,
        status: DownloadStatus,
    ) -> None:
        self.fake_count = count
        self.fake_job = job
        self.fake_status = status
        self.start_calls = 0
        self.wait_calls = 0
        self.count_filters = None
        self.start_filters = None

    def count_transactions(self, filters, *, spending_level="transactions"):
        self.count_filters = filters
        assert spending_level == "transactions"
        return self.fake_count

    def start_search_download(
        self,
        filters,
        *,
        columns=None,
        spending_levels=("transactions",),
        file_format="csv",
        limit=None,
    ):
        self.start_calls += 1
        self.start_filters = filters
        assert spending_levels == ("transactions",)
        assert limit is None
        return self.fake_job

    def wait_for_download(self, job, **kwargs):
        self.wait_calls += 1
        assert job is self.fake_job
        return self.fake_status


class _FakeArtifactStore(USAspendingArtifactStore):
    def __init__(self, receipt: ArtifactReceipt) -> None:
        self.receipt = receipt
        self.calls = 0

    def materialize_finished_job(self, job, status, directory, *, overwrite=False):
        self.calls += 1
        assert overwrite is False
        return self.receipt


def test_live_preparation_is_bounded_verified_and_frozen(tmp_path: Path) -> None:
    receipt = _artifact(tmp_path, _rows())
    plan = _live_plan(minimum_transactions=1)
    client, store = _source_doubles(receipt, transaction_count=4)

    prepared = live.prepare_live_usaspending_dataset(
        plan,
        tmp_path / "work",
        client=client,
        artifact_store=store,
        loader=USAspendingDatasetLoader(),
    )

    assert prepared.transaction_count == 4
    assert prepared.load_report.complete is True
    assert prepared.quality_gate.status.value == "ready"
    assert prepared.analysis_allowed is True
    assert prepared.artifact.sha256_hex == receipt.sha256_hex
    assert client.start_calls == 1
    assert store.calls == 1
    assert client.count_filters == client.start_filters == plan.filters_payload
    assert len(prepared.transaction_population_sha256) == 64
    assert len(prepared.evidence_sha256) == 64


def test_server_limit_stops_before_download_job(tmp_path: Path) -> None:
    receipt = _artifact(tmp_path, _rows())
    plan = _live_plan(minimum_transactions=1)
    client, store = _source_doubles(
        receipt,
        transaction_count=101,
        maximum_transaction_limit=100,
        over_limit=True,
    )

    with pytest.raises(live.USAspendingPopulationLimitError):
        live.prepare_live_usaspending_dataset(
            plan,
            tmp_path / "work-limit",
            client=client,
            artifact_store=store,
            loader=USAspendingDatasetLoader(),
        )

    assert client.start_calls == 0
    assert client.wait_calls == 0
    assert store.calls == 0


def test_failed_download_stops_before_artifact_materialization(tmp_path: Path) -> None:
    receipt = _artifact(tmp_path, _rows())
    plan = _live_plan(minimum_transactions=1)
    client, store = _source_doubles(receipt, transaction_count=4, failed=True)

    with pytest.raises(live.USAspendingDownloadFailed):
        live.prepare_live_usaspending_dataset(
            plan,
            tmp_path / "work-failed",
            client=client,
            artifact_store=store,
            loader=USAspendingDatasetLoader(),
        )

    assert client.start_calls == 1
    assert client.wait_calls == 1
    assert store.calls == 0


def test_blocked_and_degraded_quality_never_silently_run_analysis(tmp_path: Path) -> None:
    receipt = _artifact(tmp_path, _rows())

    blocked = _live_plan(minimum_transactions=100)
    client, store = _source_doubles(receipt, transaction_count=4)
    called = False

    def forbidden_runner(**kwargs):
        nonlocal called
        called = True
        raise AssertionError("analysis runner must not be called")

    with pytest.raises(live.USAspendingQualityGateBlocked) as captured:
        live.run_live_usaspending_analysis(
            blocked,
            tmp_path / "work-blocked",
            client=client,
            artifact_store=store,
            loader=USAspendingDatasetLoader(),
            analysis_runner=forbidden_runner,
        )
    assert captured.value.prepared.quality_gate.status.value == "blocked"
    assert called is False

    degraded = _live_plan(
        minimum_transactions=100,
        severity=GateSeverity.WARNING,
        allow_degraded=False,
    )
    client, store = _source_doubles(receipt, transaction_count=4)
    with pytest.raises(live.USAspendingQualityGateBlocked) as captured:
        live.run_live_usaspending_analysis(
            degraded,
            tmp_path / "work-degraded",
            client=client,
            artifact_store=store,
            loader=USAspendingDatasetLoader(),
            analysis_runner=forbidden_runner,
        )
    assert captured.value.prepared.quality_gate.status.value == "degraded"
    assert called is False


def test_explicit_degraded_opt_in_forwards_exact_source_revision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt = _artifact(tmp_path, _rows())
    plan = _live_plan(
        minimum_transactions=100,
        severity=GateSeverity.WARNING,
        allow_degraded=True,
    )
    client, store = _source_doubles(receipt, transaction_count=4)
    observed = {}

    @dataclass(frozen=True)
    class _Analysis:
        run_name: str
        feature_plan: FeatureBuildPlan
        model_plan: ModelReviewPlan
        evidence_sha256: str = "e" * 64

    monkeypatch.setattr(live, "ProcureLensAnalysisRun", _Analysis)

    def runner(**kwargs):
        observed.update(kwargs)
        return _Analysis(kwargs["run_name"], kwargs["feature_plan"], kwargs["model_plan"])

    result = live.run_live_usaspending_analysis(
        plan,
        tmp_path / "work-degraded-allowed",
        client=client,
        artifact_store=store,
        loader=USAspendingDatasetLoader(),
        software_components={"test": "live-orchestration"},
        analysis_runner=runner,
    )

    assert result.prepared.quality_gate.status.value == "degraded"
    assert result.prepared.analysis_allowed is True
    assert observed["reference_transactions"] == result.prepared.transactions
    assert observed["scoring_transactions"] == result.prepared.transactions
    assert observed["source_revision"] == receipt.sha256_hex
    assert observed["run_name"] == plan.name


def test_live_plan_freezes_filter_input_and_fingerprints_policy() -> None:
    filters = {"time_period": [{"start_date": "2026-01-01", "end_date": "2026-01-31"}]}
    plan = _live_plan(minimum_transactions=1, filters=filters)
    before = plan.sha256_hex
    filters["time_period"][0]["start_date"] = "1900-01-01"

    assert plan.filters_payload["time_period"][0]["start_date"] == "2026-01-01"
    assert plan.sha256_hex == before

    changed = _live_plan(
        minimum_transactions=2,
        filters={"time_period": [{"start_date": "2026-01-01", "end_date": "2026-01-31"}]},
    )
    assert changed.sha256_hex != before


def _live_plan(
    *,
    minimum_transactions: int,
    severity: GateSeverity = GateSeverity.BLOCK,
    allow_degraded: bool = False,
    filters=None,
) -> live.LiveUSAspendingPlan:
    catalog = feature_catalog()
    feature_plan = FeatureBuildPlan(
        name="live-source-feature-plan",
        description="Explicit policy used only to validate live source orchestration.",
        feature_catalog_sha256=catalog.sha256_hex,
        amount_peer_plan=federal_contract_amount_peer_plan(),
        amount_basis=AmountBasis.ACTION_OBLIGATION,
        amount_minimum_peer_count=1,
        amount_reference_policy=AmountReferencePolicy(),
        vendor_peer_plan=federal_vendor_market_plan(),
        vendor_scope=VendorIdentityScope.ENTITY,
        vendor_support=VendorMarketSupportSpec(
            minimum_observed_new_awards=1,
            minimum_identified_new_awards=1,
            minimum_observed_winning_vendors=1,
            minimum_vendor_identity_coverage=Decimal("0"),
        ),
        vendor_market_policy=VendorMarketPolicy(),
        competition_peer_plan=federal_competition_context_plan(),
        competition_support=CompetitionContextSupportSpec(
            minimum_base_awards=1,
            minimum_process_known=0,
            minimum_offers_known=0,
            minimum_procedure_known=0,
            minimum_process_coverage=Decimal("0"),
            minimum_offer_coverage=Decimal("0"),
            minimum_procedure_coverage=Decimal("0"),
        ),
        competition_reference_policy=CompetitionReferencePolicy(),
        award_change_peer_plan=federal_award_change_reference_plan(),
        award_change_support=AwardChangeContextSupportSpec(minimum_peer_awards=1),
        award_change_reference_policy=AwardChangeReferencePolicy(),
        quantile_method=QuantileMethod.LINEAR_TYPE7,
    )

    first_name = catalog.entries[0].name
    selection_spec = make_feature_selection_spec(
        name="live-source-selection",
        description="One explicit catalog feature; modeling is not executed in preparation tests.",
        feature_names=(first_name,),
        catalog=catalog,
    )
    selection = resolve_feature_selection(selection_spec, catalog=catalog)
    preprocessing_spec = PreprocessingSpec(
        name="live-source-preprocessing",
        description="Explicit preparation contract for orchestration tests.",
        selection_sha256=selection.sha256_hex,
        catalog_sha256=catalog.sha256_hex,
        rules=(
            FeaturePreprocessingRule(
                feature_name=first_name,
                missing_strategy=MissingValueStrategy.REQUIRE_PRESENT,
                add_missing_indicator=False,
                transform=NumericTransform.IDENTITY,
            ),
        ),
    )
    preprocessing = resolve_preprocessing_spec(selection, preprocessing_spec)
    model_plan = ModelReviewPlan(
        name="live-source-model-plan",
        description="Valid model contract; source preparation tests do not fit it.",
        feature_selection=selection,
        preprocessing=preprocessing,
        isolation_forest_runs=(
            IsolationForestRunPlan(
                run_name="if-seed-7",
                config=IsolationForestConfig(
                    n_estimators=8,
                    max_samples="auto",
                    max_features=Decimal("1"),
                    bootstrap=False,
                    random_state=7,
                    n_jobs=1,
                ),
                primary=True,
            ),
        ),
        calibration=ScoreCalibrationSpec(
            name="live-source-calibration",
            quantile_method=QuantileMethod.LINEAR_TYPE7,
        ),
        ensemble=TwoDetectorEnsemblePlan(
            method=EnsembleMethod.WEIGHTED_MEAN,
            isolation_forest_weight=Decimal("0.5"),
            empirical_tail_weight=Decimal("0.5"),
        ),
        build_stability_report=False,
        review_policy=ReviewPolicySpec(
            name="live-source-review",
            description="Explicit non-hidden review threshold for configuration validity.",
            method=ReviewSelectionMethod.MINIMUM_SCORE,
            minimum_score=Decimal("50"),
        ),
        explanation=ExplanationSpec(
            name="live-source-explanation",
            description="Factual selected evidence.",
            catalog_sha256=catalog.sha256_hex,
            feature_names=(first_name,),
            include_missing_features=True,
        ),
        exports=(
            ExportSerializationSpec(format=ExportFormat.JSON, json_pretty=False),
            ExportSerializationSpec(
                format=ExportFormat.CSV,
                csv_safety_mode=CsvSafetyMode.PROGRAMMATIC,
            ),
        ),
    )
    return live.LiveUSAspendingPlan(
        name="bounded-live-source-test",
        description="Bounded explicit live-source orchestration contract.",
        filters=(
            filters
            if filters is not None
            else {"time_period": [{"start_date": "2026-01-01", "end_date": "2026-01-31"}]}
        ),
        quality_gate=QualityGateSpec(
            analysis_name="live-source-quality",
            requirements=(
                PopulationRequirement(
                    requirement_id="population",
                    minimum_transactions=minimum_transactions,
                    severity=severity,
                ),
            ),
        ),
        feature_plan=feature_plan,
        model_plan=model_plan,
        allow_degraded_quality=allow_degraded,
    )


def _source_doubles(
    receipt: ArtifactReceipt,
    *,
    transaction_count: int,
    maximum_transaction_limit: int = 100,
    over_limit: bool = False,
    failed: bool = False,
) -> tuple[_FakeClient, _FakeArtifactStore]:
    count_fp = RequestFingerprint("a" * 64, "{}")
    job_fp = RequestFingerprint("b" * 64, "{}")
    count = DownloadCount(
        calculated_transaction_count=transaction_count,
        maximum_transaction_limit=maximum_transaction_limit,
        transaction_rows_gt_limit=over_limit,
        calculated_count=transaction_count,
        spending_level="transactions",
        maximum_limit=maximum_transaction_limit,
        rows_gt_limit=over_limit,
        messages=(),
        request_fingerprint=count_fp,
    )
    job = DownloadJob(
        status_url="https://api.usaspending.gov/api/v2/download/status/?file=x",
        file_name=receipt.file_name,
        file_url=receipt.source_url,
        download_request={"spending_level": ["transactions"]},
        request_fingerprint=job_fp,
    )
    status = DownloadStatus(
        status="failed" if failed else "finished",
        file_name=receipt.file_name,
        file_url=None if failed else receipt.source_url,
        message="synthetic failure" if failed else None,
        total_rows=None if failed else transaction_count,
        total_columns=None,
        total_size=None,
        seconds_elapsed=0.1,
        checked_at=datetime(2026, 9, 5, 0, 0, tzinfo=timezone.utc),
    )
    return _FakeClient(count=count, job=job, status=status), _FakeArtifactStore(receipt)


def _artifact(tmp_path: Path, rows: tuple[dict[str, str], ...]) -> ArtifactReceipt:
    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=_HEADERS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)

    path = tmp_path / "bounded-live-source.zip"
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("contracts.csv", buffer.getvalue().encode("utf-8"))
    payload = path.read_bytes()
    with ZipFile(path, "r") as archive:
        members = tuple(
            ArchiveMember(
                name=info.filename,
                compressed_bytes=info.compress_size,
                uncompressed_bytes=info.file_size,
                crc32=info.CRC,
                compression_method=info.compress_type,
                is_directory=info.is_dir(),
            )
            for info in archive.infolist()
        )
    return ArtifactReceipt(
        path=path,
        source_url="https://files.usaspending.gov/bounded-live-source.zip",
        final_url="https://files.usaspending.gov/bounded-live-source.zip",
        file_name=path.name,
        size_bytes=len(payload),
        sha256_hex=sha256(payload).hexdigest(),
        downloaded_at=datetime(2026, 9, 5, 0, 0, tzinfo=timezone.utc),
        resumed_from_bytes=0,
        etag='"bounded-live-source"',
        last_modified=None,
        content_type="application/zip",
        request_fingerprint_sha256="b" * 64,
        archive_members=members,
        total_uncompressed_bytes=sum(member.uncompressed_bytes for member in members),
    )


def _rows() -> tuple[dict[str, str], ...]:
    return tuple(
        {
            "contract_award_unique_key": f"LIVE-AWARD-{index}",
            "contract_transaction_unique_key": f"LIVE-TX-{index}",
            "award_id_piid": f"LIVE-PIID-{index}",
            "modification_number": "0",
            "recipient_name": f"Synthetic Vendor {index}",
            "recipient_uei": f"UEI-LIVE-{index}",
            "action_date": f"2026-01-{index + 1:02d}",
            "federal_action_obligation": str(1000 + index * 100),
            "total_dollars_obligated": str(1000 + index * 100),
            "awarding_agency_code": "1000",
            "awarding_agency_name": "Synthetic Agency",
            "awarding_sub_agency_code": "1001",
            "awarding_sub_agency_name": "Synthetic Subagency",
            "naics_code": "541511",
            "product_or_service_code": "D302",
            "extent_competed_code": "A",
            "extent_competed": "Full and Open Competition",
            "number_of_offers_received": str(index + 1),
            "solicitation_procedures_code": "NP",
            "solicitation_procedures": "Negotiated Proposal/Quote",
            "other_than_full_and_open_competition_code": "",
            "other_than_full_and_open_competition": "",
        }
        for index in range(1, 5)
    )
