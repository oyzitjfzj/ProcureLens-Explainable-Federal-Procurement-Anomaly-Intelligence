from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from decimal import Decimal
from hashlib import sha256
from io import StringIO
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from procurelens.detectors.calibration import ScoreCalibrationSpec
from procurelens.detectors.ensemble import EnsembleMethod
from procurelens.detectors.isolation_forest import IsolationForestConfig
from procurelens.export.writer import CsvSafetyMode, ExportFormat, ExportSerializationSpec
from procurelens.features.amount_reference import AmountBasis, AmountReferencePolicy
from procurelens.features.award_change_context import AwardChangeContextSupportSpec
from procurelens.features.award_change_reference import AwardChangeReferencePolicy, federal_award_change_reference_plan
from procurelens.features.catalog import feature_catalog
from procurelens.features.competition_context import CompetitionContextSupportSpec
from procurelens.features.competition_reference import CompetitionReferencePolicy, federal_competition_context_plan
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
from procurelens.pipeline.config import IsolationForestRunPlan, ModelReviewPlan, TwoDetectorEnsemblePlan
from procurelens.pipeline.feature_config import FeatureBuildPlan
from procurelens.pipeline.features import build_candidate_features, build_feature_references
from procurelens.pipeline.run import run_procurelens_analysis
from procurelens.quality.gate import (
    CoverageKind,
    CoverageMetric,
    CoverageRequirement,
    GateStatus,
    PopulationRequirement,
    QualityGateSpec,
    evaluate_quality_gate,
)
from procurelens.quality.profile import profile_transactions
from procurelens.review.explanation import ExplanationSpec
from procurelens.review.policy import ReviewPolicySpec, ReviewSelectionMethod, ReviewTiePolicy
from procurelens.sources.usaspending.artifact import ArchiveMember, ArtifactReceipt
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


def test_verified_usaspending_artifact_reaches_full_analysis_without_bypassing_quality(
    tmp_path: Path,
) -> None:
    receipt = _artifact(tmp_path, _rows())

    loader = USAspendingDatasetLoader()
    session = loader.open_session(receipt)
    transactions = tuple(item.transaction for item in session.iter_transactions())

    assert session.report.complete is True
    assert session.report.transactions_emitted == len(transactions) == 24
    assert all(tx.lineage.source_name for tx in transactions)

    quality = profile_transactions(transactions)
    gate = evaluate_quality_gate(
        quality,
        QualityGateSpec(
            analysis_name="bounded-usaspending-integration",
            description="Explicit fixture readiness checks before anomaly analysis.",
            requirements=(
                PopulationRequirement(
                    requirement_id="population",
                    minimum_transactions=20,
                ),
                CoverageRequirement(
                    requirement_id="vendor-identity",
                    metric=CoverageMetric(CoverageKind.CONTEXT, "vendor_identity"),
                    minimum_rate=Decimal("1"),
                ),
                CoverageRequirement(
                    requirement_id="competition-core",
                    metric=CoverageMetric(CoverageKind.CONTEXT, "competition_core_pair"),
                    minimum_rate=Decimal("1"),
                ),
                CoverageRequirement(
                    requirement_id="category",
                    metric=CoverageMetric(CoverageKind.CONTEXT, "procurement_category"),
                    minimum_rate=Decimal("1"),
                ),
            ),
        ),
    )
    assert gate.status is GateStatus.READY
    assert gate.allowed is True

    catalog = feature_catalog()
    feature_plan = _feature_plan(catalog.sha256_hex)
    references = build_feature_references(transactions, feature_plan, catalog=catalog)
    candidates = build_candidate_features(transactions, references, catalog=catalog)
    selected_names = _complete_varying_features(candidates.rows, limit=8)
    assert len(selected_names) >= 4

    model_plan = _model_plan(catalog.sha256_hex, selected_names)
    result = run_procurelens_analysis(
        reference_transactions=transactions,
        scoring_transactions=transactions,
        feature_plan=feature_plan,
        model_plan=model_plan,
        run_name="bounded-usaspending-source-to-analysis",
        source_revision=receipt.sha256_hex,
        software_components={"fixture": "usaspending-zip-v1"},
    )

    assert result.training_features.row_count == len(transactions)
    assert result.scoring_features is result.training_features
    assert result.model_review.review_scores.row_count == len(transactions)
    assert result.model_review.review_selection.selected_count == 4
    assert result.model_review.stability is not None

    exports = result.model_review.serialized_exports
    assert tuple(item.format for item in exports) == (ExportFormat.JSON, ExportFormat.CSV)

    payload = json.loads(exports[0].payload_text)
    assert payload["row_count"] == len(transactions)
    assert len(payload["records"]) == len(transactions)

    manifest = result.manifest
    assert manifest.artifact_index["model_review_run"].sha256_hex == result.model_review.evidence_sha256


def _feature_plan(catalog_sha256: str) -> FeatureBuildPlan:
    return FeatureBuildPlan(
        name="bounded-usaspending-feature-plan",
        description="Explicit source-to-analysis integration fixture policy.",
        feature_catalog_sha256=catalog_sha256,
        amount_peer_plan=federal_contract_amount_peer_plan(),
        amount_basis=AmountBasis.ACTION_OBLIGATION,
        amount_minimum_peer_count=4,
        amount_reference_policy=AmountReferencePolicy(),
        vendor_peer_plan=federal_vendor_market_plan(),
        vendor_scope=VendorIdentityScope.ENTITY,
        vendor_support=VendorMarketSupportSpec(
            minimum_observed_new_awards=4,
            minimum_identified_new_awards=4,
            minimum_observed_winning_vendors=2,
            minimum_vendor_identity_coverage=Decimal("0.75"),
        ),
        vendor_market_policy=VendorMarketPolicy(),
        competition_peer_plan=federal_competition_context_plan(),
        competition_support=CompetitionContextSupportSpec(
            minimum_base_awards=4,
            minimum_process_known=3,
            minimum_offers_known=3,
            minimum_procedure_known=3,
            minimum_process_coverage=Decimal("0.5"),
            minimum_offer_coverage=Decimal("0.5"),
            minimum_procedure_coverage=Decimal("0.5"),
        ),
        competition_reference_policy=CompetitionReferencePolicy(),
        award_change_peer_plan=federal_award_change_reference_plan(),
        award_change_support=AwardChangeContextSupportSpec(minimum_peer_awards=4),
        award_change_reference_policy=AwardChangeReferencePolicy(),
        quantile_method=QuantileMethod.LINEAR_TYPE7,
    )


def _model_plan(catalog_sha256: str, feature_names: tuple[str, ...]) -> ModelReviewPlan:
    catalog = feature_catalog()
    assert catalog.sha256_hex == catalog_sha256
    selection_spec = make_feature_selection_spec(
        name="bounded-usaspending-selection",
        description="Complete varying source-loaded features for integration.",
        feature_names=feature_names,
        catalog=catalog,
    )
    selection = resolve_feature_selection(selection_spec, catalog=catalog)
    preprocessing_spec = PreprocessingSpec(
        name="bounded-usaspending-preprocessing",
        description="Require selected fixture evidence without hidden imputation.",
        selection_sha256=selection.sha256_hex,
        catalog_sha256=catalog.sha256_hex,
        rules=tuple(
            FeaturePreprocessingRule(
                feature_name=name,
                missing_strategy=MissingValueStrategy.REQUIRE_PRESENT,
                add_missing_indicator=False,
                transform=NumericTransform.IDENTITY,
            )
            for name in feature_names
        ),
    )
    preprocessing = resolve_preprocessing_spec(selection, preprocessing_spec)
    return ModelReviewPlan(
        name="bounded-usaspending-model-plan",
        description="Real detector/review path over source-loaded canonical data.",
        feature_selection=selection,
        preprocessing=preprocessing,
        isolation_forest_runs=(
            IsolationForestRunPlan(
                run_name="if-seed-7",
                config=IsolationForestConfig(
                    n_estimators=32,
                    max_samples="auto",
                    max_features=Decimal("1"),
                    bootstrap=False,
                    random_state=7,
                    n_jobs=1,
                ),
                primary=True,
            ),
            IsolationForestRunPlan(
                run_name="if-seed-19",
                config=IsolationForestConfig(
                    n_estimators=32,
                    max_samples="auto",
                    max_features=Decimal("1"),
                    bootstrap=False,
                    random_state=19,
                    n_jobs=1,
                ),
                primary=False,
            ),
        ),
        calibration=ScoreCalibrationSpec(
            name="bounded-usaspending-calibration",
            quantile_method=QuantileMethod.LINEAR_TYPE7,
        ),
        ensemble=TwoDetectorEnsemblePlan(
            method=EnsembleMethod.WEIGHTED_MEAN,
            isolation_forest_weight=Decimal("0.5"),
            empirical_tail_weight=Decimal("0.5"),
        ),
        build_stability_report=True,
        review_policy=ReviewPolicySpec(
            name="bounded-usaspending-top-four",
            description="Fixture-only explicit review budget.",
            method=ReviewSelectionMethod.TOP_N,
            top_n=4,
            tie_policy=ReviewTiePolicy.DETERMINISTIC_IDENTITY,
        ),
        explanation=ExplanationSpec(
            name="bounded-usaspending-explanation",
            description="Factual fixture evidence without causal attribution.",
            catalog_sha256=catalog.sha256_hex,
            feature_names=feature_names[: min(5, len(feature_names))],
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


def _complete_varying_features(rows: tuple[object, ...], *, limit: int) -> tuple[str, ...]:
    names: list[str] = []
    first = rows[0]
    for candidate in first.values:
        values = tuple(row.get(candidate.name).value for row in rows)
        if all(value is not None for value in values) and len(set(values)) > 1:
            names.append(candidate.name)
            if len(names) == limit:
                break
    return tuple(names)


def _artifact(tmp_path: Path, rows: tuple[dict[str, str], ...]) -> ArtifactReceipt:
    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=_HEADERS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)

    path = tmp_path / "bounded-usaspending-contracts.zip"
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
        source_url="https://files.usaspending.gov/bounded-fixture.zip",
        final_url="https://files.usaspending.gov/bounded-fixture.zip",
        file_name=path.name,
        size_bytes=len(payload),
        sha256_hex=sha256(payload).hexdigest(),
        downloaded_at=datetime(2026, 9, 5, 0, 0, tzinfo=timezone.utc),
        resumed_from_bytes=0,
        etag='"bounded-fixture"',
        last_modified=None,
        content_type="application/zip",
        request_fingerprint_sha256=sha256(b"bounded-usaspending-request").hexdigest(),
        archive_members=members,
        total_uncompressed_bytes=sum(member.uncompressed_bytes for member in members),
    )


def _rows() -> tuple[dict[str, str], ...]:
    rows: list[dict[str, str]] = []
    for index in range(1, 13):
        base = Decimal(100 + index * index * 17)
        change = Decimal((index * 11) % 67 - 25)
        vendor = (
            "UEI-VENDOR-A"
            if index <= 5
            else "UEI-VENDOR-B"
            if index <= 8
            else f"UEI-VENDOR-{index}"
        )
        offers = (index % 5) + 1
        rows.append(
            _row(
                award_index=index,
                transaction_suffix="base",
                modification="0",
                vendor_uei=vendor,
                action_date=f"2026-01-{index + 2:02d}",
                action_obligation=str(base),
                award_total=str(base),
                offers=str(offers),
            )
        )
        rows.append(
            _row(
                award_index=index,
                transaction_suffix="mod1",
                modification="P00001",
                vendor_uei=vendor,
                action_date=f"2026-03-{index + 2:02d}",
                action_obligation=str(change),
                award_total=str(base + change),
                offers=str(offers),
            )
        )
    return tuple(rows)


def _row(
    *,
    award_index: int,
    transaction_suffix: str,
    modification: str,
    vendor_uei: str,
    action_date: str,
    action_obligation: str,
    award_total: str,
    offers: str,
) -> dict[str, str]:
    return {
        "contract_award_unique_key": f"USA-AWARD-{award_index}",
        "contract_transaction_unique_key": f"USA-TX-{award_index}-{transaction_suffix}",
        "award_id_piid": f"USA-PIID-{award_index}",
        "modification_number": modification,
        "recipient_name": f"Synthetic Vendor {vendor_uei}",
        "recipient_uei": vendor_uei,
        "action_date": action_date,
        "federal_action_obligation": action_obligation,
        "total_dollars_obligated": award_total,
        "awarding_agency_code": "1000",
        "awarding_agency_name": "Synthetic Agency",
        "awarding_sub_agency_code": "1001",
        "awarding_sub_agency_name": "Synthetic Subagency",
        "naics_code": "541511",
        "product_or_service_code": "D302",
        "extent_competed_code": "A",
        "extent_competed": "Full and Open Competition",
        "number_of_offers_received": offers,
        "solicitation_procedures_code": "NP",
        "solicitation_procedures": "Negotiated Proposal/Quote",
        "other_than_full_and_open_competition_code": "",
        "other_than_full_and_open_competition": "",
    }
