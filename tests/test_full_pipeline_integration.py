from __future__ import annotations

import csv
import json
from datetime import date, datetime, timezone
from decimal import Decimal
from io import StringIO

from procurelens.detectors.calibration import ScoreCalibrationSpec
from procurelens.detectors.ensemble import EnsembleMethod
from procurelens.detectors.isolation_forest import IsolationForestConfig
from procurelens.domain.transaction import ProcurementTransaction, SourceRecordRef
from procurelens.export.writer import (
    CsvSafetyMode,
    ExportFormat,
    ExportSerializationSpec,
)
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
from procurelens.pipeline.features import build_candidate_features, build_feature_references
from procurelens.pipeline.run import run_procurelens_analysis
from procurelens.review.explanation import ExplanationSpec
from procurelens.review.policy import (
    ReviewPolicySpec,
    ReviewSelectionMethod,
    ReviewTiePolicy,
)
from procurelens.statistics.robust import QuantileMethod


def test_real_full_pipeline_reaches_review_exports_and_manifest() -> None:
    catalog = feature_catalog()
    feature_plan = _feature_plan(catalog.sha256_hex)
    population = _population()

    # Resolve the model contract from real candidate rows. This dynamic selection
    # is test-only: production plans remain caller-owned and explicitly pinned.
    references = build_feature_references(population, feature_plan, catalog=catalog)
    candidate_batch = build_candidate_features(population, references, catalog=catalog)
    selected_names = _complete_varying_features(candidate_batch.rows, limit=8)
    assert len(selected_names) >= 4

    selection_spec = make_feature_selection_spec(
        name="synthetic-integration-selection",
        description="Complete varying feature columns for the full integration test.",
        feature_names=selected_names,
        catalog=catalog,
    )
    selection = resolve_feature_selection(selection_spec, catalog=catalog)
    preprocessing_spec = PreprocessingSpec(
        name="synthetic-identity-preprocessing",
        description="Require selected evidence and preserve its numeric scale.",
        selection_sha256=selection.sha256_hex,
        catalog_sha256=catalog.sha256_hex,
        rules=tuple(
            FeaturePreprocessingRule(
                feature_name=name,
                missing_strategy=MissingValueStrategy.REQUIRE_PRESENT,
                add_missing_indicator=False,
                transform=NumericTransform.IDENTITY,
            )
            for name in selected_names
        ),
    )
    preprocessing = resolve_preprocessing_spec(selection, preprocessing_spec)

    model_plan = ModelReviewPlan(
        name="synthetic-full-run",
        description="Real ProcureLens detector/review integration over synthetic federal-contract evidence.",
        feature_selection=selection,
        preprocessing=preprocessing,
        isolation_forest_runs=(
            IsolationForestRunPlan(
                run_name="if-seed-11",
                config=IsolationForestConfig(
                    n_estimators=32,
                    max_samples="auto",
                    max_features=Decimal("1"),
                    bootstrap=False,
                    random_state=11,
                    n_jobs=1,
                ),
                primary=True,
            ),
            IsolationForestRunPlan(
                run_name="if-seed-29",
                config=IsolationForestConfig(
                    n_estimators=32,
                    max_samples="auto",
                    max_features=Decimal("1"),
                    bootstrap=False,
                    random_state=29,
                    n_jobs=1,
                ),
                primary=False,
            ),
        ),
        calibration=ScoreCalibrationSpec(
            name="synthetic-score-calibration",
            quantile_method=QuantileMethod.LINEAR_TYPE7,
        ),
        ensemble=TwoDetectorEnsemblePlan(
            method=EnsembleMethod.WEIGHTED_MEAN,
            isolation_forest_weight=Decimal("0.5"),
            empirical_tail_weight=Decimal("0.5"),
        ),
        build_stability_report=True,
        review_policy=ReviewPolicySpec(
            name="synthetic-top-five",
            description="Select exactly five rows for deterministic review testing.",
            method=ReviewSelectionMethod.TOP_N,
            top_n=5,
            tie_policy=ReviewTiePolicy.DETERMINISTIC_IDENTITY,
        ),
        explanation=ExplanationSpec(
            name="synthetic-review-explanation",
            description="Surface selected factual evidence without causal attribution.",
            catalog_sha256=catalog.sha256_hex,
            feature_names=selected_names[: min(5, len(selected_names))],
            include_missing_features=True,
        ),
        exports=(
            ExportSerializationSpec(
                format=ExportFormat.JSON,
                json_pretty=False,
            ),
            ExportSerializationSpec(
                format=ExportFormat.CSV,
                csv_safety_mode=CsvSafetyMode.PROGRAMMATIC,
            ),
        ),
    )

    result = run_procurelens_analysis(
        reference_transactions=population,
        scoring_transactions=population,
        feature_plan=feature_plan,
        model_plan=model_plan,
        run_name="synthetic-full-integration",
        source_revision="synthetic-fixture-v1",
        software_components={"fixture": "1"},
    )

    assert result.training_features.row_count == len(population)
    assert result.scoring_features is result.training_features
    assert result.model_review.review_scores.row_count == len(population)
    assert result.model_review.review_selection.selected_count == 5
    assert result.model_review.stability is not None
    assert len(result.model_review.isolation_forest_runs) == 2

    for row in result.model_review.review_scores.rows:
        assert Decimal(0) <= row.review_score <= Decimal(100)
        assert row.review_score_lower <= row.review_score <= row.review_score_upper

    exports = result.model_review.serialized_exports
    assert tuple(item.format for item in exports) == (ExportFormat.JSON, ExportFormat.CSV)
    json_payload = json.loads(exports[0].payload_text)
    assert json_payload["row_count"] == len(population)
    assert len(json_payload["records"]) == len(population)

    csv_rows = tuple(csv.DictReader(StringIO(exports[1].payload_text)))
    assert len(csv_rows) == len(population)
    assert sum(row["flagged_for_review"] == "Y" for row in csv_rows) == 5

    manifest_index = result.manifest.artifact_index
    assert manifest_index["model_review_run"].sha256_hex == result.model_review.evidence_sha256
    assert manifest_index["export_00_json"].sha256_hex == exports[0].payload_sha256
    assert manifest_index["export_01_csv"].sha256_hex == exports[1].payload_sha256


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


def _feature_plan(catalog_sha256: str) -> FeatureBuildPlan:
    return FeatureBuildPlan(
        name="synthetic-full-feature-plan",
        description="Explicit frozen-reference policy for full-pipeline integration.",
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


def _population() -> tuple[ProcurementTransaction, ...]:
    rows: list[ProcurementTransaction] = []
    for index in range(1, 13):
        base = Decimal(80 + index * index * 19)
        modification = Decimal((index * 13) % 71 - 30)
        vendor_group = (
            "VENDOR-A" if index <= 5 else
            "VENDOR-B" if index <= 8 else
            f"VENDOR-{index}"
        )
        offers = (index % 5) + 1
        rows.append(
            _transaction(
                award_index=index,
                vendor_uei=vendor_group,
                modification_number="0",
                suffix="base",
                action_date=date(2026, 1, 3 + index),
                action_obligation=base,
                award_total=base,
                offers=offers,
            )
        )
        rows.append(
            _transaction(
                award_index=index,
                vendor_uei=vendor_group,
                modification_number="P00001",
                suffix="mod1",
                action_date=date(2026, 3, 2 + index),
                action_obligation=modification,
                award_total=base + modification,
                offers=offers,
            )
        )
    return tuple(rows)


def _transaction(
    *,
    award_index: int,
    vendor_uei: str,
    modification_number: str,
    suffix: str,
    action_date: date,
    action_obligation: Decimal,
    award_total: Decimal,
    offers: int,
) -> ProcurementTransaction:
    transaction_id = f"FULL-TX-{award_index}-{suffix}"
    return ProcurementTransaction(
        lineage=SourceRecordRef(
            source_name="synthetic-full-integration",
            source_transaction_id=transaction_id,
            retrieved_at=datetime(2026, 4, 1, tzinfo=timezone.utc),
            source_schema="synthetic-full-v1",
        ),
        award_id=f"FULL-AWARD-{award_index}",
        transaction_id=transaction_id,
        piid=f"FULL-PIID-{award_index}",
        modification_number=modification_number,
        parent_award_id=None,
        award_type_code="A",
        recipient_name=f"Synthetic Vendor {vendor_uei}",
        recipient_uei=vendor_uei,
        recipient_legacy_id=None,
        parent_recipient_name=None,
        parent_recipient_uei=None,
        parent_recipient_legacy_id=None,
        action_date=action_date,
        action_obligation=action_obligation,
        award_total_obligation=award_total,
        awarding_agency_code="AGENCY",
        awarding_agency_name="Synthetic Agency",
        awarding_subtier_agency_code="SUBTIER",
        awarding_subtier_agency_name="Synthetic Subtier",
        naics_code="541512",
        psc_code="D302",
        extent_competed_code="A",
        extent_competed_description="Full and Open Competition",
        number_of_offers_received=offers,
        other_than_full_and_open_code=None,
        other_than_full_and_open_description=None,
        solicitation_procedure_code="NP",
        solicitation_procedure_description="Negotiated Proposal/Quote",
    )
