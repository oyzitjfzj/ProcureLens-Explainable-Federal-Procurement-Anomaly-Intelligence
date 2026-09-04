from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal

from procurelens.domain.transaction import ProcurementTransaction, SourceRecordRef
from procurelens.features.amount_reference import AmountBasis, AmountReferencePolicy
from procurelens.features.award_change_context import AwardChangeContextSupportSpec
from procurelens.features.award_change_reference import (
    AwardChangeReferencePolicy,
    federal_award_change_reference_plan,
)
from procurelens.features.catalog import FeatureSource, feature_catalog
from procurelens.features.competition_context import CompetitionContextSupportSpec
from procurelens.features.competition_reference import (
    CompetitionReferencePolicy,
    federal_competition_context_plan,
)
from procurelens.features.peer_groups import federal_contract_amount_peer_plan
from procurelens.features.vendor_identity import VendorIdentityScope
from procurelens.features.vendor_market import (
    VendorMarketPolicy,
    federal_vendor_market_plan,
)
from procurelens.features.vendor_market_context import VendorMarketSupportSpec
from procurelens.pipeline.feature_config import FeatureBuildPlan
from procurelens.pipeline.features import build_candidate_features, build_feature_references
from procurelens.statistics.robust import QuantileMethod


def test_real_feature_builders_form_one_frozen_reference_chain() -> None:
    catalog = feature_catalog()
    plan = _feature_plan(catalog.sha256_hex)
    population = _population()

    references = build_feature_references(population, plan, catalog=catalog)
    reference_sha_before = references.evidence_sha256
    batch = build_candidate_features(population, references, catalog=catalog)

    assert references.reference_transaction_count == 8
    assert references.amount.total_transactions == 8
    assert references.vendor.observed_base_awards == 4
    assert references.competition.observed_base_awards == 4
    assert references.award_change.total_awards == 4
    assert references.award_change.eligible_awards == 4

    assert batch.row_count == 8
    assert batch.row_identities == tuple(
        (transaction.transaction_id, transaction.award_id)
        for transaction in population
    )
    assert batch.feature_catalog_sha256 == catalog.sha256_hex
    assert references.evidence_sha256 == reference_sha_before

    expected_sources = set(FeatureSource)
    for row in batch.rows:
        assert set(row.source_evidence_sha256) == expected_sources
        assert all(
            row.source_evidence_sha256[source] is not None
            for source in expected_sources
        )
        assert len(row.values) == len(catalog.entries)
        assert row.available_count + row.missing_count == len(catalog.entries)


def test_reference_bundle_is_order_independent_but_target_batch_is_ordered() -> None:
    catalog = feature_catalog()
    plan = _feature_plan(catalog.sha256_hex)
    population = _population()

    forward = build_feature_references(population, plan, catalog=catalog)
    reverse = build_feature_references(
        tuple(reversed(population)),
        plan,
        catalog=catalog,
    )

    assert forward.reference_population_sha256 == reverse.reference_population_sha256
    assert forward.evidence_sha256 == reverse.evidence_sha256

    forward_batch = build_candidate_features(population, forward, catalog=catalog)
    reverse_batch = build_candidate_features(
        tuple(reversed(population)),
        forward,
        catalog=catalog,
    )

    assert forward_batch.target_population_sha256 != reverse_batch.target_population_sha256
    assert forward_batch.row_identities == tuple(
        (transaction.transaction_id, transaction.award_id)
        for transaction in population
    )
    assert reverse_batch.row_identities == tuple(reversed(forward_batch.row_identities))


def test_external_target_does_not_mutate_frozen_references() -> None:
    catalog = feature_catalog()
    plan = _feature_plan(catalog.sha256_hex)
    references = build_feature_references(_population(), plan, catalog=catalog)
    before = references.evidence_sha256

    external = (
        _transaction(
            award_index=5,
            modification="0",
            transaction_suffix="base",
            action_date=date(2026, 3, 10),
            action_obligation=Decimal("950"),
            award_total=Decimal("950"),
            offers=1,
        ),
    )
    result = build_candidate_features(external, references, catalog=catalog)

    assert result.row_count == 1
    assert references.evidence_sha256 == before
    assert result.reference_bundle_sha256 == before
    assert result.rows[0].transaction_id == external[0].transaction_id


def _feature_plan(catalog_sha256: str) -> FeatureBuildPlan:
    return FeatureBuildPlan(
        name="synthetic-federal-contract-evidence",
        description="Synthetic integration policy exercising every ProcureLens feature family.",
        feature_catalog_sha256=catalog_sha256,
        amount_peer_plan=federal_contract_amount_peer_plan(),
        amount_basis=AmountBasis.ACTION_OBLIGATION,
        amount_minimum_peer_count=2,
        amount_reference_policy=AmountReferencePolicy(),
        vendor_peer_plan=federal_vendor_market_plan(),
        vendor_scope=VendorIdentityScope.ENTITY,
        vendor_support=VendorMarketSupportSpec(
            minimum_observed_new_awards=2,
            minimum_identified_new_awards=2,
            minimum_observed_winning_vendors=2,
            minimum_vendor_identity_coverage=Decimal("0.75"),
        ),
        vendor_market_policy=VendorMarketPolicy(),
        competition_peer_plan=federal_competition_context_plan(),
        competition_support=CompetitionContextSupportSpec(
            minimum_base_awards=2,
            minimum_process_known=1,
            minimum_offers_known=1,
            minimum_procedure_known=1,
            minimum_process_coverage=Decimal("0.25"),
            minimum_offer_coverage=Decimal("0.25"),
            minimum_procedure_coverage=Decimal("0.25"),
        ),
        competition_reference_policy=CompetitionReferencePolicy(),
        award_change_peer_plan=federal_award_change_reference_plan(),
        award_change_support=AwardChangeContextSupportSpec(
            minimum_peer_awards=2,
        ),
        award_change_reference_policy=AwardChangeReferencePolicy(),
        quantile_method=QuantileMethod.LINEAR_TYPE7,
    )


def _population() -> tuple[ProcurementTransaction, ...]:
    base_amounts = (
        Decimal("100"),
        Decimal("240"),
        Decimal("380"),
        Decimal("620"),
    )
    modification_amounts = (
        Decimal("15"),
        Decimal("-25"),
        Decimal("60"),
        Decimal("-80"),
    )
    offers = (3, 1, 4, 2)
    rows: list[ProcurementTransaction] = []
    for index in range(1, 5):
        base = base_amounts[index - 1]
        rows.append(
            _transaction(
                award_index=index,
                modification="0",
                transaction_suffix="base",
                action_date=date(2026, 1, 4 + index),
                action_obligation=base,
                award_total=base,
                offers=offers[index - 1],
            )
        )
        rows.append(
            _transaction(
                award_index=index,
                modification="P00001",
                transaction_suffix="mod1",
                action_date=date(2026, 2, 10 + index),
                action_obligation=modification_amounts[index - 1],
                award_total=base + modification_amounts[index - 1],
                offers=offers[index - 1],
            )
        )
    return tuple(rows)


def _transaction(
    *,
    award_index: int,
    modification: str,
    transaction_suffix: str,
    action_date: date,
    action_obligation: Decimal,
    award_total: Decimal,
    offers: int,
) -> ProcurementTransaction:
    transaction_id = f"TX-{award_index}-{transaction_suffix}"
    award_id = f"AWARD-{award_index}"
    vendor = f"VENDOR-{award_index}"
    return ProcurementTransaction(
        lineage=SourceRecordRef(
            source_name="synthetic-integration",
            source_transaction_id=transaction_id,
            retrieved_at=datetime(2026, 3, 20, tzinfo=timezone.utc),
            source_schema="synthetic-v1",
        ),
        award_id=award_id,
        transaction_id=transaction_id,
        piid=f"PIID-{award_index}",
        modification_number=modification,
        parent_award_id=None,
        award_type_code="A",
        recipient_name=f"Synthetic Vendor {award_index}",
        recipient_uei=vendor,
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
