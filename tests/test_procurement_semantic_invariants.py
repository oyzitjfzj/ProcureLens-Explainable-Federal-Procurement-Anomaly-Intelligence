from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal

from procurelens.domain.transaction import ProcurementTransaction, SourceRecordRef
from procurelens.features.award_change_activity import AwardChangeIndex
from procurelens.features.award_lifecycle import AwardActionKind, classify_award_action
from procurelens.features.competition_evidence import (
    CompetitionExtentKind,
    OfferOutcomeKind,
    SolicitationProcedureKind,
    build_competition_evidence,
)
from procurelens.features.vendor_identity import (
    VendorIdentityMethod,
    VendorIdentityScope,
    resolve_vendor_identities,
    resolve_vendor_identity,
)


def test_lifecycle_uses_only_reported_modification_number() -> None:
    zero = classify_award_action(_transaction("TX-BASE", modification="000"))
    modified = classify_award_action(
        _transaction("TX-MOD", modification="p00001", action_date=date(2026, 2, 1))
    )
    unknown = classify_award_action(
        _transaction("TX-UNKNOWN", modification=None, action_date=date(2026, 3, 1))
    )

    assert zero.kind is AwardActionKind.BASE_AWARD
    assert zero.normalized_modification_number == "0"
    assert zero.is_observed_new_award_action is True

    assert modified.kind is AwardActionKind.MODIFICATION
    assert modified.normalized_modification_number == "P00001"
    assert modified.is_observed_new_award_action is False

    assert unknown.kind is AwardActionKind.UNKNOWN
    assert unknown.normalized_modification_number is None
    assert unknown.classification_reason == "modification_number_missing"


def test_vendor_identity_prefers_stable_ids_and_never_crosses_parent_scope() -> None:
    tx = _transaction(
        "TX-ID",
        recipient_name="  Example   Subsidiary, LLC  ",
        recipient_uei="uei-child-1",
        recipient_legacy_id="legacy-child",
        parent_recipient_name=None,
        parent_recipient_uei=None,
        parent_recipient_legacy_id=None,
    )

    entity = resolve_vendor_identity(tx, VendorIdentityScope.ENTITY)
    parent = resolve_vendor_identity(tx, VendorIdentityScope.ULTIMATE_PARENT)
    both = resolve_vendor_identities(tx)

    assert entity is not None
    assert entity.method is VendorIdentityMethod.UEI
    assert entity.value == "UEI-CHILD-1"
    assert entity.display_name == "Example Subsidiary, LLC"
    assert parent is None
    assert both.entity == entity
    assert both.ultimate_parent is None
    assert both.ultimate_parent_unavailable_reason == "ultimate_parent_identity_missing"
    assert both.parent_matches_entity_identifier is None

    name_only = resolve_vendor_identity(
        _transaction(
            "TX-NAME",
            recipient_name="  Acme   Widgets, Inc.  ",
            recipient_uei=None,
            recipient_legacy_id=None,
        )
    )
    assert name_only is not None
    assert name_only.method is VendorIdentityMethod.NORMALIZED_NAME
    # Conservative fallback: whitespace/case normalize only; punctuation is preserved.
    assert name_only.value == "acme widgets, inc."


def test_single_offer_does_not_reclassify_full_and_open_process() -> None:
    evidence = build_competition_evidence(
        _transaction(
            "TX-COMP",
            offers=1,
            extent_description="Full and Open Competition",
            solicitation_description="Only One Source Solicited",
        )
    )

    assert evidence.extent_kind is CompetitionExtentKind.FULL_AND_OPEN
    assert evidence.reported_process_competitive is True
    assert evidence.offer_outcome_kind is OfferOutcomeKind.SINGLE_OFFER
    assert evidence.single_offer_reported is True
    assert evidence.solicitation_procedure_kind is SolicitationProcedureKind.ONLY_ONE_SOURCE
    assert "single_offer_is_outcome_not_process_classification" in evidence.evidence_notes
    assert "competitive_extent_with_only_one_source_procedure" in evidence.evidence_conflicts
    assert evidence.has_conflicting_evidence is True


def test_award_change_money_uses_action_obligations_not_award_total_sums() -> None:
    rows = (
        _transaction(
            "TX-BASE",
            modification="0",
            action_date=date(2026, 1, 1),
            action_obligation=Decimal("100"),
            award_total=Decimal("100"),
        ),
        _transaction(
            "TX-M1",
            modification="P00001",
            action_date=date(2026, 1, 10),
            action_obligation=Decimal("25"),
            award_total=Decimal("125"),
        ),
        _transaction(
            "TX-M2",
            modification="P00002",
            action_date=date(2026, 1, 20),
            action_obligation=Decimal("-10"),
            award_total=Decimal("115"),
        ),
        _transaction(
            "TX-M3",
            modification="P00003",
            action_date=date(2026, 1, 25),
            action_obligation=Decimal("0"),
            award_total=Decimal("115"),
        ),
        _transaction(
            "TX-M4",
            modification="P00004",
            action_date=date(2026, 2, 1),
            action_obligation=Decimal("5"),
            award_total=Decimal("120"),
        ),
    )

    snapshot = AwardChangeIndex().observe_many(rows).snapshot()
    activity = snapshot.get("AWARD-1")

    assert activity is not None
    assert activity.base_award_action_count == 1
    assert activity.modification_action_count == 4
    assert activity.distinct_modification_number_count == 4
    assert activity.positive_modification_obligation == Decimal("30")
    assert activity.deobligation_magnitude == Decimal("10")
    assert activity.net_modification_obligation == Decimal("20")
    assert activity.absolute_modification_obligation_activity == Decimal("40")
    assert activity.zero_modification_obligation_count == 1
    assert activity.maximum_absolute_modification_obligation == Decimal("25")
    # 100 + 125 + 115 + 115 + 120 would be meaningless; award totals are not summed.
    assert activity.net_modification_obligation != Decimal("575")


def _transaction(
    transaction_id: str,
    *,
    modification: str | None = "0",
    action_date: date = date(2026, 1, 15),
    action_obligation: Decimal = Decimal("100"),
    award_total: Decimal = Decimal("100"),
    recipient_name: str | None = "Synthetic Vendor",
    recipient_uei: str | None = "UEI000000001",
    recipient_legacy_id: str | None = None,
    parent_recipient_name: str | None = None,
    parent_recipient_uei: str | None = None,
    parent_recipient_legacy_id: str | None = None,
    offers: int | None = 3,
    extent_description: str | None = "Full and Open Competition",
    solicitation_description: str | None = "Negotiated Proposal/Quote",
) -> ProcurementTransaction:
    return ProcurementTransaction(
        lineage=SourceRecordRef(
            source_name="semantic-regression",
            source_transaction_id=transaction_id,
            retrieved_at=datetime(2026, 9, 4, 12, 0, tzinfo=timezone.utc),
            source_schema="synthetic-v1",
        ),
        award_id="AWARD-1",
        transaction_id=transaction_id,
        piid="PIID-1",
        modification_number=modification,
        parent_award_id=None,
        award_type_code="A",
        recipient_name=recipient_name,
        recipient_uei=recipient_uei,
        recipient_legacy_id=recipient_legacy_id,
        parent_recipient_name=parent_recipient_name,
        parent_recipient_uei=parent_recipient_uei,
        parent_recipient_legacy_id=parent_recipient_legacy_id,
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
        extent_competed_description=extent_description,
        number_of_offers_received=offers,
        other_than_full_and_open_code=None,
        other_than_full_and_open_description=None,
        solicitation_procedure_code="NP",
        solicitation_procedure_description=solicitation_description,
    )
