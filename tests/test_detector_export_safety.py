from __future__ import annotations

import csv
from datetime import date, datetime, timezone
from decimal import Decimal
from io import StringIO

import pytest

from procurelens.detectors.base import (
    DetectorScoreBatch,
    ScoreOrientation,
    build_detector_scores,
    orient_score,
)
from procurelens.detectors.calibration import (
    ScoreCalibrationError,
    ScoreCalibrationSpec,
    apply_score_calibration,
    fit_score_calibration,
)
from procurelens.export.records import ExportRecordBatch, ProcureLensExportRecord
from procurelens.export.writer import (
    CsvSafetyMode,
    ExportFormat,
    ExportSerializationSpec,
    serialize_export,
)
from procurelens.review.explanation import EXPLANATION_SEMANTICS
from procurelens.review.score import SCORE_SEMANTICS
from procurelens.statistics.robust import QuantileMethod


_SHA_A = "a" * 64
_SHA_B = "b" * 64
_SHA_C = "c" * 64
_SHA_D = "d" * 64
_SHA_E = "e" * 64


def test_detector_score_orientation_is_explicit_and_monotonic() -> None:
    assert orient_score(-0.75, ScoreOrientation.LOWER_IS_MORE_ANOMALOUS) == 0.75
    assert orient_score(0.75, ScoreOrientation.HIGHER_IS_MORE_ANOMALOUS) == 0.75

    scores = build_detector_scores(
        (("TX-1", "AWARD-1"), ("TX-2", "AWARD-2")),
        (-0.1, -0.9),
        orientation=ScoreOrientation.LOWER_IS_MORE_ANOMALOUS,
    )
    assert scores[0].raw_score == -0.1
    assert scores[0].anomaly_score == 0.1
    assert scores[1].anomaly_score == 0.9


def test_calibration_preserves_tie_interval_and_out_of_range_tail_distance() -> None:
    training = _score_batch(
        (1.0, 2.0, 2.0, 4.0),
        identities=(
            ("TRAIN-1", "AWARD-1"),
            ("TRAIN-2", "AWARD-2"),
            ("TRAIN-3", "AWARD-3"),
            ("TRAIN-4", "AWARD-4"),
        ),
        scoring_matrix_sha=_SHA_C,
    )
    calibration = fit_score_calibration(
        training,
        ScoreCalibrationSpec(
            name="regression-calibration",
            quantile_method=QuantileMethod.LINEAR_TYPE7,
        ),
    )

    scoring = _score_batch(
        (2.0, 8.0),
        identities=(("SCORE-TIE", "AWARD-T"), ("SCORE-TAIL", "AWARD-X")),
        scoring_matrix_sha=_SHA_D,
    )
    calibrated = apply_score_calibration(calibration, scoring)
    tie, tail = calibrated.scores

    assert tie.empirical_lower_fraction < tie.empirical_midpoint_fraction
    assert tie.empirical_midpoint_fraction < tie.empirical_upper_fraction
    assert tail.empirical_lower_fraction == Decimal(1)
    assert tail.empirical_midpoint_fraction == Decimal(1)
    assert tail.empirical_upper_fraction == Decimal(1)
    assert tail.modified_z is not None
    assert tie.modified_z is not None
    assert tail.modified_z > tie.modified_z
    assert tail.iqr_distance is not None
    assert tie.iqr_distance is not None
    assert tail.iqr_distance > tie.iqr_distance


def test_calibration_cannot_be_fitted_from_nontraining_score_population() -> None:
    nontraining = _score_batch(
        (1.0, 2.0),
        identities=(("TX-1", "AWARD-1"), ("TX-2", "AWARD-2")),
        scoring_matrix_sha=_SHA_D,
    )
    with pytest.raises(ScoreCalibrationError, match="exact training matrix"):
        fit_score_calibration(
            nontraining,
            ScoreCalibrationSpec(
                name="must-be-training-only",
                quantile_method=QuantileMethod.LINEAR_TYPE7,
            ),
        )


def test_export_is_deterministic_and_spreadsheet_safety_does_not_corrupt_money() -> None:
    batch = ExportRecordBatch(
        explanation_batch_sha256=_SHA_A,
        records=(_export_record(),),
    )

    json_spec = ExportSerializationSpec(format=ExportFormat.JSON)
    first_json = serialize_export(batch, json_spec)
    second_json = serialize_export(batch, json_spec)
    assert first_json.payload_text == second_json.payload_text
    assert first_json.payload_sha256 == second_json.payload_sha256

    programmatic = serialize_export(
        batch,
        ExportSerializationSpec(
            format=ExportFormat.CSV,
            csv_safety_mode=CsvSafetyMode.PROGRAMMATIC,
        ),
    )
    spreadsheet = serialize_export(
        batch,
        ExportSerializationSpec(
            format=ExportFormat.CSV,
            csv_safety_mode=CsvSafetyMode.SPREADSHEET,
        ),
    )

    raw_row = next(csv.DictReader(StringIO(programmatic.payload_text)))
    safe_row = next(csv.DictReader(StringIO(spreadsheet.payload_text)))

    assert raw_row["vendor_name"] == "=HYPERLINK(\"https://example.invalid\")"
    assert safe_row["vendor_name"].startswith("\t=")
    # Numeric deobligation stays numeric text; spreadsheet safety applies only to text-origin cells.
    assert raw_row["action_obligation"] == "-1000"
    assert safe_row["action_obligation"] == "-1000"
    assert raw_row["risk_score_0_100"] == raw_row["review_priority_score"]
    assert safe_row["risk_score_0_100"] == safe_row["review_priority_score"]


def _score_batch(
    raw_scores: tuple[float, ...],
    *,
    identities: tuple[tuple[str, str], ...],
    scoring_matrix_sha: str,
) -> DetectorScoreBatch:
    return DetectorScoreBatch(
        detector_name="synthetic_detector",
        detector_family="synthetic_family",
        implementation_name="procurelens-test",
        implementation_version="1",
        score_orientation=ScoreOrientation.HIGHER_IS_MORE_ANOMALOUS,
        config_sha256=_SHA_A,
        fitted_model_sha256=_SHA_B,
        training_matrix_sha256=_SHA_C,
        scoring_matrix_sha256=scoring_matrix_sha,
        preprocessor_sha256=_SHA_E,
        output_feature_names=("synthetic_feature",),
        scores=build_detector_scores(
            identities,
            raw_scores,
            orientation=ScoreOrientation.HIGHER_IS_MORE_ANOMALOUS,
        ),
    )


def _export_record() -> ProcureLensExportRecord:
    return ProcureLensExportRecord(
        transaction_id="TX-EXPORT-1",
        award_id="AWARD-EXPORT-1",
        piid="PIID-EXPORT-1",
        modification_number="P00001",
        action_date=date(2026, 9, 1),
        vendor_name='=HYPERLINK("https://example.invalid")',
        vendor_uei="UEI-EXPORT-1",
        vendor_legacy_id=None,
        awarding_agency_name="Synthetic Agency",
        awarding_subtier_agency_name="Synthetic Subagency",
        psc_code="D302",
        naics_code="541512",
        award_type_code="A",
        action_obligation=Decimal("-1000"),
        award_total_obligation=Decimal("9000"),
        extent_competed_description="Full and Open Competition",
        number_of_offers_received=1,
        solicitation_procedure_description="Negotiated Proposal/Quote",
        other_than_full_and_open_description=None,
        review_priority_score_lower=Decimal("70"),
        review_priority_score=Decimal("75"),
        review_priority_score_upper=Decimal("80"),
        anomaly_position=Decimal("0.75"),
        score_semantics=SCORE_SEMANTICS,
        flagged_for_review=True,
        review_rank_lower=1,
        review_rank_upper=1,
        selection_reason="explicit regression review policy",
        detector_disagreement_points=Decimal("12.5"),
        feature_completeness_fraction=Decimal("0.8"),
        stability_available=False,
        stability_position_span_points=None,
        stability_median_absolute_deviation_points=None,
        explanation_summary="Evidence requires human review; no misconduct conclusion.",
        explanation_semantics=EXPLANATION_SEMANTICS,
        evidence_facts=(),
        source_name="synthetic-source",
        source_transaction_id="TX-EXPORT-1",
        source_schema="synthetic-v1",
        source_retrieved_at=datetime(2026, 9, 4, 12, 0, tzinfo=timezone.utc),
        raw_record_sha256=None,
        transaction_evidence_sha256=_SHA_B,
        explanation_evidence_sha256=_SHA_C,
    )
