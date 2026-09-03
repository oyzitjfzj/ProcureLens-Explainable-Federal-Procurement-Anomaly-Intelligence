"""Deterministic JSON/CSV serialization for ProcureLens export records.

Serializes an already-validated ExportRecordBatch without changing anomaly
semantics or source evidence. JSON preserves the full structured record. CSV is a
flat investigator/client view with explicit score semantics and a canonical JSON
cell containing structured evidence facts.

CSV spreadsheet safety is explicit. PROGRAMMATIC mode preserves text exactly.
SPREADSHEET mode neutralizes formula-leading *text* cells for human viewing in
spreadsheet software; numeric fields are never rewritten as text.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
from io import StringIO
import json
from typing import Any

from procurelens.export.records import ExportRecordBatch, ProcureLensExportRecord


class ExportWriterError(ValueError):
    """Raised when serialization policy or export payload is inconsistent."""


EXPORT_SCHEMA_NAME = "procurelens_review_export"
EXPORT_SCHEMA_VERSION = 1
JSON_MEDIA_TYPE = "application/json"
CSV_MEDIA_TYPE = "text/csv"
CSV_LINE_TERMINATOR = "\n"
CSV_DIALECT_DESCRIPTION = "utf8_rfc4180_style_quote_all_lf"
SPREADSHEET_PREFIX = "\t"
_FORMULA_PREFIXES = frozenset(
    (
        "=",
        "+",
        "-",
        "@",
        "\t",
        "\r",
        "\n",
        "\N{FULLWIDTH EQUALS SIGN}",
        "\N{FULLWIDTH PLUS SIGN}",
        "\N{FULLWIDTH HYPHEN-MINUS}",
        "\N{FULLWIDTH COMMERCIAL AT}",
    )
)


class ExportFormat(str, Enum):
    JSON = "json"
    CSV = "csv"


class CsvSafetyMode(str, Enum):
    PROGRAMMATIC = "programmatic"
    SPREADSHEET = "spreadsheet"


@dataclass(frozen=True, slots=True)
class ExportSerializationSpec:
    """Explicit, fingerprinted serialization policy."""

    format: ExportFormat
    csv_safety_mode: CsvSafetyMode | None = None
    json_pretty: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "format", ExportFormat(self.format))
        if not isinstance(self.json_pretty, bool):
            raise ExportWriterError("json_pretty must be bool")

        if self.format is ExportFormat.CSV:
            if self.csv_safety_mode is None:
                raise ExportWriterError(
                    "CSV serialization requires explicit csv_safety_mode"
                )
            object.__setattr__(
                self, "csv_safety_mode", CsvSafetyMode(self.csv_safety_mode)
            )
            if self.json_pretty:
                raise ExportWriterError(
                    "json_pretty is not valid for CSV serialization"
                )
        elif self.csv_safety_mode is not None:
            raise ExportWriterError(
                "csv_safety_mode is only valid for CSV serialization"
            )

    @property
    def sha256_hex(self) -> str:
        return _digest(self.as_dict())

    def as_dict(self) -> dict[str, Any]:
        return {
            "format": self.format.value,
            "csv_safety_mode": (
                None
                if self.csv_safety_mode is None
                else self.csv_safety_mode.value
            ),
            "json_pretty": self.json_pretty,
            "export_schema_name": EXPORT_SCHEMA_NAME,
            "export_schema_version": EXPORT_SCHEMA_VERSION,
        }


@dataclass(frozen=True, slots=True)
class SerializedExport:
    """Immutable UTF-8 export payload plus exact source/spec provenance."""

    format: ExportFormat
    media_type: str
    source_record_batch_sha256: str
    serialization_spec_sha256: str
    row_count: int
    payload_text: str
    payload_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "format", ExportFormat(self.format))
        media = self.media_type.strip()
        if not media:
            raise ExportWriterError("media_type must not be blank")
        object.__setattr__(self, "media_type", media)

        for field_name in (
            "source_record_batch_sha256",
            "serialization_spec_sha256",
            "payload_sha256",
        ):
            object.__setattr__(
                self,
                field_name,
                _digest_hex(getattr(self, field_name), field_name),
            )

        if (
            isinstance(self.row_count, bool)
            or not isinstance(self.row_count, int)
            or self.row_count < 1
        ):
            raise ExportWriterError("row_count must be positive integer")
        if not isinstance(self.payload_text, str) or not self.payload_text:
            raise ExportWriterError("payload_text must be non-empty str")
        if not self.payload_text.endswith("\n"):
            raise ExportWriterError(
                "serialized export payload must end with one newline"
            )
        if _bytes_digest(self.payload_bytes) != self.payload_sha256:
            raise ExportWriterError(
                "payload_sha256 differs from UTF-8 payload bytes"
            )

        expected_media = (
            JSON_MEDIA_TYPE
            if self.format is ExportFormat.JSON
            else CSV_MEDIA_TYPE
        )
        if self.media_type != expected_media:
            raise ExportWriterError(
                "media_type differs from export format"
            )

    @property
    def payload_bytes(self) -> bytes:
        return self.payload_text.encode("utf-8")

    @property
    def byte_count(self) -> int:
        return len(self.payload_bytes)

    @property
    def evidence_sha256(self) -> str:
        return _digest(
            {
                "format": self.format.value,
                "media_type": self.media_type,
                "source_record_batch_sha256":
                    self.source_record_batch_sha256,
                "serialization_spec_sha256":
                    self.serialization_spec_sha256,
                "row_count": self.row_count,
                "payload_sha256": self.payload_sha256,
                "byte_count": self.byte_count,
            }
        )


CSV_COLUMNS = (
    "transaction_id",
    "award_id",
    "piid",
    "modification_number",
    "action_date",
    "vendor_name",
    "vendor_uei",
    "vendor_legacy_id",
    "awarding_agency_name",
    "awarding_subtier_agency_name",
    "psc_code",
    "naics_code",
    "award_type_code",
    "action_obligation",
    "award_total_obligation",
    "extent_competed_description",
    "number_of_offers_received",
    "solicitation_procedure_description",
    "other_than_full_and_open_description",
    "review_priority_score_lower",
    "review_priority_score",
    "review_priority_score_upper",
    "risk_score_0_100",
    "anomaly_position",
    "score_semantics",
    "flagged_for_review",
    "review_rank_lower",
    "review_rank_upper",
    "review_rank",
    "selection_reason",
    "detector_disagreement_points",
    "feature_completeness_fraction",
    "stability_available",
    "stability_position_span_points",
    "stability_median_absolute_deviation_points",
    "explanation_summary",
    "explanation_semantics",
    "evidence_facts_json",
    "source_name",
    "source_transaction_id",
    "source_schema",
    "source_retrieved_at",
    "raw_record_sha256",
    "transaction_evidence_sha256",
    "explanation_evidence_sha256",
    "record_evidence_sha256",
)


_TEXT_COLUMNS = frozenset(
    {
        "transaction_id",
        "award_id",
        "piid",
        "modification_number",
        "vendor_name",
        "vendor_uei",
        "vendor_legacy_id",
        "awarding_agency_name",
        "awarding_subtier_agency_name",
        "psc_code",
        "naics_code",
        "award_type_code",
        "extent_competed_description",
        "solicitation_procedure_description",
        "other_than_full_and_open_description",
        "score_semantics",
        "selection_reason",
        "explanation_summary",
        "explanation_semantics",
        "evidence_facts_json",
        "source_name",
        "source_transaction_id",
        "source_schema",
        "raw_record_sha256",
        "transaction_evidence_sha256",
        "explanation_evidence_sha256",
        "record_evidence_sha256",
    }
)


def serialize_export(
    batch: ExportRecordBatch,
    spec: ExportSerializationSpec,
) -> SerializedExport:
    """Serialize a validated export batch without changing record semantics."""

    if not isinstance(batch, ExportRecordBatch):
        raise TypeError("batch must be ExportRecordBatch")
    if not isinstance(spec, ExportSerializationSpec):
        raise TypeError("spec must be ExportSerializationSpec")

    if spec.format is ExportFormat.JSON:
        text = _serialize_json(batch, pretty=spec.json_pretty)
        media_type = JSON_MEDIA_TYPE
    else:
        assert spec.csv_safety_mode is not None
        text = _serialize_csv(batch, safety=spec.csv_safety_mode)
        media_type = CSV_MEDIA_TYPE

    payload_sha = _bytes_digest(text.encode("utf-8"))
    return SerializedExport(
        format=spec.format,
        media_type=media_type,
        source_record_batch_sha256=batch.evidence_sha256,
        serialization_spec_sha256=spec.sha256_hex,
        row_count=batch.row_count,
        payload_text=text,
        payload_sha256=payload_sha,
    )


def _serialize_json(batch: ExportRecordBatch, *, pretty: bool) -> str:
    payload = {
        "schema": {
            "name": EXPORT_SCHEMA_NAME,
            "version": EXPORT_SCHEMA_VERSION,
        },
        "source_record_batch_sha256": batch.evidence_sha256,
        "row_count": batch.row_count,
        "records": [
            record.as_dict(include_sha=True) for record in batch.records
        ],
    }
    if pretty:
        rendered = json.dumps(
            payload,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
    else:
        rendered = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    return rendered + "\n"


def _serialize_csv(
    batch: ExportRecordBatch,
    *,
    safety: CsvSafetyMode,
) -> str:
    buffer = StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=list(CSV_COLUMNS),
        extrasaction="raise",
        restval="",
        dialect="excel",
        quoting=csv.QUOTE_ALL,
        lineterminator=CSV_LINE_TERMINATOR,
    )
    writer.writeheader()
    for record in batch.records:
        raw = _csv_record(record)
        if set(raw) != set(CSV_COLUMNS):
            raise ExportWriterError(
                "CSV row columns differ from canonical CSV_COLUMNS"
            )
        if safety is CsvSafetyMode.SPREADSHEET:
            raw = {
                name: (
                    _spreadsheet_safe_text(value)
                    if name in _TEXT_COLUMNS
                    else value
                )
                for name, value in raw.items()
            }
        writer.writerow(raw)
    text = buffer.getvalue()
    if not text.endswith("\n"):
        raise ExportWriterError("CSV writer produced non-terminated payload")
    return text


def _csv_record(record: ProcureLensExportRecord) -> dict[str, Any]:
    facts_json = json.dumps(
        [fact.as_dict() for fact in record.evidence_facts],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return {
        "transaction_id": record.transaction_id,
        "award_id": record.award_id,
        "piid": _optional_text(record.piid),
        "modification_number": _optional_text(record.modification_number),
        "action_date": record.action_date.isoformat(),
        "vendor_name": _optional_text(record.vendor_name),
        "vendor_uei": _optional_text(record.vendor_uei),
        "vendor_legacy_id": _optional_text(record.vendor_legacy_id),
        "awarding_agency_name": _optional_text(record.awarding_agency_name),
        "awarding_subtier_agency_name":
            _optional_text(record.awarding_subtier_agency_name),
        "psc_code": _optional_text(record.psc_code),
        "naics_code": _optional_text(record.naics_code),
        "award_type_code": _optional_text(record.award_type_code),
        "action_obligation": str(record.action_obligation),
        "award_total_obligation": _decimal_text(record.award_total_obligation),
        "extent_competed_description":
            _optional_text(record.extent_competed_description),
        "number_of_offers_received": (
            "" if record.number_of_offers_received is None
            else str(record.number_of_offers_received)
        ),
        "solicitation_procedure_description":
            _optional_text(record.solicitation_procedure_description),
        "other_than_full_and_open_description":
            _optional_text(record.other_than_full_and_open_description),
        "review_priority_score_lower":
            str(record.review_priority_score_lower),
        "review_priority_score": str(record.review_priority_score),
        "review_priority_score_upper":
            str(record.review_priority_score_upper),
        # Compatibility alias for the original client-facing "Risk Score"
        # requirement. Semantics remain explicit in score_semantics.
        "risk_score_0_100": str(record.review_priority_score),
        "anomaly_position": str(record.anomaly_position),
        "score_semantics": record.score_semantics,
        "flagged_for_review":
            "Y" if record.flagged_for_review else "N",
        "review_rank_lower": str(record.review_rank_lower),
        "review_rank_upper": str(record.review_rank_upper),
        "review_rank": record.review_rank_text,
        "selection_reason": record.selection_reason,
        "detector_disagreement_points":
            str(record.detector_disagreement_points),
        "feature_completeness_fraction":
            str(record.feature_completeness_fraction),
        "stability_available":
            "Y" if record.stability_available else "N",
        "stability_position_span_points":
            _decimal_text(record.stability_position_span_points),
        "stability_median_absolute_deviation_points":
            _decimal_text(
                record.stability_median_absolute_deviation_points
            ),
        "explanation_summary": record.explanation_summary,
        "explanation_semantics": record.explanation_semantics,
        "evidence_facts_json": facts_json,
        "source_name": record.source_name,
        "source_transaction_id": record.source_transaction_id,
        "source_schema": _optional_text(record.source_schema),
        "source_retrieved_at": record.source_retrieved_at.isoformat(),
        "raw_record_sha256": _optional_text(record.raw_record_sha256),
        "transaction_evidence_sha256":
            record.transaction_evidence_sha256,
        "explanation_evidence_sha256":
            record.explanation_evidence_sha256,
        "record_evidence_sha256": record.evidence_sha256,
    }


def _spreadsheet_safe_text(value: Any) -> Any:
    if not isinstance(value, str) or not value:
        return value
    if value[0] in _FORMULA_PREFIXES:
        return SPREADSHEET_PREFIX + value
    return value


def _optional_text(value: str | None) -> str:
    return "" if value is None else value


def _decimal_text(value: Any) -> str:
    return "" if value is None else str(value)


def _digest_hex(value: str, name: str) -> str:
    digest = value.strip().lower()
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ExportWriterError(f"{name} must be a SHA-256 hex digest")
    return digest


def _bytes_digest(payload: bytes) -> str:
    return sha256(payload).hexdigest()


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
