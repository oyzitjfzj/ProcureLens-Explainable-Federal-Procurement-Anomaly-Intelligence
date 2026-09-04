from __future__ import annotations

import csv
from datetime import datetime, timezone
from hashlib import sha256
from io import StringIO
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pytest

from procurelens.sources.usaspending.artifact import ArchiveMember, ArtifactReceipt
from procurelens.sources.usaspending.loader import (
    DroppedExactDuplicate,
    LoaderPolicy,
    QuarantinedRow,
    RowErrorMode,
    USAspendingDatasetLoader,
)
from procurelens.sources.usaspending.reader import (
    ArchiveChangedError,
    ReaderPolicy,
    USAspendingArchiveReader,
)


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


def test_verified_usaspending_zip_loads_signed_canonical_transaction(tmp_path: Path) -> None:
    receipt = _artifact(
        tmp_path,
        (
            _row(
                transaction_id="TX-NEG-1",
                modification="P00001",
                action_obligation="-25.50",
                award_total="74.50",
                offers="1",
            ),
        ),
    )

    loader = USAspendingDatasetLoader()
    plan = loader.plan(receipt)
    session = loader.open_session(receipt, plan=plan)
    loaded = tuple(session.iter_transactions())

    assert len(loaded) == 1
    item = loaded[0]
    tx = item.transaction
    assert tx.transaction_id == "TX-NEG-1"
    assert tx.award_id == "AWARD-1"
    assert tx.modification_number == "P00001"
    assert str(tx.action_obligation) == "-25.50"
    assert str(tx.award_total_obligation) == "74.50"
    assert tx.number_of_offers_received == 1
    assert tx.extent_competed_description == "Full and Open Competition"
    assert tx.solicitation_procedure_description == "Negotiated Proposal/Quote"
    assert item.provenance.member_name == "contracts.csv"
    assert item.profile_name == "usaspending-contract-transaction-download"
    assert item.additive_schema_headers == ()
    assert session.report.complete is True
    assert session.report.rows_seen == 1
    assert session.report.transactions_emitted == 1
    assert session.report.rows_quarantined == 0


def test_loader_distinguishes_exact_and_conflicting_duplicate_transactions(
    tmp_path: Path,
) -> None:
    first = _row(transaction_id="TX-DUP-1", action_obligation="100")
    exact = dict(first)
    conflicting = dict(first)
    conflicting["federal_action_obligation"] = "125"

    receipt = _artifact(tmp_path, (first, exact, conflicting))
    loader = USAspendingDatasetLoader(
        policy=LoaderPolicy(
            conflicting_duplicate_mode=RowErrorMode.QUARANTINE,
        )
    )
    session = loader.open_session(receipt)
    outcomes = tuple(session.iter_outcomes())

    assert len(outcomes) == 3
    assert outcomes[0].transaction.transaction_id == "TX-DUP-1"
    assert isinstance(outcomes[1], DroppedExactDuplicate)
    assert isinstance(outcomes[2], QuarantinedRow)
    assert outcomes[2].error_type == "ConflictingDuplicateTransaction"
    assert outcomes[2].transaction_id_hint == "TX-DUP-1"
    assert session.report.complete is True
    assert session.report.rows_seen == 3
    assert session.report.transactions_emitted == 1
    assert session.report.exact_duplicates_dropped == 1
    assert session.report.conflicting_duplicates == 1
    assert session.report.rows_quarantined == 1


def test_reader_rejects_same_size_artifact_tampering_before_parsing(tmp_path: Path) -> None:
    receipt = _artifact(tmp_path, (_row(transaction_id="TX-TAMPER-1"),))
    original = bytearray(receipt.path.read_bytes())
    assert original
    original[len(original) // 2] ^= 0x01
    receipt.path.write_bytes(original)
    assert receipt.path.stat().st_size == receipt.size_bytes

    reader = USAspendingArchiveReader(
        ReaderPolicy(verify_receipt_sha256=True)
    )
    with pytest.raises(ArchiveChangedError, match="SHA-256"):
        reader.scan(receipt)


def _artifact(tmp_path: Path, rows: tuple[dict[str, str], ...]) -> ArtifactReceipt:
    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=_HEADERS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)

    path = tmp_path / "usaspending-contracts.zip"
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
        source_url="https://files.usaspending.gov/example.zip",
        final_url="https://files.usaspending.gov/example.zip",
        file_name=path.name,
        size_bytes=len(payload),
        sha256_hex=sha256(payload).hexdigest(),
        downloaded_at=datetime(2026, 9, 4, 12, 0, tzinfo=timezone.utc),
        resumed_from_bytes=0,
        etag='"synthetic-etag"',
        last_modified=None,
        content_type="application/zip",
        request_fingerprint_sha256=sha256(b"synthetic-request").hexdigest(),
        archive_members=members,
        total_uncompressed_bytes=sum(member.uncompressed_bytes for member in members),
    )


def _row(
    *,
    transaction_id: str,
    modification: str = "0",
    action_obligation: str = "100",
    award_total: str = "100",
    offers: str = "3",
) -> dict[str, str]:
    return {
        "contract_award_unique_key": "AWARD-1",
        "contract_transaction_unique_key": transaction_id,
        "award_id_piid": "PIID-1",
        "modification_number": modification,
        "recipient_name": "Synthetic Vendor LLC",
        "recipient_uei": "UEI000000001",
        "action_date": "2026-01-15",
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
