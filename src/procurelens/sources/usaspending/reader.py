"""Schema-faithful streaming reader for verified USAspending tabular archives.

This module owns tabular decoding only. It does not interpret procurement
semantics, normalize source fields into the domain model, score anomalies, or
apply risk policy.

The reader consumes a verified artifact produced by ``artifact.py`` and emits
immutable row mappings with explicit archive/member/row provenance. It rejects
ambiguous headers and row-shape drift instead of relying on ``csv.DictReader``
behaviour that can silently overwrite duplicate column names.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
import csv
from dataclasses import dataclass
from hashlib import sha256
from io import TextIOWrapper
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any
from zipfile import BadZipFile, ZipFile

from procurelens.sources.usaspending.artifact import ArtifactReceipt


class USAspendingReaderError(RuntimeError):
    """Base error for verified-artifact tabular reading failures."""


class ArchiveChangedError(USAspendingReaderError):
    """Raised when the artifact no longer matches its verification receipt."""


class UnsupportedMemberError(USAspendingReaderError):
    """Raised when an archive member has no configured tabular format."""


class HeaderIntegrityError(USAspendingReaderError):
    """Raised when a header is missing, blank, duplicate, or ambiguous."""


class RowIntegrityError(USAspendingReaderError):
    """Raised when a row cannot be represented without information loss."""


class ReaderLimitError(USAspendingReaderError):
    """Raised when a configured decoding/resource budget is exceeded."""


@dataclass(frozen=True, slots=True)
class TabularFormat:
    """Explicit parsing contract for one filename suffix."""

    suffix: str
    delimiter: str
    name: str

    def __post_init__(self) -> None:
        suffix = self.suffix.strip().casefold()
        if not suffix.startswith(".") or len(suffix) < 2:
            raise ValueError("suffix must look like '.csv'")
        if len(self.delimiter) != 1:
            raise ValueError("delimiter must be exactly one character")
        name = self.name.strip()
        if not name:
            raise ValueError("format name must not be blank")
        object.__setattr__(self, "suffix", suffix)
        object.__setattr__(self, "name", name)


# USAspending's current download lookup defines these exact formats:
# csv -> ',', tsv -> '\t', pstxt -> '|' with a .txt extension.
_DEFAULT_FORMATS: tuple[TabularFormat, ...] = (
    TabularFormat(".csv", ",", "csv"),
    TabularFormat(".tsv", "\t", "tsv"),
    TabularFormat(".txt", "|", "pstxt"),
)


@dataclass(frozen=True, slots=True)
class ReaderPolicy:
    """Configurable, non-semantic safety policy for tabular decoding."""

    encoding: str = "utf-8-sig"
    decoding_errors: str = "strict"
    formats: tuple[TabularFormat, ...] = _DEFAULT_FORMATS
    max_header_columns: int = 2048
    max_cell_characters: int | None = 128 * 1024
    max_rows_per_member: int | None = None
    max_rows_total: int | None = None
    reject_blank_rows: bool = False
    require_rectangular_rows: bool = True
    strip_header_whitespace: bool = True
    verify_receipt_size: bool = True
    verify_member_inventory: bool = True
    verify_receipt_sha256: bool = False
    require_tabular_members: bool = True

    def __post_init__(self) -> None:
        if not self.encoding.strip():
            raise ValueError("encoding must not be blank")
        if self.decoding_errors not in {
            "strict",
            "ignore",
            "replace",
            "surrogateescape",
            "backslashreplace",
        }:
            raise ValueError("unsupported decoding_errors policy")
        if not self.formats:
            raise ValueError("at least one tabular format is required")

        suffixes = [item.suffix for item in self.formats]
        if len(set(suffixes)) != len(suffixes):
            raise ValueError("tabular format suffixes must be unique")
        if self.max_header_columns < 1:
            raise ValueError("max_header_columns must be positive")

        for name in (
            "max_cell_characters",
            "max_rows_per_member",
            "max_rows_total",
        ):
            value = getattr(self, name)
            if value is not None and value < 1:
                raise ValueError(f"{name} must be positive or None")


@dataclass(frozen=True, slots=True)
class RowProvenance:
    """Stable location of a parsed row inside a verified artifact."""

    artifact_sha256: str
    member_name: str
    member_index: int
    record_index: int
    csv_line_number: int
    schema_sha256: str


@dataclass(frozen=True, slots=True)
class TabularRecord:
    """One immutable source row plus its provenance."""

    values: Mapping[str, str]
    provenance: RowProvenance

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "values",
            MappingProxyType(
                {str(key): str(value) for key, value in self.values.items()}
            ),
        )


@dataclass(frozen=True, slots=True)
class MemberSchema:
    """Observed header contract for one tabular archive member."""

    member_name: str
    format_name: str
    delimiter: str
    headers: tuple[str, ...]
    normalized_headers: tuple[str, ...]
    schema_sha256: str
    compressed_bytes: int
    uncompressed_bytes: int


@dataclass(frozen=True, slots=True)
class ArchiveScan:
    """Deterministic tabular-member inventory of one verified artifact."""

    artifact_sha256: str
    members: tuple[MemberSchema, ...]
    ignored_members: tuple[str, ...]

    @property
    def tabular_member_count(self) -> int:
        return len(self.members)


_HEADER_SEPARATOR = re.compile(r"[^a-z0-9]+")


def normalize_header(value: str) -> str:
    """Normalize spelling for collision detection, never semantic mapping."""

    return _HEADER_SEPARATOR.sub("_", value.strip().casefold()).strip("_")


def _schema_digest(
    *,
    format_name: str,
    delimiter: str,
    normalized_headers: tuple[str, ...],
) -> str:
    # Length-prefix components so concatenation cannot create hash ambiguity.
    pieces = [format_name, delimiter, *normalized_headers]
    encoded = b"".join(
        len(piece.encode("utf-8")).to_bytes(8, "big")
        + piece.encode("utf-8")
        for piece in pieces
    )
    return sha256(encoded).hexdigest()


class USAspendingArchiveReader:
    """Stream rows from a verified USAspending ZIP without semantic guessing."""

    __slots__ = ("policy", "_format_by_suffix")

    def __init__(self, policy: ReaderPolicy | None = None) -> None:
        self.policy = policy or ReaderPolicy()
        self._format_by_suffix = MappingProxyType(
            {item.suffix: item for item in self.policy.formats}
        )

    def scan(self, receipt: ArtifactReceipt) -> ArchiveScan:
        """Fingerprint tabular headers without reading all data rows."""

        path = self._validated_path(receipt)
        members: list[MemberSchema] = []
        ignored: list[str] = []

        try:
            with ZipFile(path, "r") as archive:
                self._verify_member_inventory(receipt, archive)
                for info in archive.infolist():
                    if info.is_dir():
                        ignored.append(info.filename)
                        continue
                    format_spec = self._format_for(info.filename)
                    if format_spec is None:
                        ignored.append(info.filename)
                        continue
                    members.append(
                        self._read_member_schema(
                            archive,
                            info.filename,
                            format_spec,
                            compressed_bytes=info.compress_size,
                            uncompressed_bytes=info.file_size,
                        )
                    )
        except BadZipFile as exc:
            raise ArchiveChangedError(
                "verified artifact is no longer a readable ZIP archive"
            ) from exc

        if self.policy.require_tabular_members and not members:
            raise UnsupportedMemberError(
                "verified artifact contains no supported tabular members"
            )

        return ArchiveScan(
            artifact_sha256=receipt.sha256_hex,
            members=tuple(members),
            ignored_members=tuple(ignored),
        )

    def iter_records(
        self,
        receipt: ArtifactReceipt,
        *,
        member_names: tuple[str, ...] | None = None,
    ) -> Iterator[TabularRecord]:
        """Lazily yield immutable rows from selected tabular members.

        ``member_names=None`` means every supported tabular member in archive
        order. An explicit tuple is validated and emitted in that tuple's order,
        making downstream dataset construction reproducible.
        """

        path = self._validated_path(receipt)
        total_rows = 0

        try:
            with ZipFile(path, "r") as archive:
                self._verify_member_inventory(receipt, archive)
                archive_infos = tuple(archive.infolist())
                infos = {info.filename: info for info in archive_infos}
                original_indexes = {
                    info.filename: index for index, info in enumerate(archive_infos)
                }
                selected = self._select_members(infos, member_names)
                if self.policy.require_tabular_members and not selected:
                    raise UnsupportedMemberError(
                        "verified artifact contains no selected tabular members"
                    )

                for name in selected:
                    member_index = original_indexes[name]
                    info = infos[name]
                    format_spec = self._format_for(name)
                    if format_spec is None:
                        raise UnsupportedMemberError(
                            f"archive member is not a supported tabular file: {name}"
                        )

                    for record in self._iter_member_records(
                        receipt=receipt,
                        archive=archive,
                        member_name=name,
                        member_index=member_index,
                        format_spec=format_spec,
                    ):
                        total_rows += 1
                        limit = self.policy.max_rows_total
                        if limit is not None and total_rows > limit:
                            raise ReaderLimitError(
                                "archive row count exceeded "
                                f"max_rows_total={limit}"
                            )
                        yield record
        except BadZipFile as exc:
            raise ArchiveChangedError(
                "verified artifact is no longer a readable ZIP archive"
            ) from exc

    def _validated_path(self, receipt: ArtifactReceipt) -> Path:
        path = Path(receipt.path)
        try:
            stat_result = path.stat()
        except OSError as exc:
            raise ArchiveChangedError(
                f"verified artifact is not accessible: {path}"
            ) from exc
        if not path.is_file():
            raise ArchiveChangedError(
                f"verified artifact is not a regular file: {path}"
            )
        if self.policy.verify_receipt_size and stat_result.st_size != receipt.size_bytes:
            raise ArchiveChangedError(
                "artifact byte size no longer matches its verification receipt"
            )

        if self.policy.verify_receipt_sha256:
            digest = sha256()
            try:
                with path.open("rb") as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(chunk)
            except OSError as exc:
                raise ArchiveChangedError(
                    f"verified artifact could not be re-hashed: {path}"
                ) from exc
            if digest.hexdigest() != receipt.sha256_hex:
                raise ArchiveChangedError(
                    "artifact SHA-256 no longer matches its verification receipt"
                )
        return path

    def _verify_member_inventory(
        self,
        receipt: ArtifactReceipt,
        archive: ZipFile,
    ) -> None:
        if not self.policy.verify_member_inventory:
            return

        observed = tuple(
            (
                info.filename,
                int(info.compress_size),
                int(info.file_size),
                int(info.CRC),
                int(info.compress_type),
                bool(info.is_dir()),
            )
            for info in archive.infolist()
        )
        expected = tuple(
            (
                member.name,
                int(member.compressed_bytes),
                int(member.uncompressed_bytes),
                int(member.crc32),
                int(member.compression_method),
                bool(member.is_directory),
            )
            for member in receipt.archive_members
        )
        if observed != expected:
            raise ArchiveChangedError(
                "ZIP member inventory no longer matches the verification receipt"
            )

    def _select_members(
        self,
        infos: Mapping[str, Any],
        member_names: tuple[str, ...] | None,
    ) -> tuple[str, ...]:
        if member_names is None:
            return tuple(
                name
                for name, info in infos.items()
                if not info.is_dir() and self._format_for(name) is not None
            )

        cleaned: list[str] = []
        seen: set[str] = set()
        for value in member_names:
            if not isinstance(value, str) or not value:
                raise ValueError("member_names must contain non-blank strings")
            if value in seen:
                raise ValueError(f"duplicate requested member: {value}")
            if value not in infos:
                raise UnsupportedMemberError(
                    f"archive member does not exist: {value}"
                )
            if infos[value].is_dir():
                raise UnsupportedMemberError(
                    f"archive member is a directory: {value}"
                )
            if self._format_for(value) is None:
                raise UnsupportedMemberError(
                    f"archive member has unsupported tabular suffix: {value}"
                )
            cleaned.append(value)
            seen.add(value)
        return tuple(cleaned)

    def _format_for(self, member_name: str) -> TabularFormat | None:
        return self._format_by_suffix.get(Path(member_name).suffix.casefold())

    def _read_member_schema(
        self,
        archive: ZipFile,
        member_name: str,
        format_spec: TabularFormat,
        *,
        compressed_bytes: int,
        uncompressed_bytes: int,
    ) -> MemberSchema:
        with archive.open(member_name, "r") as binary:
            with TextIOWrapper(
                binary,
                encoding=self.policy.encoding,
                errors=self.policy.decoding_errors,
                newline="",
            ) as text:
                reader = csv.reader(
                    text,
                    delimiter=format_spec.delimiter,
                    strict=True,
                )
                headers = self._read_headers(reader, member_name)

        normalized = tuple(normalize_header(header) for header in headers)
        return MemberSchema(
            member_name=member_name,
            format_name=format_spec.name,
            delimiter=format_spec.delimiter,
            headers=headers,
            normalized_headers=normalized,
            schema_sha256=_schema_digest(
                format_name=format_spec.name,
                delimiter=format_spec.delimiter,
                normalized_headers=normalized,
            ),
            compressed_bytes=compressed_bytes,
            uncompressed_bytes=uncompressed_bytes,
        )

    def _iter_member_records(
        self,
        *,
        receipt: ArtifactReceipt,
        archive: ZipFile,
        member_name: str,
        member_index: int,
        format_spec: TabularFormat,
    ) -> Iterator[TabularRecord]:
        with archive.open(member_name, "r") as binary:
            with TextIOWrapper(
                binary,
                encoding=self.policy.encoding,
                errors=self.policy.decoding_errors,
                newline="",
            ) as text:
                reader = csv.reader(
                    text,
                    delimiter=format_spec.delimiter,
                    strict=True,
                )
                headers = self._read_headers(reader, member_name)
                normalized = tuple(normalize_header(header) for header in headers)
                schema_sha = _schema_digest(
                    format_name=format_spec.name,
                    delimiter=format_spec.delimiter,
                    normalized_headers=normalized,
                )

                member_rows = 0
                try:
                    for parsed in reader:
                        if self._is_blank_row(parsed):
                            if self.policy.reject_blank_rows:
                                raise RowIntegrityError(
                                    f"{member_name}: blank row at CSV line "
                                    f"{reader.line_num}"
                                )
                            continue

                        member_rows += 1
                        limit = self.policy.max_rows_per_member
                        if limit is not None and member_rows > limit:
                            raise ReaderLimitError(
                                f"{member_name}: row count exceeded "
                                f"max_rows_per_member={limit}"
                            )

                        if (
                            self.policy.require_rectangular_rows
                            and len(parsed) != len(headers)
                        ):
                            raise RowIntegrityError(
                                f"{member_name}: row {member_rows} has "
                                f"{len(parsed)} cells but header has {len(headers)} "
                                f"(CSV line {reader.line_num})"
                            )
                        if len(parsed) > len(headers):
                            raise RowIntegrityError(
                                f"{member_name}: row {member_rows} has extra cells; "
                                "refusing silent truncation"
                            )
                        if len(parsed) < len(headers):
                            parsed = [
                                *parsed,
                                *([""] * (len(headers) - len(parsed))),
                            ]

                        self._check_cell_limits(
                            parsed,
                            member_name=member_name,
                            record_index=member_rows,
                        )
                        values = {
                            header: value
                            for header, value in zip(headers, parsed, strict=True)
                        }
                        yield TabularRecord(
                            values=values,
                            provenance=RowProvenance(
                                artifact_sha256=receipt.sha256_hex,
                                member_name=member_name,
                                member_index=member_index,
                                record_index=member_rows,
                                csv_line_number=reader.line_num,
                                schema_sha256=schema_sha,
                            ),
                        )
                except csv.Error as exc:
                    raise RowIntegrityError(
                        f"{member_name}: CSV parse failure near line "
                        f"{reader.line_num}: {exc}"
                    ) from exc

    def _read_headers(
        self,
        reader: Any,
        member_name: str,
    ) -> tuple[str, ...]:
        try:
            raw = next(reader)
        except StopIteration as exc:
            raise HeaderIntegrityError(
                f"{member_name}: tabular file is empty"
            ) from exc
        except csv.Error as exc:
            raise HeaderIntegrityError(
                f"{member_name}: header parse failure: {exc}"
            ) from exc

        if not raw:
            raise HeaderIntegrityError(f"{member_name}: header row is empty")
        if len(raw) > self.policy.max_header_columns:
            raise ReaderLimitError(
                f"{member_name}: header contains {len(raw)} columns, exceeding "
                f"max_header_columns={self.policy.max_header_columns}"
            )

        headers = tuple(
            value.strip() if self.policy.strip_header_whitespace else value
            for value in raw
        )
        self._check_cell_limits(
            headers,
            member_name=member_name,
            record_index=0,
        )

        blank = [index + 1 for index, value in enumerate(headers) if not value]
        if blank:
            raise HeaderIntegrityError(
                f"{member_name}: blank header columns at positions {blank}"
            )

        seen: dict[str, tuple[int, str]] = {}
        duplicates: list[tuple[str, int, int]] = []
        for index, header in enumerate(headers, start=1):
            normalized = normalize_header(header)
            if not normalized:
                raise HeaderIntegrityError(
                    f"{member_name}: header {header!r} normalizes to blank"
                )
            previous = seen.get(normalized)
            if previous is None:
                seen[normalized] = (index, header)
            else:
                duplicates.append((normalized, previous[0], index))

        if duplicates:
            details = ", ".join(
                f"{name!r} at columns {first}/{second}"
                for name, first, second in duplicates
            )
            raise HeaderIntegrityError(
                f"{member_name}: duplicate/ambiguous headers after "
                f"normalization: {details}"
            )
        return headers

    def _check_cell_limits(
        self,
        values: tuple[str, ...] | list[str],
        *,
        member_name: str,
        record_index: int,
    ) -> None:
        limit = self.policy.max_cell_characters
        if limit is None:
            return
        for column_index, value in enumerate(values, start=1):
            if len(value) > limit:
                location = "header" if record_index == 0 else f"row {record_index}"
                raise ReaderLimitError(
                    f"{member_name}: cell at {location}, column {column_index} "
                    f"exceeds max_cell_characters={limit}"
                )

    @staticmethod
    def _is_blank_row(values: list[str]) -> bool:
        return not values or all(not value.strip() for value in values)
