"""Trustworthy USAspending download-artifact materialization for ProcureLens.

This module owns the data-plane boundary after ``client.py`` has completed the
USAspending control-plane workflow. It streams a finished ZIP download to a
same-directory partial file, resumes only when the remote representation can be
validated, verifies HTTP range semantics and byte counts, hashes the exact
artifact, inspects the ZIP before extraction, and atomically promotes a verified
artifact into place.

No CSV parsing, feature engineering, anomaly detection, or risk policy belongs
here.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from hashlib import sha256
from http.client import HTTPException, IncompleteRead
import json
import os
from pathlib import Path, PurePosixPath
import random
import re
import stat
import time
from types import MappingProxyType
from typing import Any, BinaryIO, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from zipfile import BadZipFile, LargeZipFile, ZipFile

from procurelens.sources.usaspending.client import DownloadJob, DownloadStatus


class ArtifactError(RuntimeError):
    """Base error for USAspending artifact materialization failures."""


class ArtifactHTTPError(ArtifactError):
    """Raised when the artifact transfer cannot complete over HTTP."""


class ArtifactProtocolError(ArtifactError):
    """Raised when HTTP response semantics are unsafe or contradictory."""


class ArtifactIntegrityError(ArtifactError):
    """Raised when downloaded bytes fail integrity or archive validation."""


class ArtifactLockedError(ArtifactError):
    """Raised when another writer already owns the destination artifact."""


class ArtifactLimitError(ArtifactIntegrityError):
    """Raised when a configured download/archive resource budget is exceeded."""


@dataclass(slots=True)
class StreamResponse:
    """Streaming HTTP response with normalized headers."""

    status_code: int
    headers: Mapping[str, str]
    final_url: str
    stream: BinaryIO

    def __post_init__(self) -> None:
        if not (100 <= self.status_code <= 599):
            raise ValueError("status_code must be a valid HTTP status")
        self.headers = MappingProxyType(
            {str(key).casefold(): str(value) for key, value in self.headers.items()}
        )

    def read(self, size: int) -> bytes:
        return self.stream.read(size)

    def close(self) -> None:
        self.stream.close()

    def __enter__(self) -> "StreamResponse":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()


class ArtifactTransport(Protocol):
    """Injectable streaming HTTP boundary."""

    def open(
        self,
        *,
        url: str,
        headers: Mapping[str, str],
        timeout_seconds: float,
    ) -> StreamResponse: ...


class UrllibArtifactTransport:
    """Dependency-free streaming transport using Python's standard library."""

    def open(
        self,
        *,
        url: str,
        headers: Mapping[str, str],
        timeout_seconds: float,
    ) -> StreamResponse:
        request = Request(url=url, headers=dict(headers), method="GET")
        try:
            response = urlopen(request, timeout=timeout_seconds)
            return StreamResponse(
                status_code=int(response.status),
                headers=dict(response.headers.items()),
                final_url=response.geturl(),
                stream=response,
            )
        except HTTPError as exc:
            # HTTPError remains readable and carries response headers, which are
            # required for safe 416 handling and Retry-After back-pressure.
            return StreamResponse(
                status_code=int(exc.code),
                headers=dict(exc.headers.items()) if exc.headers is not None else {},
                final_url=exc.geturl(),
                stream=exc,
            )
        except (URLError, TimeoutError, OSError) as exc:
            raise ArtifactHTTPError(f"artifact network request failed: {exc}") from exc


@dataclass(frozen=True, slots=True)
class TransferPolicy:
    """Configurable transfer safety/reliability policy.

    Defaults are deliberately finite, but none of these values are protocol
    truths. Public deployments can tune them for their storage, network, and
    dataset size without modifying downloader logic.
    """

    timeout_seconds: float = 60.0
    chunk_size_bytes: int = 1024 * 1024
    max_download_bytes: int | None = 4 * 1024 * 1024 * 1024
    max_attempts: int = 4
    base_retry_seconds: float = 0.75
    max_retry_seconds: float = 8.0
    max_total_retry_delay_seconds: float = 20.0
    retry_jitter_ratio: float = 0.25
    retry_status_codes: frozenset[int] = field(
        default_factory=lambda: frozenset({408, 425, 429, 500, 502, 503, 504})
    )
    require_https: bool = True

    def __post_init__(self) -> None:
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.chunk_size_bytes <= 0:
            raise ValueError("chunk_size_bytes must be positive")
        if self.max_download_bytes is not None and self.max_download_bytes <= 0:
            raise ValueError("max_download_bytes must be positive or None")
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if self.base_retry_seconds < 0 or self.max_retry_seconds < 0:
            raise ValueError("retry delays must not be negative")
        if self.max_retry_seconds < self.base_retry_seconds:
            raise ValueError("max_retry_seconds must be >= base_retry_seconds")
        if self.max_total_retry_delay_seconds < 0:
            raise ValueError("max_total_retry_delay_seconds must not be negative")
        if not (0 <= self.retry_jitter_ratio <= 1):
            raise ValueError("retry_jitter_ratio must be between 0 and 1")
        if any(not (400 <= code <= 599) for code in self.retry_status_codes):
            raise ValueError("retry_status_codes must contain HTTP error statuses")


@dataclass(frozen=True, slots=True)
class ArchivePolicy:
    """Configurable ZIP inspection budgets applied before CRC verification."""

    max_members: int = 512
    max_member_uncompressed_bytes: int = 4 * 1024 * 1024 * 1024
    max_total_uncompressed_bytes: int = 8 * 1024 * 1024 * 1024
    max_compression_ratio: float = 1000.0
    verify_crc: bool = True
    reject_encrypted: bool = True
    reject_symlinks: bool = True
    reject_unsafe_paths: bool = True

    def __post_init__(self) -> None:
        if self.max_members < 1:
            raise ValueError("max_members must be positive")
        if self.max_member_uncompressed_bytes <= 0:
            raise ValueError("max_member_uncompressed_bytes must be positive")
        if self.max_total_uncompressed_bytes <= 0:
            raise ValueError("max_total_uncompressed_bytes must be positive")
        if self.max_compression_ratio <= 0:
            raise ValueError("max_compression_ratio must be positive")


@dataclass(frozen=True, slots=True)
class ArtifactSpec:
    """Stable source identity for one remote USAspending ZIP."""

    source_url: str
    file_name: str
    request_fingerprint_sha256: str | None = None

    def __post_init__(self) -> None:
        parsed = urlparse(self.source_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("source_url must be an absolute HTTP(S) URL")

        name = self.file_name.strip()
        if not name or name in {".", ".."}:
            raise ValueError("file_name must not be blank")
        if Path(name).name != name or "/" in name or "\\" in name:
            raise ValueError("file_name must be a plain basename")
        if not name.casefold().endswith(".zip"):
            raise ValueError("USAspending artifact file_name must end in .zip")
        object.__setattr__(self, "file_name", name)

        digest = self.request_fingerprint_sha256
        if digest is not None:
            cleaned = digest.strip().lower()
            if not _is_sha256(cleaned):
                raise ValueError("request_fingerprint_sha256 must be a SHA-256 hex digest")
            object.__setattr__(self, "request_fingerprint_sha256", cleaned)

    @classmethod
    def from_finished_job(
        cls,
        job: DownloadJob,
        status: DownloadStatus,
    ) -> "ArtifactSpec":
        """Bind a control-plane job to a finished data-plane artifact."""

        if status.status != "finished":
            raise ArtifactProtocolError(
                f"download job is not finished; current status is {status.status!r}"
            )
        if status.file_name != job.file_name:
            raise ArtifactProtocolError(
                "download status file_name does not match the originating job"
            )
        source_url = status.file_url or job.file_url
        return cls(
            source_url=source_url,
            file_name=job.file_name,
            request_fingerprint_sha256=job.request_fingerprint.sha256_hex,
        )


@dataclass(frozen=True, slots=True)
class ArchiveMember:
    name: str
    compressed_bytes: int
    uncompressed_bytes: int
    crc32: int
    compression_method: int
    is_directory: bool


@dataclass(frozen=True, slots=True)
class ArtifactReceipt:
    """Evidence emitted only after a verified artifact is atomically promoted."""

    path: Path
    source_url: str
    final_url: str
    file_name: str
    size_bytes: int
    sha256_hex: str
    downloaded_at: datetime
    resumed_from_bytes: int
    etag: str | None
    last_modified: str | None
    content_type: str | None
    request_fingerprint_sha256: str | None
    archive_members: tuple[ArchiveMember, ...]
    total_uncompressed_bytes: int

    @property
    def member_count(self) -> int:
        return len(self.archive_members)


@dataclass(frozen=True, slots=True)
class _PartialMetadata:
    source_url: str
    etag: str | None
    last_modified: str | None
    expected_total_bytes: int | None

    @property
    def validator(self) -> str | None:
        # A strong ETag is preferred. Weak ETags are not safe for If-Range.
        if self.etag and not self.etag.lstrip().casefold().startswith("w/"):
            return self.etag
        return self.last_modified


@dataclass(frozen=True, slots=True)
class _RangeInfo:
    start: int | None
    end: int | None
    total: int | None
    unsatisfied: bool = False


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_CONTENT_RANGE_RE = re.compile(r"^bytes\s+(\d+)-(\d+)/(\d+|\*)$", re.IGNORECASE)
_UNSATISFIED_RANGE_RE = re.compile(r"^bytes\s+\*/(\d+)$", re.IGNORECASE)


def _is_sha256(value: str) -> bool:
    return bool(_SHA256_RE.fullmatch(value))


def _parse_nonnegative_int(value: str | None, header_name: str) -> int | None:
    if value is None:
        return None
    text = value.strip()
    if not text.isdigit():
        raise ArtifactProtocolError(f"{header_name} must be a non-negative integer")
    return int(text)


def _parse_content_range(value: str | None) -> _RangeInfo | None:
    if value is None:
        return None
    text = value.strip()

    unsatisfied = _UNSATISFIED_RANGE_RE.fullmatch(text)
    if unsatisfied:
        return _RangeInfo(
            start=None,
            end=None,
            total=int(unsatisfied.group(1)),
            unsatisfied=True,
        )

    match = _CONTENT_RANGE_RE.fullmatch(text)
    if not match:
        raise ArtifactProtocolError(f"invalid Content-Range header: {value!r}")
    start = int(match.group(1))
    end = int(match.group(2))
    total_text = match.group(3)
    total = None if total_text == "*" else int(total_text)
    if end < start:
        raise ArtifactProtocolError("Content-Range end precedes start")
    if total is not None and total <= end:
        raise ArtifactProtocolError("Content-Range total is inconsistent with end")
    return _RangeInfo(start=start, end=end, total=total)


def _header_text(headers: Mapping[str, str], name: str) -> str | None:
    value = headers.get(name.casefold())
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def _safe_archive_name(name: str) -> bool:
    if not name or "\x00" in name:
        return False

    # ZIP member names are specified with '/', but backslashes are separators
    # on Windows and therefore must participate in traversal validation.
    normalized = name.replace("\\", "/")
    if normalized.startswith("/"):
        return False
    # Reject drive-letter paths such as C:/...
    if re.match(r"^[A-Za-z]:/", normalized):
        return False

    parts = PurePosixPath(normalized).parts
    return bool(parts) and all(part not in {"", ".", ".."} for part in parts)


def _is_zip_symlink(external_attr: int) -> bool:
    mode = (external_attr >> 16) & 0xFFFF
    return bool(mode) and stat.S_ISLNK(mode)


def _fsync_directory(directory: Path) -> None:
    """Best-effort directory durability after atomic metadata/file replacement."""

    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    try:
        fd = os.open(directory, flags)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        # Some platforms/filesystems do not support directory fsync.
        pass
    finally:
        os.close(fd)


def _atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    temp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    try:
        with open(temp, "xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
        _fsync_directory(path.parent)
    finally:
        try:
            temp.unlink()
        except FileNotFoundError:
            pass


def _hash_file(path: Path, chunk_size: int) -> tuple[Any, int]:
    digest = sha256()
    total = 0
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
            total += len(chunk)
    return digest, total


class USAspendingArtifactStore:
    """Materialize verified USAspending ZIP artifacts into a local directory."""

    __slots__ = (
        "_transport",
        "_transfer_policy",
        "_archive_policy",
        "_sleep",
        "_random",
        "_clock",
        "user_agent",
    )

    def __init__(
        self,
        *,
        transport: ArtifactTransport | None = None,
        transfer_policy: TransferPolicy | None = None,
        archive_policy: ArchivePolicy | None = None,
        user_agent: str = "ProcureLens/0",
        sleep: Callable[[float], None] = time.sleep,
        random_source: Callable[[], float] = random.random,
        clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        if not user_agent.strip():
            raise ValueError("user_agent must not be blank")
        self._transport = transport or UrllibArtifactTransport()
        self._transfer_policy = transfer_policy or TransferPolicy()
        self._archive_policy = archive_policy or ArchivePolicy()
        self._sleep = sleep
        self._random = random_source
        self._clock = clock
        self.user_agent = user_agent.strip()

    def materialize_finished_job(
        self,
        job: DownloadJob,
        status: DownloadStatus,
        directory: str | os.PathLike[str],
        *,
        overwrite: bool = False,
    ) -> ArtifactReceipt:
        """Materialize a finished USAspending download job."""

        return self.materialize(
            ArtifactSpec.from_finished_job(job, status),
            directory,
            overwrite=overwrite,
        )

    def materialize(
        self,
        spec: ArtifactSpec,
        directory: str | os.PathLike[str],
        *,
        overwrite: bool = False,
    ) -> ArtifactReceipt:
        """Download, verify, and atomically promote one ZIP artifact."""

        self._validate_url(spec.source_url)
        target_dir = Path(directory)
        target_dir.mkdir(parents=True, exist_ok=True)
        if not target_dir.is_dir():
            raise ArtifactError(f"artifact directory is not a directory: {target_dir}")

        destination = target_dir / spec.file_name
        partial = target_dir / f".{spec.file_name}.part"
        metadata_path = target_dir / f".{spec.file_name}.part.json"
        lock_path = target_dir / f".{spec.file_name}.lock"

        if destination.exists() and not overwrite:
            raise FileExistsError(
                f"artifact already exists; refusing silent replacement: {destination}"
            )

        self._acquire_lock(lock_path, spec)
        try:
            receipt = self._download_with_retries(
                spec=spec,
                destination=destination,
                partial=partial,
                metadata_path=metadata_path,
                overwrite=overwrite,
            )
            return receipt
        finally:
            try:
                lock_path.unlink()
            except FileNotFoundError:
                pass
            _fsync_directory(target_dir)

    def _download_with_retries(
        self,
        *,
        spec: ArtifactSpec,
        destination: Path,
        partial: Path,
        metadata_path: Path,
        overwrite: bool,
    ) -> ArtifactReceipt:
        policy = self._transfer_policy
        total_retry_delay = 0.0
        last_error: BaseException | None = None

        for attempt in range(1, policy.max_attempts + 1):
            try:
                return self._download_once(
                    spec=spec,
                    destination=destination,
                    partial=partial,
                    metadata_path=metadata_path,
                    overwrite=overwrite,
                )
            except ArtifactIntegrityError:
                # Corrupt or unsafe bytes are not a transient network condition.
                self._discard_partial(partial, metadata_path)
                raise
            except _RetryableHTTPStatus as exc:
                last_error = exc
                if attempt >= policy.max_attempts:
                    break
                delay = self._retry_delay(
                    attempt=attempt,
                    retry_after=exc.retry_after,
                    total_retry_delay=total_retry_delay,
                )
                self._sleep(delay)
                total_retry_delay += delay
            except (ArtifactHTTPError, HTTPException, IncompleteRead) as exc:
                last_error = exc
                if attempt >= policy.max_attempts:
                    break
                delay = self._retry_delay(
                    attempt=attempt,
                    retry_after=None,
                    total_retry_delay=total_retry_delay,
                )
                self._sleep(delay)
                total_retry_delay += delay

        raise ArtifactHTTPError(
            f"artifact transfer failed after {policy.max_attempts} attempts: {last_error}"
        ) from last_error

    def _download_once(
        self,
        *,
        spec: ArtifactSpec,
        destination: Path,
        partial: Path,
        metadata_path: Path,
        overwrite: bool,
    ) -> ArtifactReceipt:
        metadata = self._load_partial_metadata(metadata_path)
        offset = 0
        resume_validator: str | None = None

        if partial.exists():
            if (
                metadata is not None
                and metadata.source_url == spec.source_url
                and metadata.validator is not None
            ):
                offset = partial.stat().st_size
                resume_validator = metadata.validator
                if (
                    metadata.expected_total_bytes is not None
                    and offset > metadata.expected_total_bytes
                ):
                    self._discard_partial(partial, metadata_path)
                    offset = 0
                    metadata = None
                    resume_validator = None
            else:
                # A partial representation without a validator cannot be safely
                # recombined with newly fetched bytes.
                self._discard_partial(partial, metadata_path)
                metadata = None

        headers = {
            "Accept": "application/zip,application/octet-stream;q=0.9,*/*;q=0.1",
            "Accept-Encoding": "identity",
            "User-Agent": self.user_agent,
        }
        if offset > 0 and resume_validator is not None:
            headers["Range"] = f"bytes={offset}-"
            headers["If-Range"] = resume_validator

        try:
            response = self._transport.open(
                url=spec.source_url,
                headers=headers,
                timeout_seconds=self._transfer_policy.timeout_seconds,
            )
        except ArtifactHTTPError:
            raise

        with response:
            self._validate_url(response.final_url)
            status = response.status_code

            if status in self._transfer_policy.retry_status_codes:
                retry_after = _header_text(response.headers, "retry-after")
                raise _RetryableHTTPStatus(status, retry_after)

            if offset > 0 and status == 416:
                range_info = _parse_content_range(
                    _header_text(response.headers, "content-range")
                )
                if (
                    range_info is None
                    or not range_info.unsatisfied
                    or range_info.total != offset
                ):
                    raise ArtifactProtocolError(
                        "416 response does not prove the partial file is complete"
                    )
                return self._promote_verified_partial(
                    spec=spec,
                    response=response,
                    destination=destination,
                    partial=partial,
                    metadata_path=metadata_path,
                    resumed_from_bytes=offset,
                    overwrite=overwrite,
                )

            if status not in {200, 206}:
                detail = response.read(64 * 1024)
                raise ArtifactHTTPError(
                    f"artifact HTTP {status}: {detail[:512]!r}"
                )

            append = offset > 0 and status == 206
            range_info = _parse_content_range(
                _header_text(response.headers, "content-range")
            )
            if status == 206:
                if range_info is None or range_info.unsatisfied:
                    raise ArtifactProtocolError("206 response requires a valid Content-Range")
                expected_start = offset if append else 0
                if range_info.start != expected_start:
                    raise ArtifactProtocolError(
                        f"Content-Range starts at {range_info.start}, expected {expected_start}"
                    )
            elif range_info is not None:
                raise ArtifactProtocolError("Content-Range is not valid on a 200 response")

            if offset > 0 and status == 200:
                # If-Range failed or the server ignored Range. A complete 200
                # representation is safe, but it must replace—not append to—the
                # previous partial bytes.
                append = False
                offset = 0

            etag = _header_text(response.headers, "etag")
            last_modified = _header_text(response.headers, "last-modified")
            content_type = _header_text(response.headers, "content-type")
            content_length = _parse_nonnegative_int(
                _header_text(response.headers, "content-length"),
                "Content-Length",
            )

            if append and metadata is not None:
                if (
                    metadata.etag
                    and etag
                    and metadata.etag != etag
                ):
                    raise ArtifactProtocolError(
                        "ETag changed during a resumed transfer"
                    )
                if (
                    not metadata.etag
                    and metadata.last_modified
                    and last_modified
                    and metadata.last_modified != last_modified
                ):
                    raise ArtifactProtocolError(
                        "Last-Modified changed during a resumed transfer"
                    )

            expected_total = range_info.total if range_info is not None else None
            if expected_total is None and content_length is not None:
                expected_total = offset + content_length if append else content_length

            self._enforce_download_budget(expected_total)

            current_meta = _PartialMetadata(
                source_url=spec.source_url,
                etag=etag,
                last_modified=last_modified,
                expected_total_bytes=expected_total,
            )
            self._write_partial_metadata(metadata_path, current_meta)

            if append:
                digest, existing_size = _hash_file(
                    partial, self._transfer_policy.chunk_size_bytes
                )
                if existing_size != offset:
                    raise ArtifactIntegrityError(
                        "partial file size changed while preparing resume"
                    )
                mode = "ab"
                resumed_from = offset
            else:
                digest = sha256()
                existing_size = 0
                mode = "wb"
                resumed_from = 0

            transferred = 0
            try:
                with open(partial, mode) as handle:
                    while True:
                        chunk = response.read(self._transfer_policy.chunk_size_bytes)
                        if not chunk:
                            break
                        transferred += len(chunk)
                        total_size = existing_size + transferred
                        self._enforce_download_budget(total_size)
                        handle.write(chunk)
                        digest.update(chunk)
                    handle.flush()
                    os.fsync(handle.fileno())
            except (OSError, TimeoutError, HTTPException, IncompleteRead) as exc:
                raise ArtifactHTTPError(
                    f"artifact stream interrupted after {transferred} response bytes"
                ) from exc

            if content_length is not None and transferred != content_length:
                raise ArtifactHTTPError(
                    "response ended before Content-Length bytes were received "
                    f"({transferred} != {content_length})"
                )

            final_size = partial.stat().st_size
            if expected_total is not None and final_size != expected_total:
                raise ArtifactHTTPError(
                    f"artifact size does not match expected total ({final_size} != {expected_total})"
                )
            if final_size == 0:
                raise ArtifactIntegrityError("downloaded artifact is empty")

            members, total_uncompressed = self._inspect_zip(partial)

            if destination.exists() and not overwrite:
                raise FileExistsError(
                    f"artifact appeared during transfer; refusing replacement: {destination}"
                )

            os.replace(partial, destination)
            _fsync_directory(destination.parent)
            try:
                metadata_path.unlink()
            except FileNotFoundError:
                pass

            return ArtifactReceipt(
                path=destination,
                source_url=spec.source_url,
                final_url=response.final_url,
                file_name=spec.file_name,
                size_bytes=final_size,
                sha256_hex=digest.hexdigest(),
                downloaded_at=self._aware_now(),
                resumed_from_bytes=resumed_from,
                etag=etag,
                last_modified=last_modified,
                content_type=content_type,
                request_fingerprint_sha256=spec.request_fingerprint_sha256,
                archive_members=members,
                total_uncompressed_bytes=total_uncompressed,
            )

    def _promote_verified_partial(
        self,
        *,
        spec: ArtifactSpec,
        response: StreamResponse,
        destination: Path,
        partial: Path,
        metadata_path: Path,
        resumed_from_bytes: int,
        overwrite: bool,
    ) -> ArtifactReceipt:
        digest, size = _hash_file(partial, self._transfer_policy.chunk_size_bytes)
        self._enforce_download_budget(size)
        if size == 0:
            raise ArtifactIntegrityError("partial artifact is empty")
        members, total_uncompressed = self._inspect_zip(partial)

        if destination.exists() and not overwrite:
            raise FileExistsError(
                f"artifact appeared during transfer; refusing replacement: {destination}"
            )
        os.replace(partial, destination)
        _fsync_directory(destination.parent)
        try:
            metadata_path.unlink()
        except FileNotFoundError:
            pass

        return ArtifactReceipt(
            path=destination,
            source_url=spec.source_url,
            final_url=response.final_url,
            file_name=spec.file_name,
            size_bytes=size,
            sha256_hex=digest.hexdigest(),
            downloaded_at=self._aware_now(),
            resumed_from_bytes=resumed_from_bytes,
            etag=_header_text(response.headers, "etag"),
            last_modified=_header_text(response.headers, "last-modified"),
            content_type=_header_text(response.headers, "content-type"),
            request_fingerprint_sha256=spec.request_fingerprint_sha256,
            archive_members=members,
            total_uncompressed_bytes=total_uncompressed,
        )

    def _inspect_zip(
        self,
        path: Path,
    ) -> tuple[tuple[ArchiveMember, ...], int]:
        policy = self._archive_policy
        try:
            with ZipFile(path, mode="r", allowZip64=True) as archive:
                infos = archive.infolist()
                if not infos:
                    raise ArtifactIntegrityError("USAspending ZIP contains no members")
                if len(infos) > policy.max_members:
                    raise ArtifactLimitError(
                        f"ZIP member count exceeds policy ({len(infos)} > {policy.max_members})"
                    )

                manifest: list[ArchiveMember] = []
                total_uncompressed = 0
                for info in infos:
                    if policy.reject_unsafe_paths and not _safe_archive_name(info.filename):
                        raise ArtifactIntegrityError(
                            f"unsafe ZIP member path: {info.filename!r}"
                        )
                    if policy.reject_encrypted and (info.flag_bits & 0x1):
                        raise ArtifactIntegrityError(
                            f"encrypted ZIP member is not accepted: {info.filename!r}"
                        )
                    if policy.reject_symlinks and _is_zip_symlink(info.external_attr):
                        raise ArtifactIntegrityError(
                            f"symbolic-link ZIP member is not accepted: {info.filename!r}"
                        )

                    if info.file_size > policy.max_member_uncompressed_bytes:
                        raise ArtifactLimitError(
                            f"ZIP member exceeds uncompressed-size policy: {info.filename!r}"
                        )
                    total_uncompressed += info.file_size
                    if total_uncompressed > policy.max_total_uncompressed_bytes:
                        raise ArtifactLimitError(
                            "ZIP total uncompressed size exceeds policy"
                        )

                    if info.file_size:
                        if info.compress_size == 0:
                            raise ArtifactLimitError(
                                f"ZIP member has infinite compression ratio: {info.filename!r}"
                            )
                        ratio = info.file_size / info.compress_size
                        if ratio > policy.max_compression_ratio:
                            raise ArtifactLimitError(
                                f"ZIP member compression ratio exceeds policy: {info.filename!r}"
                            )

                    manifest.append(
                        ArchiveMember(
                            name=info.filename,
                            compressed_bytes=info.compress_size,
                            uncompressed_bytes=info.file_size,
                            crc32=info.CRC,
                            compression_method=info.compress_type,
                            is_directory=info.is_dir(),
                        )
                    )

                if policy.verify_crc:
                    bad_member = archive.testzip()
                    if bad_member is not None:
                        raise ArtifactIntegrityError(
                            f"ZIP CRC verification failed for member: {bad_member!r}"
                        )

                return tuple(manifest), total_uncompressed
        except (BadZipFile, LargeZipFile, RuntimeError, NotImplementedError) as exc:
            raise ArtifactIntegrityError(f"invalid or unsupported ZIP artifact: {exc}") from exc

    def _enforce_download_budget(self, size: int | None) -> None:
        limit = self._transfer_policy.max_download_bytes
        if size is not None and limit is not None and size > limit:
            raise ArtifactLimitError(
                f"artifact exceeds configured download budget ({size} > {limit})"
            )

    def _validate_url(self, url: str) -> None:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ArtifactProtocolError("artifact URL must be absolute HTTP(S)")
        if self._transfer_policy.require_https and parsed.scheme != "https":
            raise ArtifactProtocolError("artifact URL must use HTTPS")

    def _acquire_lock(self, lock_path: Path, spec: ArtifactSpec) -> None:
        payload = {
            "pid": os.getpid(),
            "source_url": spec.source_url,
            "file_name": spec.file_name,
            "started_at": self._aware_now().isoformat(),
        }
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        try:
            fd = os.open(lock_path, flags, 0o600)
        except FileExistsError as exc:
            raise ArtifactLockedError(
                f"artifact destination is already locked: {lock_path}"
            ) from exc

        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        with os.fdopen(fd, "wb", closefd=True) as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        _fsync_directory(lock_path.parent)

    @staticmethod
    def _discard_partial(partial: Path, metadata_path: Path) -> None:
        for path in (partial, metadata_path):
            try:
                path.unlink()
            except FileNotFoundError:
                pass

    @staticmethod
    def _load_partial_metadata(path: Path) -> _PartialMetadata | None:
        try:
            raw = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise ArtifactError(f"cannot read partial metadata: {path}") from exc

        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return None
        if not isinstance(payload, Mapping) or payload.get("version") != 1:
            return None

        source_url = payload.get("source_url")
        etag = payload.get("etag")
        last_modified = payload.get("last_modified")
        expected = payload.get("expected_total_bytes")
        if not isinstance(source_url, str):
            return None
        if etag is not None and not isinstance(etag, str):
            return None
        if last_modified is not None and not isinstance(last_modified, str):
            return None
        if (
            expected is not None
            and (isinstance(expected, bool) or not isinstance(expected, int) or expected < 0)
        ):
            return None
        return _PartialMetadata(
            source_url=source_url,
            etag=etag,
            last_modified=last_modified,
            expected_total_bytes=expected,
        )

    @staticmethod
    def _write_partial_metadata(path: Path, metadata: _PartialMetadata) -> None:
        _atomic_write_json(
            path,
            {
                "version": 1,
                "source_url": metadata.source_url,
                "etag": metadata.etag,
                "last_modified": metadata.last_modified,
                "expected_total_bytes": metadata.expected_total_bytes,
            },
        )

    def _retry_delay(
        self,
        *,
        attempt: int,
        retry_after: str | None,
        total_retry_delay: float,
    ) -> float:
        policy = self._transfer_policy
        parsed_retry_after = self._parse_retry_after(retry_after)
        if parsed_retry_after is not None:
            delay = parsed_retry_after
        else:
            exponential = min(
                policy.max_retry_seconds,
                policy.base_retry_seconds * (2 ** max(0, attempt - 1)),
            )
            jitter = exponential * policy.retry_jitter_ratio * (
                (self._random() * 2.0) - 1.0
            )
            delay = max(0.0, exponential + jitter)

        if total_retry_delay + delay > policy.max_total_retry_delay_seconds:
            raise ArtifactHTTPError("artifact retry-delay budget exhausted")
        return delay

    def _parse_retry_after(self, value: str | None) -> float | None:
        if value is None:
            return None
        text = value.strip()
        try:
            return max(0.0, float(text))
        except ValueError:
            try:
                parsed = parsedate_to_datetime(text)
            except (TypeError, ValueError):
                return None
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return max(
                0.0,
                (parsed.astimezone(timezone.utc) - self._aware_now()).total_seconds(),
            )

    def _aware_now(self) -> datetime:
        value = self._clock()
        if not isinstance(value, datetime):
            raise ArtifactError("clock must return datetime")
        if value.tzinfo is None or value.utcoffset() is None:
            raise ArtifactError("clock must return a timezone-aware datetime")
        return value.astimezone(timezone.utc)


class _RetryableHTTPStatus(ArtifactHTTPError):
    def __init__(self, status_code: int, retry_after: str | None) -> None:
        super().__init__(f"retryable artifact HTTP status: {status_code}")
        self.status_code = status_code
        self.retry_after = retry_after
