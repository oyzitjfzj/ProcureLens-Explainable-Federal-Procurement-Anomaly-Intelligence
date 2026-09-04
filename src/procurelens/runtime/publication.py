"""Atomic filesystem publication for completed ProcureLens analysis runs.

Publishes validated analysis artifacts as one directory bundle containing the
exact run manifest, deterministic review exports, publication integrity metadata,
and—when the run originates from the live USAspending bridge—the bounded source
acquisition/loading/quality evidence that produced the analysis.

The bundle is staged on the destination filesystem and renamed into place only
after every payload hash is verified and file/directory buffers are flushed.
Existing bundles are never overwritten. A sibling lock prevents concurrent
ProcureLens publishers from racing on the same bundle name.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Iterable


class PublicationError(RuntimeError):
    """Raised when a completed run cannot be published safely."""


PUBLICATION_SCHEMA_NAME = "procurelens_publication_bundle"
PUBLICATION_SCHEMA_VERSION = 2
_BUNDLE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class PublicationRunKind(str, Enum):
    ANALYSIS = "procurelens_analysis"
    LIVE_USASPENDING = "live_usaspending_analysis"


@dataclass(frozen=True, slots=True)
class PublishedFile:
    relative_path: str
    media_type: str
    sha256_hex: str
    byte_count: int

    def __post_init__(self) -> None:
        path = self.relative_path.strip()
        media = self.media_type.strip()
        parsed = Path(path)
        if (
            not path
            or parsed.is_absolute()
            or ".." in parsed.parts
            or path.endswith("/")
        ):
            raise PublicationError("published file path must be safe and relative")
        if not media:
            raise PublicationError("published file media_type must not be blank")
        object.__setattr__(self, "relative_path", path)
        object.__setattr__(self, "media_type", media)
        object.__setattr__(self, "sha256_hex", _digest_hex(self.sha256_hex, "sha256_hex"))
        if (
            isinstance(self.byte_count, bool)
            or not isinstance(self.byte_count, int)
            or self.byte_count < 1
        ):
            raise PublicationError("published file byte_count must be positive")

    def as_dict(self) -> dict[str, Any]:
        return {
            "relative_path": self.relative_path,
            "media_type": self.media_type,
            "sha256": self.sha256_hex,
            "byte_count": self.byte_count,
        }


@dataclass(frozen=True, slots=True)
class PublicationReceipt:
    bundle_name: str
    bundle_path: Path
    run_kind: PublicationRunKind
    run_evidence_sha256: str
    analysis_evidence_sha256: str
    manifest_evidence_sha256: str
    manifest_payload_sha256: str
    publication_metadata_sha256: str
    files: tuple[PublishedFile, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "bundle_name", _bundle_name(self.bundle_name))
        path = Path(self.bundle_path)
        if not path.is_dir():
            raise PublicationError(
                "publication receipt bundle_path must exist as a directory"
            )
        object.__setattr__(self, "bundle_path", path)
        object.__setattr__(self, "run_kind", PublicationRunKind(self.run_kind))
        for field_name in (
            "run_evidence_sha256",
            "analysis_evidence_sha256",
            "manifest_evidence_sha256",
            "manifest_payload_sha256",
            "publication_metadata_sha256",
        ):
            object.__setattr__(
                self,
                field_name,
                _digest_hex(getattr(self, field_name), field_name),
            )
        files = tuple(self.files)
        if not files:
            raise PublicationError("publication receipt requires published files")
        names = tuple(item.relative_path for item in files)
        if len(names) != len(set(names)):
            raise PublicationError("publication receipt contains duplicate file paths")
        if "manifest.json" not in names or "publication.json" not in names:
            raise PublicationError(
                "publication receipt requires manifest.json and publication.json"
            )
        object.__setattr__(self, "files", files)

    @property
    def evidence_sha256(self) -> str:
        return _digest(
            {
                "bundle_name": self.bundle_name,
                "run_kind": self.run_kind.value,
                "run_evidence_sha256": self.run_evidence_sha256,
                "analysis_evidence_sha256": self.analysis_evidence_sha256,
                "manifest_evidence_sha256": self.manifest_evidence_sha256,
                "manifest_payload_sha256": self.manifest_payload_sha256,
                "publication_metadata_sha256": self.publication_metadata_sha256,
                "files": [item.as_dict() for item in self.files],
            }
        )


@dataclass(frozen=True, slots=True)
class _ExtraPayload:
    relative_path: str
    media_type: str
    payload: bytes

    def __post_init__(self) -> None:
        published = PublishedFile(
            self.relative_path,
            self.media_type,
            _bytes_digest(self.payload),
            len(self.payload),
        )
        if published.relative_path in {"manifest.json", "publication.json"}:
            raise PublicationError("extra payload uses reserved publication path")
        if published.relative_path.startswith("exports/"):
            raise PublicationError("extra payload cannot occupy exports/ namespace")
        if not self.payload:
            raise PublicationError("extra publication payload must not be empty")

    @property
    def published_file(self) -> PublishedFile:
        return PublishedFile(
            self.relative_path,
            self.media_type,
            _bytes_digest(self.payload),
            len(self.payload),
        )


def publish_analysis_run(
    run: Any,
    output_root: str | os.PathLike[str],
    *,
    bundle_name: str,
) -> PublicationReceipt:
    """Atomically publish one validated ProcureLensAnalysisRun directory bundle."""

    from procurelens.pipeline.run import ProcureLensAnalysisRun

    if not isinstance(run, ProcureLensAnalysisRun):
        raise TypeError("run must be ProcureLensAnalysisRun")
    return _publish_bundle(
        analysis=run,
        run_kind=PublicationRunKind.ANALYSIS,
        top_level_evidence_sha256=run.evidence_sha256,
        output_root=output_root,
        bundle_name=bundle_name,
        extra_payloads=(),
    )


def publish_live_usaspending_run(
    run: Any,
    output_root: str | os.PathLike[str],
    *,
    bundle_name: str,
) -> PublicationReceipt:
    """Publish live USAspending source evidence and its analysis as one atomic bundle."""

    from procurelens.pipeline.usaspending_live import LiveUSAspendingAnalysisRun

    if not isinstance(run, LiveUSAspendingAnalysisRun):
        raise TypeError("run must be LiveUSAspendingAnalysisRun")

    plan_payload = _json_bytes(run.prepared.plan.as_dict(include_sha=True))
    prepared_payload = _json_bytes(run.prepared.as_dict(include_sha=True))
    live_payload = _json_bytes(run.as_dict())
    extras = (
        _ExtraPayload("source/live_plan.json", "application/json", plan_payload),
        _ExtraPayload(
            "source/prepared_dataset.json",
            "application/json",
            prepared_payload,
        ),
        _ExtraPayload("live_run.json", "application/json", live_payload),
    )
    return _publish_bundle(
        analysis=run.analysis,
        run_kind=PublicationRunKind.LIVE_USASPENDING,
        top_level_evidence_sha256=run.evidence_sha256,
        output_root=output_root,
        bundle_name=bundle_name,
        extra_payloads=extras,
    )


def _publish_bundle(
    *,
    analysis: Any,
    run_kind: PublicationRunKind,
    top_level_evidence_sha256: str,
    output_root: str | os.PathLike[str],
    bundle_name: str,
    extra_payloads: Iterable[_ExtraPayload],
) -> PublicationReceipt:
    # Keep the implementation independent of top-level source acquisition while
    # validating the analysis object at the boundary shared by both public entrypoints.
    from procurelens.pipeline.run import ProcureLensAnalysisRun

    if not isinstance(analysis, ProcureLensAnalysisRun):
        raise TypeError("analysis must be ProcureLensAnalysisRun")
    run_kind = PublicationRunKind(run_kind)
    top_level_sha = _digest_hex(
        top_level_evidence_sha256, "top_level_evidence_sha256"
    )
    extras = tuple(extra_payloads)
    if any(not isinstance(item, _ExtraPayload) for item in extras):
        raise TypeError("extra_payloads must contain internal publication payloads")
    extra_names = tuple(item.relative_path for item in extras)
    if len(extra_names) != len(set(extra_names)):
        raise PublicationError("extra publication payload paths contain duplicates")

    name = _bundle_name(bundle_name)
    root = Path(output_root)
    if root.exists() and not root.is_dir():
        raise PublicationError("output_root exists but is not a directory")
    root.mkdir(parents=True, exist_ok=True)

    destination = root / name
    lock_path = root / f".{name}.publish.lock"
    stage: Path | None = None
    lock_fd: int | None = None
    try:
        try:
            lock_fd = os.open(
                lock_path,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                0o600,
            )
        except FileExistsError as exc:
            raise PublicationError(
                "publication lock already exists for this bundle name"
            ) from exc
        os.write(lock_fd, (top_level_sha + "\n").encode("ascii"))
        os.fsync(lock_fd)

        if destination.exists() or destination.is_symlink():
            raise PublicationError("destination bundle already exists")

        stage = Path(tempfile.mkdtemp(prefix=f".{name}.staging-", dir=root))

        manifest_payload = _json_bytes(analysis.manifest.as_dict(include_sha=True))
        manifest_sha = _bytes_digest(manifest_payload)
        manifest_file = PublishedFile(
            relative_path="manifest.json",
            media_type="application/json",
            sha256_hex=manifest_sha,
            byte_count=len(manifest_payload),
        )
        _write_fsynced(stage / manifest_file.relative_path, manifest_payload)

        published: list[PublishedFile] = [manifest_file]
        for index, export in enumerate(analysis.model_review.serialized_exports):
            payload = export.payload_bytes
            observed_sha = _bytes_digest(payload)
            if observed_sha != export.payload_sha256:
                raise PublicationError(
                    f"serialized export payload hash mismatch at index {index}"
                )
            extension = export.format.value
            item = PublishedFile(
                relative_path=f"exports/export_{index:02d}.{extension}",
                media_type=export.media_type,
                sha256_hex=observed_sha,
                byte_count=len(payload),
            )
            _write_fsynced(stage / item.relative_path, payload)
            published.append(item)

        for extra in extras:
            item = extra.published_file
            _write_fsynced(stage / item.relative_path, extra.payload)
            published.append(item)

        file_names = tuple(item.relative_path for item in published)
        if len(file_names) != len(set(file_names)):
            raise PublicationError("publication payload paths contain duplicates")

        publication_payload = _json_bytes(
            {
                "schema": {
                    "name": PUBLICATION_SCHEMA_NAME,
                    "version": PUBLICATION_SCHEMA_VERSION,
                },
                "bundle_name": name,
                "run_kind": run_kind.value,
                "run_evidence_sha256": top_level_sha,
                "analysis_evidence_sha256": analysis.evidence_sha256,
                "manifest_evidence_sha256": analysis.manifest.evidence_sha256,
                "manifest_payload_sha256": manifest_sha,
                "files": [item.as_dict() for item in published],
            }
        )
        publication_sha = _bytes_digest(publication_payload)
        publication_file = PublishedFile(
            relative_path="publication.json",
            media_type="application/json",
            sha256_hex=publication_sha,
            byte_count=len(publication_payload),
        )
        _write_fsynced(stage / publication_file.relative_path, publication_payload)
        published.append(publication_file)

        _fsync_tree_directories(stage)
        _fsync_directory(root)

        # The sibling lock serializes ProcureLens publishers using this API. The
        # second existence check also fails closed if another actor creates the path.
        if destination.exists() or destination.is_symlink():
            raise PublicationError("destination bundle appeared during publication")
        os.rename(stage, destination)
        stage = None
        _fsync_directory(root)

        return PublicationReceipt(
            bundle_name=name,
            bundle_path=destination,
            run_kind=run_kind,
            run_evidence_sha256=top_level_sha,
            analysis_evidence_sha256=analysis.evidence_sha256,
            manifest_evidence_sha256=analysis.manifest.evidence_sha256,
            manifest_payload_sha256=manifest_sha,
            publication_metadata_sha256=publication_sha,
            files=tuple(published),
        )
    finally:
        if stage is not None:
            shutil.rmtree(stage, ignore_errors=True)
        if lock_fd is not None:
            os.close(lock_fd)
            try:
                lock_path.unlink()
            except FileNotFoundError:
                pass
        try:
            _fsync_directory(root)
        except OSError:
            # Do not mask the primary publication error during cleanup.
            pass


def _bundle_name(value: str) -> str:
    if not isinstance(value, str):
        raise PublicationError("bundle_name must be a string")
    name = value.strip()
    if not _BUNDLE_NAME.fullmatch(name) or name in {".", ".."}:
        raise PublicationError(
            "bundle_name must be one safe filesystem component using letters, digits, '.', '_' or '-'"
        )
    return name


def _write_fsynced(path: Path, payload: bytes) -> None:
    if not payload:
        raise PublicationError("published payload must not be empty")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise PublicationError(f"staging file already exists: {path.name}") from exc


def _fsync_tree_directories(root: Path) -> None:
    directories = [item for item in root.rglob("*") if item.is_dir()]
    directories.sort(key=lambda item: len(item.parts), reverse=True)
    for directory in directories:
        _fsync_directory(directory)
    _fsync_directory(root)


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _bytes_digest(payload: bytes) -> str:
    return sha256(payload).hexdigest()


def _digest_hex(value: str, name: str) -> str:
    if not isinstance(value, str):
        raise PublicationError(f"{name} must be a string")
    digest = value.strip().lower()
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise PublicationError(f"{name} must be a SHA-256 hex digest")
    return digest


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