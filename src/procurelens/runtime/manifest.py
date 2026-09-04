"""Deterministic run-provenance manifests for ProcureLens.

Represents an analysis run as immutable input/output artifacts connected by an
ordered sequence of stage executions. The structure is intentionally close to the
entity/activity/usage/generation ideas of W3C PROV while remaining compact and
native to ProcureLens.

Two fingerprints are exposed:
- recipe_sha256 identifies inputs, stage implementations/configuration, requested
  outputs, and software environment while excluding generated artifact hashes.
- evidence_sha256 includes generated hashes as well, so nondeterministic output
  under an otherwise identical recipe becomes observable.

No wall-clock timestamp is included in these deterministic fingerprints.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import platform
from types import MappingProxyType
from typing import Any, Iterable, Mapping


class RunManifestError(ValueError):
    """Raised when run provenance is incomplete or internally inconsistent."""


MANIFEST_SCHEMA_NAME = "procurelens_run_provenance"
MANIFEST_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class SoftwareComponent:
    name: str
    version: str

    def __post_init__(self) -> None:
        for field_name in ("name", "version"):
            text = getattr(self, field_name).strip()
            if not text:
                raise RunManifestError(f"{field_name} must not be blank")
            object.__setattr__(self, field_name, text)

    def as_dict(self) -> dict[str, str]:
        return {"name": self.name, "version": self.version}


@dataclass(frozen=True, slots=True)
class RuntimeEnvironment:
    python_implementation: str
    python_version: str
    platform_system: str
    platform_machine: str
    components: tuple[SoftwareComponent, ...]

    def __post_init__(self) -> None:
        for field_name in (
            "python_implementation",
            "python_version",
            "platform_system",
            "platform_machine",
        ):
            text = getattr(self, field_name).strip()
            if not text:
                raise RunManifestError(f"{field_name} must not be blank")
            object.__setattr__(self, field_name, text)

        components = tuple(self.components)
        names = tuple(item.name for item in components)
        if len(names) != len(set(names)):
            raise RunManifestError("software component names must be unique")
        canonical = tuple(sorted(components, key=lambda item: item.name))
        if components != canonical:
            raise RunManifestError(
                "software components must be sorted by component name"
            )
        object.__setattr__(self, "components", components)

    @property
    def sha256_hex(self) -> str:
        return _digest(self.as_dict())

    def as_dict(self) -> dict[str, Any]:
        return {
            "python_implementation": self.python_implementation,
            "python_version": self.python_version,
            "platform_system": self.platform_system,
            "platform_machine": self.platform_machine,
            "components": [item.as_dict() for item in self.components],
        }


@dataclass(frozen=True, slots=True)
class PipelineArtifact:
    """One immutable pipeline entity identified by a strong content/evidence hash."""

    artifact_id: str
    artifact_kind: str
    sha256_hex: str
    media_type: str | None = None
    attributes: Mapping[str, str] | None = None

    def __post_init__(self) -> None:
        for field_name in ("artifact_id", "artifact_kind"):
            text = getattr(self, field_name).strip()
            if not text:
                raise RunManifestError(f"{field_name} must not be blank")
            object.__setattr__(self, field_name, text)
        object.__setattr__(
            self, "sha256_hex", _digest_hex(self.sha256_hex, "sha256_hex")
        )
        if self.media_type is not None:
            media = self.media_type.strip()
            object.__setattr__(self, "media_type", media or None)

        raw = {} if self.attributes is None else dict(self.attributes)
        attributes: dict[str, str] = {}
        for key, value in raw.items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise RunManifestError(
                    "artifact attributes must map strings to strings"
                )
            normalized_key = key.strip()
            normalized_value = value.strip()
            if not normalized_key or not normalized_value:
                raise RunManifestError(
                    "artifact attribute keys/values must not be blank"
                )
            if normalized_key in attributes:
                raise RunManifestError(
                    f"duplicate normalized artifact attribute: {normalized_key}"
                )
            attributes[normalized_key] = normalized_value
        object.__setattr__(
            self,
            "attributes",
            MappingProxyType(dict(sorted(attributes.items()))),
        )

    def recipe_dict(self) -> dict[str, Any]:
        """Artifact identity/shape without its generated content hash."""
        return {
            "artifact_id": self.artifact_id,
            "artifact_kind": self.artifact_kind,
            "media_type": self.media_type,
            "attributes": dict(self.attributes),
        }

    def as_dict(self) -> dict[str, Any]:
        result = self.recipe_dict()
        result["sha256"] = self.sha256_hex
        return result


@dataclass(frozen=True, slots=True)
class StageExecution:
    """One ordered pipeline activity using existing entities and generating new ones."""

    stage_id: str
    implementation: str
    input_artifact_ids: tuple[str, ...]
    output_artifacts: tuple[PipelineArtifact, ...]
    config_sha256: str | None = None

    def __post_init__(self) -> None:
        for field_name in ("stage_id", "implementation"):
            text = getattr(self, field_name).strip()
            if not text:
                raise RunManifestError(f"{field_name} must not be blank")
            object.__setattr__(self, field_name, text)

        inputs = tuple(_artifact_id(value) for value in self.input_artifact_ids)
        if len(inputs) != len(set(inputs)):
            raise RunManifestError(
                f"{self.stage_id}: input artifact IDs must be unique"
            )
        if not inputs:
            raise RunManifestError(
                f"{self.stage_id}: stage requires at least one input artifact"
            )
        object.__setattr__(self, "input_artifact_ids", inputs)

        outputs = tuple(self.output_artifacts)
        if not outputs:
            raise RunManifestError(
                f"{self.stage_id}: stage requires at least one output artifact"
            )
        if any(not isinstance(item, PipelineArtifact) for item in outputs):
            raise TypeError("output_artifacts must be PipelineArtifact")
        output_ids = tuple(item.artifact_id for item in outputs)
        if len(output_ids) != len(set(output_ids)):
            raise RunManifestError(
                f"{self.stage_id}: output artifact IDs must be unique"
            )
        object.__setattr__(self, "output_artifacts", outputs)

        if self.config_sha256 is not None:
            object.__setattr__(
                self,
                "config_sha256",
                _digest_hex(self.config_sha256, "config_sha256"),
            )

    def recipe_dict(self) -> dict[str, Any]:
        return {
            "stage_id": self.stage_id,
            "implementation": self.implementation,
            "config_sha256": self.config_sha256,
            "input_artifact_ids": list(self.input_artifact_ids),
            "output_artifacts": [
                item.recipe_dict() for item in self.output_artifacts
            ],
        }

    def as_dict(self) -> dict[str, Any]:
        result = self.recipe_dict()
        result["output_artifacts"] = [
            item.as_dict() for item in self.output_artifacts
        ]
        return result


@dataclass(frozen=True, slots=True)
class RunManifest:
    """Validated deterministic provenance graph for one ProcureLens run."""

    run_name: str
    source_revision: str | None
    environment: RuntimeEnvironment
    initial_artifacts: tuple[PipelineArtifact, ...]
    stages: tuple[StageExecution, ...]
    final_artifact_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        run_name = self.run_name.strip()
        if not run_name:
            raise RunManifestError("run_name must not be blank")
        object.__setattr__(self, "run_name", run_name)

        if self.source_revision is not None:
            revision = self.source_revision.strip()
            object.__setattr__(self, "source_revision", revision or None)

        if not isinstance(self.environment, RuntimeEnvironment):
            raise TypeError("environment must be RuntimeEnvironment")

        initial = tuple(self.initial_artifacts)
        if not initial:
            raise RunManifestError(
                "run manifest requires at least one initial artifact"
            )
        if any(not isinstance(item, PipelineArtifact) for item in initial):
            raise TypeError("initial_artifacts must be PipelineArtifact")

        initial_ids = tuple(item.artifact_id for item in initial)
        if len(initial_ids) != len(set(initial_ids)):
            raise RunManifestError("initial artifact IDs must be unique")
        object.__setattr__(self, "initial_artifacts", initial)

        stages = tuple(self.stages)
        if not stages:
            raise RunManifestError("run manifest requires at least one stage")
        if any(not isinstance(stage, StageExecution) for stage in stages):
            raise TypeError("stages must be StageExecution")
        stage_ids = tuple(stage.stage_id for stage in stages)
        if len(stage_ids) != len(set(stage_ids)):
            raise RunManifestError("stage IDs must be globally unique")

        available = set(initial_ids)
        generated: set[str] = set()
        for stage in stages:
            missing = [
                artifact_id
                for artifact_id in stage.input_artifact_ids
                if artifact_id not in available
            ]
            if missing:
                raise RunManifestError(
                    f"{stage.stage_id}: inputs are unavailable at this stage: "
                    + ", ".join(missing)
                )
            for artifact in stage.output_artifacts:
                if artifact.artifact_id in available or artifact.artifact_id in generated:
                    raise RunManifestError(
                        "artifact ID generated more than once or collides with "
                        f"initial input: {artifact.artifact_id}"
                    )
                generated.add(artifact.artifact_id)
                available.add(artifact.artifact_id)

        finals = tuple(_artifact_id(value) for value in self.final_artifact_ids)
        if not finals:
            raise RunManifestError(
                "run manifest requires at least one final artifact"
            )
        if len(finals) != len(set(finals)):
            raise RunManifestError("final artifact IDs must be unique")
        unknown = [value for value in finals if value not in available]
        if unknown:
            raise RunManifestError(
                "final artifacts were never provided/generated: "
                + ", ".join(unknown)
            )

        object.__setattr__(self, "stages", stages)
        object.__setattr__(self, "final_artifact_ids", finals)

    @property
    def artifact_count(self) -> int:
        return len(self.initial_artifacts) + sum(
            len(stage.output_artifacts) for stage in self.stages
        )

    @property
    def stage_count(self) -> int:
        return len(self.stages)

    @property
    def artifact_index(self) -> Mapping[str, PipelineArtifact]:
        result = {
            item.artifact_id: item for item in self.initial_artifacts
        }
        for stage in self.stages:
            result.update(
                {item.artifact_id: item for item in stage.output_artifacts}
            )
        return MappingProxyType(result)

    @property
    def recipe_sha256(self) -> str:
        """Hash the reproducible recipe while excluding generated output hashes."""
        return _digest(
            {
                "schema": {
                    "name": MANIFEST_SCHEMA_NAME,
                    "version": MANIFEST_SCHEMA_VERSION,
                },
                "run_name": self.run_name,
                "source_revision": self.source_revision,
                "environment": self.environment.as_dict(),
                # Initial input hashes are part of the recipe/data snapshot.
                "initial_artifacts": [
                    item.as_dict() for item in self.initial_artifacts
                ],
                "stages": [stage.recipe_dict() for stage in self.stages],
                "final_artifact_ids": list(self.final_artifact_ids),
            }
        )

    @property
    def evidence_sha256(self) -> str:
        """Hash the complete observed run, including every generated artifact hash."""
        return _digest(self.as_dict(include_sha=False))

    def as_dict(self, *, include_sha: bool = True) -> dict[str, Any]:
        result = {
            "schema": {
                "name": MANIFEST_SCHEMA_NAME,
                "version": MANIFEST_SCHEMA_VERSION,
            },
            "run_name": self.run_name,
            "source_revision": self.source_revision,
            "environment": self.environment.as_dict(),
            "environment_sha256": self.environment.sha256_hex,
            "initial_artifacts": [
                item.as_dict() for item in self.initial_artifacts
            ],
            "stages": [stage.as_dict() for stage in self.stages],
            "final_artifact_ids": list(self.final_artifact_ids),
            "artifact_count": self.artifact_count,
            "stage_count": self.stage_count,
            "recipe_sha256": self.recipe_sha256,
        }
        if include_sha:
            result["evidence_sha256"] = self.evidence_sha256
        return result


def capture_runtime_environment(
    components: Mapping[str, str] | Iterable[SoftwareComponent] = (),
) -> RuntimeEnvironment:
    """Capture deterministic runtime identifiers without wall-clock metadata."""

    if isinstance(components, Mapping):
        items = tuple(
            SoftwareComponent(str(name), str(version))
            for name, version in components.items()
        )
    else:
        items = tuple(components)
        if any(not isinstance(item, SoftwareComponent) for item in items):
            raise TypeError(
                "components must be a mapping or iterable of SoftwareComponent"
            )

    canonical = tuple(sorted(items, key=lambda item: item.name))
    return RuntimeEnvironment(
        python_implementation=platform.python_implementation(),
        python_version=platform.python_version(),
        platform_system=platform.system() or "unknown",
        platform_machine=platform.machine() or "unknown",
        components=canonical,
    )


def _artifact_id(value: str) -> str:
    if not isinstance(value, str):
        raise RunManifestError("artifact IDs must be strings")
    text = value.strip()
    if not text:
        raise RunManifestError("artifact ID must not be blank")
    return text


def _digest_hex(value: str, name: str) -> str:
    if not isinstance(value, str):
        raise RunManifestError(f"{name} must be a string")
    digest = value.strip().lower()
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise RunManifestError(f"{name} must be a SHA-256 hex digest")
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
