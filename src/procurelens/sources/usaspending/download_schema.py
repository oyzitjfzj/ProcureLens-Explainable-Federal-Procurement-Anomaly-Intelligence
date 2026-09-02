"""Versioned USAspending download-schema contracts for ProcureLens.

This module bridges structural rows from ``reader.py`` to semantic normalization
in ``adapter.py``. It deliberately does not parse CSV, fetch data, score risk,
or infer field meaning from fuzzy similarity.

USAspending's public contract-transaction downloads use a presentation schema
that is not identical to the transaction-search/backend schema. Some literal
names even change meaning across those surfaces. ProcureLens therefore uses
explicit, source-profile-specific semantic contracts instead of one global
alias table.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
import json
from types import MappingProxyType

from procurelens.sources.usaspending.reader import MemberSchema, normalize_header


class DownloadSchemaError(RuntimeError):
    """Base error for USAspending download-schema contract failures."""


class DownloadSchemaDefinitionError(DownloadSchemaError):
    """Raised when a schema profile is internally ambiguous or invalid."""


class DownloadSchemaResolutionError(DownloadSchemaError):
    """Raised when observed headers cannot be resolved without guessing."""


@dataclass(frozen=True, slots=True)
class DownloadFieldContract:
    """One canonical ProcureLens field as represented by one source profile."""

    canonical_field: str
    source_headers: tuple[str, ...]
    required: bool = False

    def __post_init__(self) -> None:
        canonical = self.canonical_field.strip()
        if not canonical:
            raise DownloadSchemaDefinitionError("canonical_field must not be blank")
        if not self.source_headers:
            raise DownloadSchemaDefinitionError(
                f"{canonical}: at least one source header is required"
            )

        cleaned: list[str] = []
        seen: set[str] = set()
        for header in self.source_headers:
            if not isinstance(header, str):
                raise DownloadSchemaDefinitionError(
                    f"{canonical}: source headers must be strings"
                )
            value = header.strip()
            normalized = normalize_header(value)
            if not value or not normalized:
                raise DownloadSchemaDefinitionError(
                    f"{canonical}: source header must not be blank"
                )
            if normalized not in seen:
                cleaned.append(value)
                seen.add(normalized)

        object.__setattr__(self, "canonical_field", canonical)
        object.__setattr__(self, "source_headers", tuple(cleaned))


@dataclass(frozen=True, slots=True)
class HeaderBinding:
    """Observed source-header binding for one canonical field."""

    canonical_field: str
    matched_headers: tuple[str, ...]
    required: bool


@dataclass(frozen=True, slots=True)
class SchemaCompatibility:
    """Compatibility evidence for one observed tabular header set."""

    profile_name: str
    profile_sha256: str
    observed_schema_sha256: str
    compatible: bool
    bindings: tuple[HeaderBinding, ...]
    missing_required_fields: tuple[str, ...]
    unrecognized_headers: tuple[str, ...]
    redundant_canonical_fields: tuple[str, ...]

    @property
    def recognized_canonical_fields(self) -> tuple[str, ...]:
        return tuple(binding.canonical_field for binding in self.bindings)

    @property
    def additive_drift_detected(self) -> bool:
        return bool(self.unrecognized_headers)


@dataclass(frozen=True, slots=True)
class DownloadSchemaProfile:
    """Explicit semantic contract for one USAspending download surface."""

    name: str
    source_family: str
    fields: tuple[DownloadFieldContract, ...]

    def __post_init__(self) -> None:
        name = self.name.strip()
        family = self.source_family.strip()
        if not name:
            raise DownloadSchemaDefinitionError("profile name must not be blank")
        if not family:
            raise DownloadSchemaDefinitionError("source_family must not be blank")
        if not self.fields:
            raise DownloadSchemaDefinitionError("profile must define at least one field")

        canonical_seen: set[str] = set()
        owner_by_header: dict[str, str] = {}
        for field in self.fields:
            canonical = field.canonical_field
            if canonical in canonical_seen:
                raise DownloadSchemaDefinitionError(
                    f"duplicate canonical field in profile: {canonical}"
                )
            canonical_seen.add(canonical)

            for header in field.source_headers:
                normalized = normalize_header(header)
                previous = owner_by_header.get(normalized)
                if previous is not None and previous != canonical:
                    raise DownloadSchemaDefinitionError(
                        "one normalized source header cannot own two meanings "
                        f"inside one profile: {header!r} -> {previous}, {canonical}"
                    )
                owner_by_header[normalized] = canonical

        if not any(field.required for field in self.fields):
            raise DownloadSchemaDefinitionError(
                "profile must define at least one required canonical field"
            )

        object.__setattr__(self, "name", name)
        object.__setattr__(self, "source_family", family)

    @property
    def sha256_hex(self) -> str:
        payload = {
            "name": self.name,
            "source_family": self.source_family,
            "fields": [
                {
                    "canonical_field": field.canonical_field,
                    "source_headers": list(field.source_headers),
                    "required": field.required,
                }
                for field in self.fields
            ],
        }
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        return sha256(encoded).hexdigest()

    @property
    def required_fields(self) -> tuple[str, ...]:
        return tuple(field.canonical_field for field in self.fields if field.required)

    def adapter_aliases(self) -> Mapping[str, tuple[str, ...]]:
        """Return an immutable alias map for USAspendingTransactionAdapter."""
        return MappingProxyType(
            {field.canonical_field: field.source_headers for field in self.fields}
        )

    def inspect_headers(
        self,
        headers: Sequence[str],
        *,
        observed_schema_sha256: str | None = None,
    ) -> SchemaCompatibility:
        """Compare observed headers with this profile without fuzzy inference."""

        if isinstance(headers, (str, bytes)):
            raise TypeError("headers must be a sequence of header strings")

        cleaned_headers: list[str] = []
        observed_by_normalized: dict[str, list[str]] = {}
        for raw in headers:
            if not isinstance(raw, str):
                raise DownloadSchemaResolutionError(
                    "observed headers must all be strings"
                )
            value = raw.strip()
            normalized = normalize_header(value)
            if not value or not normalized:
                raise DownloadSchemaResolutionError(
                    "observed headers must not contain blank names"
                )
            cleaned_headers.append(value)
            observed_by_normalized.setdefault(normalized, []).append(value)

        if len(cleaned_headers) != len(
            {normalize_header(value) for value in cleaned_headers}
        ):
            raise DownloadSchemaResolutionError(
                "observed headers contain duplicate/ambiguous normalized names"
            )

        bindings: list[HeaderBinding] = []
        missing_required: list[str] = []
        redundant: list[str] = []
        recognized_normalized: set[str] = set()

        for field in self.fields:
            matched: list[str] = []
            for alias in field.source_headers:
                normalized = normalize_header(alias)
                observed = observed_by_normalized.get(normalized)
                if observed:
                    matched.extend(observed)
                    recognized_normalized.add(normalized)

            if matched:
                if len(matched) > 1:
                    redundant.append(field.canonical_field)
                bindings.append(
                    HeaderBinding(
                        canonical_field=field.canonical_field,
                        matched_headers=tuple(matched),
                        required=field.required,
                    )
                )
            elif field.required:
                missing_required.append(field.canonical_field)

        unrecognized = tuple(
            value
            for value in cleaned_headers
            if normalize_header(value) not in recognized_normalized
        )

        digest = observed_schema_sha256
        if digest is None:
            digest = _observed_schema_digest(cleaned_headers)
        else:
            digest = digest.strip().lower()
            if (
                len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise DownloadSchemaResolutionError(
                    "observed_schema_sha256 must be a 64-character SHA-256 hex digest"
                )

        return SchemaCompatibility(
            profile_name=self.name,
            profile_sha256=self.sha256_hex,
            observed_schema_sha256=digest,
            compatible=not missing_required,
            bindings=tuple(bindings),
            missing_required_fields=tuple(missing_required),
            unrecognized_headers=unrecognized,
            redundant_canonical_fields=tuple(redundant),
        )

    def inspect_member(self, member: MemberSchema) -> SchemaCompatibility:
        return self.inspect_headers(
            member.headers,
            observed_schema_sha256=member.schema_sha256,
        )


class DownloadSchemaRegistry:
    """Immutable registry that resolves explicit download profiles safely."""

    __slots__ = ("_profiles",)

    def __init__(self, profiles: Iterable[DownloadSchemaProfile]) -> None:
        by_name: dict[str, DownloadSchemaProfile] = {}
        for profile in profiles:
            key = profile.name.casefold()
            if key in by_name:
                raise DownloadSchemaDefinitionError(
                    f"duplicate download-schema profile name: {profile.name}"
                )
            by_name[key] = profile
        if not by_name:
            raise DownloadSchemaDefinitionError(
                "schema registry must contain at least one profile"
            )
        self._profiles = MappingProxyType(by_name)

    @property
    def profiles(self) -> tuple[DownloadSchemaProfile, ...]:
        return tuple(self._profiles.values())

    def get(self, name: str) -> DownloadSchemaProfile:
        key = name.strip().casefold()
        try:
            return self._profiles[key]
        except KeyError as exc:
            raise DownloadSchemaResolutionError(
                f"unknown download-schema profile: {name!r}"
            ) from exc

    def resolve(
        self,
        headers: Sequence[str],
        *,
        profile_name: str | None = None,
        reject_additive_drift: bool = False,
    ) -> tuple[DownloadSchemaProfile, SchemaCompatibility]:
        """Resolve by required structural contracts, never similarity scores."""

        if profile_name is not None:
            profile = self.get(profile_name)
            report = profile.inspect_headers(headers)
            self._enforce(report, reject_additive_drift=reject_additive_drift)
            return profile, report

        matches: list[tuple[DownloadSchemaProfile, SchemaCompatibility]] = []
        failures: list[SchemaCompatibility] = []
        for profile in self.profiles:
            report = profile.inspect_headers(headers)
            if report.compatible:
                matches.append((profile, report))
            else:
                failures.append(report)

        if len(matches) == 1:
            profile, report = matches[0]
            self._enforce(report, reject_additive_drift=reject_additive_drift)
            return profile, report

        if len(matches) > 1:
            names = ", ".join(profile.name for profile, _ in matches)
            raise DownloadSchemaResolutionError(
                "observed headers satisfy multiple semantic profiles; choose one "
                f"explicitly instead of guessing: {names}"
            )

        details = "; ".join(
            f"{report.profile_name}: missing {', '.join(report.missing_required_fields)}"
            for report in failures
        )
        raise DownloadSchemaResolutionError(
            "no registered download-schema profile is structurally compatible"
            + (f" ({details})" if details else "")
        )

    @staticmethod
    def _enforce(
        report: SchemaCompatibility,
        *,
        reject_additive_drift: bool,
    ) -> None:
        if not report.compatible:
            raise DownloadSchemaResolutionError(
                f"{report.profile_name} is missing required canonical fields: "
                + ", ".join(report.missing_required_fields)
            )
        if reject_additive_drift and report.unrecognized_headers:
            raise DownloadSchemaResolutionError(
                f"{report.profile_name} observed unrecognized additive headers: "
                + ", ".join(report.unrecognized_headers)
            )


def _observed_schema_digest(headers: Sequence[str]) -> str:
    pieces = tuple(normalize_header(header) for header in headers)
    payload = b"".join(
        len(piece.encode("utf-8")).to_bytes(8, "big") + piece.encode("utf-8")
        for piece in pieces
    )
    return sha256(payload).hexdigest()


# Current public USAspending contract-transaction download surface.
# Fixed names below are source-interface facts, not ProcureLens risk policy.
CONTRACT_TRANSACTION_DOWNLOAD = DownloadSchemaProfile(
    name="usaspending-contract-transaction-download",
    source_family="USAspending.gov public contract transaction download",
    fields=(
        DownloadFieldContract("award_id", ("contract_award_unique_key",), required=True),
        DownloadFieldContract("transaction_id", ("contract_transaction_unique_key",), required=True),
        DownloadFieldContract("piid", ("award_id_piid",)),
        DownloadFieldContract("modification_number", ("modification_number",)),
        DownloadFieldContract("parent_award_id", ("parent_award_id_piid",)),
        DownloadFieldContract("award_type_code", ("award_type_code",)),
        DownloadFieldContract("recipient_name", ("recipient_name", "recipient_name_raw")),
        DownloadFieldContract("recipient_uei", ("recipient_uei",)),
        DownloadFieldContract("recipient_legacy_id", ("recipient_duns",)),
        DownloadFieldContract("parent_recipient_name", ("recipient_parent_name", "recipient_parent_name_raw")),
        DownloadFieldContract("parent_recipient_uei", ("recipient_parent_uei",)),
        DownloadFieldContract("parent_recipient_legacy_id", ("recipient_parent_duns",)),
        DownloadFieldContract("action_date", ("action_date",), required=True),
        DownloadFieldContract("action_obligation", ("federal_action_obligation",), required=True),
        DownloadFieldContract("award_total_obligation", ("total_dollars_obligated",)),
        DownloadFieldContract("awarding_agency_code", ("awarding_agency_code",)),
        DownloadFieldContract("awarding_agency_name", ("awarding_agency_name",)),
        DownloadFieldContract("awarding_subtier_agency_code", ("awarding_sub_agency_code",)),
        DownloadFieldContract("awarding_subtier_agency_name", ("awarding_sub_agency_name",)),
        DownloadFieldContract("awarding_office_code", ("awarding_office_code",)),
        DownloadFieldContract("awarding_office_name", ("awarding_office_name",)),
        DownloadFieldContract("funding_agency_code", ("funding_agency_code",)),
        DownloadFieldContract("funding_agency_name", ("funding_agency_name",)),
        DownloadFieldContract("naics_code", ("naics_code",)),
        DownloadFieldContract("psc_code", ("product_or_service_code",)),
        DownloadFieldContract("description", ("transaction_description",)),
        DownloadFieldContract("extent_competed_code", ("extent_competed_code",)),
        DownloadFieldContract("extent_competed_description", ("extent_competed",)),
        DownloadFieldContract("number_of_offers_received", ("number_of_offers_received",)),
        DownloadFieldContract("other_than_full_and_open_code", ("other_than_full_and_open_competition_code",)),
        DownloadFieldContract("other_than_full_and_open_description", ("other_than_full_and_open_competition",)),
        DownloadFieldContract("solicitation_procedure_code", ("solicitation_procedures_code",)),
        DownloadFieldContract("solicitation_procedure_description", ("solicitation_procedures",)),
    ),
)


DEFAULT_DOWNLOAD_SCHEMA_REGISTRY = DownloadSchemaRegistry(
    (CONTRACT_TRANSACTION_DOWNLOAD,)
)
