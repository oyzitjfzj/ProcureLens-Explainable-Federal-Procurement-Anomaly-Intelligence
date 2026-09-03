"""Explicit vendor identity resolution for ProcureLens.

Federal recipients can be represented by entity-level identifiers and by an
ultimate-parent recipient. ProcureLens keeps those scopes separate so a
subsidiary is never silently merged into its corporate parent.

Resolution priority is factual and deterministic:
UEI -> legacy recipient identifier -> normalized reported name.

Name fallback performs only whitespace normalization and case-folding. It is
not fuzzy entity resolution, and its weaker identity method remains visible to
later quality and analysis layers.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
import json
from typing import Any

from procurelens.domain.transaction import ProcurementTransaction


class VendorIdentityError(ValueError):
    """Raised when vendor identity evidence is internally inconsistent."""


class VendorIdentityScope(str, Enum):
    ENTITY = "entity"
    ULTIMATE_PARENT = "ultimate_parent"


class VendorIdentityMethod(str, Enum):
    UEI = "uei"
    LEGACY_ID = "legacy_id"
    NORMALIZED_NAME = "normalized_name"


@dataclass(frozen=True, slots=True)
class VendorIdentity:
    """One explicit vendor identity at one grouping scope."""

    scope: VendorIdentityScope
    method: VendorIdentityMethod
    value: str
    display_name: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "scope", VendorIdentityScope(self.scope))
        object.__setattr__(self, "method", VendorIdentityMethod(self.method))
        value = self.value.strip()
        if not value:
            raise VendorIdentityError("vendor identity value must not be blank")
        if self.method in (VendorIdentityMethod.UEI, VendorIdentityMethod.LEGACY_ID):
            value = value.upper()
        elif self.method is VendorIdentityMethod.NORMALIZED_NAME:
            value = _normalize_name(value)
            if value is None:
                raise VendorIdentityError("normalized vendor name must not be blank")
        object.__setattr__(self, "value", value)

        if self.display_name is not None:
            display = " ".join(self.display_name.split()) or None
            object.__setattr__(self, "display_name", display)

    @property
    def canonical_identifier(self) -> str:
        """Identifier without grouping scope, useful for entity-vs-parent comparison."""

        return f"{self.method.value}:{self.value}"

    @property
    def key(self) -> str:
        """Stable grouping key; scope is included to prevent silent cross-scope merges."""

        return f"{self.scope.value}:{self.canonical_identifier}"

    @property
    def uses_stable_identifier(self) -> bool:
        """True for reported government identifiers, false for name fallback."""

        return self.method is not VendorIdentityMethod.NORMALIZED_NAME

    @property
    def sha256_hex(self) -> str:
        return _digest(
            {
                "scope": self.scope.value,
                "method": self.method.value,
                "value": self.value,
            }
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "scope": self.scope.value,
            "method": self.method.value,
            "value": self.value,
            "canonical_identifier": self.canonical_identifier,
            "key": self.key,
            "uses_stable_identifier": self.uses_stable_identifier,
            "display_name": self.display_name,
            "sha256": self.sha256_hex,
        }


@dataclass(frozen=True, slots=True)
class VendorIdentityResolution:
    """Entity and ultimate-parent views for one procurement transaction."""

    transaction_id: str
    entity: VendorIdentity | None
    ultimate_parent: VendorIdentity | None
    entity_unavailable_reason: str | None = None
    ultimate_parent_unavailable_reason: str | None = None

    def __post_init__(self) -> None:
        txid = self.transaction_id.strip()
        if not txid:
            raise VendorIdentityError("transaction_id must not be blank")
        object.__setattr__(self, "transaction_id", txid)
        self._validate_slot(
            "entity",
            self.entity,
            VendorIdentityScope.ENTITY,
            self.entity_unavailable_reason,
        )
        self._validate_slot(
            "ultimate_parent",
            self.ultimate_parent,
            VendorIdentityScope.ULTIMATE_PARENT,
            self.ultimate_parent_unavailable_reason,
        )

    @staticmethod
    def _validate_slot(
        label: str,
        identity: VendorIdentity | None,
        expected_scope: VendorIdentityScope,
        reason: str | None,
    ) -> None:
        if identity is not None:
            if identity.scope is not expected_scope:
                raise VendorIdentityError(f"{label} identity has the wrong scope")
            if reason is not None:
                raise VendorIdentityError(
                    f"available {label} identity cannot carry an unavailable reason"
                )
            return
        cleaned = None if reason is None else reason.strip()
        if not cleaned:
            raise VendorIdentityError(
                f"missing {label} identity requires an unavailable reason"
            )

    @property
    def parent_matches_entity_identifier(self) -> bool | None:
        """Whether reported entity and parent resolve to the same underlying identifier."""

        if self.entity is None or self.ultimate_parent is None:
            return None
        return (
            self.entity.canonical_identifier
            == self.ultimate_parent.canonical_identifier
        )

    def get(self, scope: VendorIdentityScope) -> VendorIdentity | None:
        scope = VendorIdentityScope(scope)
        if scope is VendorIdentityScope.ENTITY:
            return self.entity
        return self.ultimate_parent

    def as_dict(self) -> dict[str, Any]:
        return {
            "transaction_id": self.transaction_id,
            "entity": None if self.entity is None else self.entity.as_dict(),
            "ultimate_parent": (
                None if self.ultimate_parent is None else self.ultimate_parent.as_dict()
            ),
            "entity_unavailable_reason": self.entity_unavailable_reason,
            "ultimate_parent_unavailable_reason": self.ultimate_parent_unavailable_reason,
            "parent_matches_entity_identifier": self.parent_matches_entity_identifier,
        }


def resolve_vendor_identity(
    transaction: ProcurementTransaction,
    scope: VendorIdentityScope = VendorIdentityScope.ENTITY,
) -> VendorIdentity | None:
    """Resolve one requested scope without falling across entity/parent boundaries."""

    if not isinstance(transaction, ProcurementTransaction):
        raise TypeError("transaction must be ProcurementTransaction")
    scope = VendorIdentityScope(scope)

    if scope is VendorIdentityScope.ENTITY:
        return _from_fields(
            scope,
            uei=transaction.recipient_uei,
            legacy_id=transaction.recipient_legacy_id,
            name=transaction.recipient_name,
        )

    return _from_fields(
        scope,
        uei=transaction.parent_recipient_uei,
        legacy_id=transaction.parent_recipient_legacy_id,
        name=transaction.parent_recipient_name,
    )


def resolve_vendor_identities(
    transaction: ProcurementTransaction,
) -> VendorIdentityResolution:
    """Resolve both supported vendor views and preserve missingness explicitly."""

    if not isinstance(transaction, ProcurementTransaction):
        raise TypeError("transaction must be ProcurementTransaction")
    entity = resolve_vendor_identity(transaction, VendorIdentityScope.ENTITY)
    parent = resolve_vendor_identity(
        transaction, VendorIdentityScope.ULTIMATE_PARENT
    )
    return VendorIdentityResolution(
        transaction_id=transaction.transaction_id,
        entity=entity,
        ultimate_parent=parent,
        entity_unavailable_reason=(
            None if entity is not None else "recipient_identity_missing"
        ),
        ultimate_parent_unavailable_reason=(
            None if parent is not None else "ultimate_parent_identity_missing"
        ),
    )


def _from_fields(
    scope: VendorIdentityScope,
    *,
    uei: str | None,
    legacy_id: str | None,
    name: str | None,
) -> VendorIdentity | None:
    display = _display_name(name)
    if uei is not None and uei.strip():
        return VendorIdentity(
            scope,
            VendorIdentityMethod.UEI,
            uei,
            display,
        )
    if legacy_id is not None and legacy_id.strip():
        return VendorIdentity(
            scope,
            VendorIdentityMethod.LEGACY_ID,
            legacy_id,
            display,
        )
    normalized = _normalize_name(name)
    if normalized is not None:
        return VendorIdentity(
            scope,
            VendorIdentityMethod.NORMALIZED_NAME,
            normalized,
            display,
        )
    return None


def _display_name(value: str | None) -> str | None:
    if value is None:
        return None
    return " ".join(value.split()) or None


def _normalize_name(value: str | None) -> str | None:
    display = _display_name(value)
    return None if display is None else display.casefold()


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
