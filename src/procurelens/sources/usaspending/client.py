"""Reliable USAspending API control-plane client for ProcureLens.

The client intentionally separates HTTP acquisition from source normalization.
It handles deterministic request fingerprints, bounded retry budgets, server
back-pressure, download-count preflight, job creation, and status polling.

Large CSV/ZIP artifact transfer is a separate responsibility so downloads can
later be made atomic, resumable, hashed, and provenance-aware without coupling
those concerns to API control-plane requests.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from hashlib import sha256
import json
import random
import time
from types import MappingProxyType
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen


class USAspendingClientError(RuntimeError):
    """Base error for USAspending acquisition failures."""


class USAspendingProtocolError(USAspendingClientError):
    """Raised when USAspending returns an unexpected response contract."""


class USAspendingHTTPError(USAspendingClientError):
    """Raised when an HTTP request cannot be completed successfully."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        response_body: bytes | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body


class USAspendingRetryExhausted(USAspendingHTTPError):
    """Raised when the bounded retry policy can no longer retry safely."""


@dataclass(frozen=True, slots=True)
class TransportResponse:
    status_code: int
    headers: Mapping[str, str]
    body: bytes

    def __post_init__(self) -> None:
        if not (100 <= self.status_code <= 599):
            raise ValueError("status_code must be a valid HTTP status")
        object.__setattr__(
            self,
            "headers",
            MappingProxyType({str(k).casefold(): str(v) for k, v in self.headers.items()}),
        )


class HTTPTransport(Protocol):
    """Minimal injectable HTTP boundary used by USAspendingClient."""

    def request(
        self,
        *,
        method: str,
        url: str,
        headers: Mapping[str, str],
        body: bytes | None,
        timeout_seconds: float,
    ) -> TransportResponse: ...


class UrllibTransport:
    """Dependency-free synchronous transport built on Python's standard library."""

    def request(
        self,
        *,
        method: str,
        url: str,
        headers: Mapping[str, str],
        body: bytes | None,
        timeout_seconds: float,
    ) -> TransportResponse:
        request = Request(url=url, data=body, headers=dict(headers), method=method)
        try:
            with urlopen(request, timeout=timeout_seconds) as response:
                return TransportResponse(
                    status_code=int(response.status),
                    headers=dict(response.headers.items()),
                    body=response.read(),
                )
        except HTTPError as exc:
            return TransportResponse(
                status_code=int(exc.code),
                headers=dict(exc.headers.items()) if exc.headers is not None else {},
                body=exc.read(),
            )
        except (URLError, TimeoutError, OSError) as exc:
            raise USAspendingHTTPError(f"network request failed: {exc}") from exc


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """Bounded retry policy for read-safe API operations.

    max_total_delay_seconds is a retry budget: retries stop instead of creating
    unbounded request amplification during a service incident.
    """

    max_attempts: int = 4
    base_delay_seconds: float = 0.5
    max_delay_seconds: float = 8.0
    max_total_delay_seconds: float = 20.0
    jitter_ratio: float = 0.25
    retry_status_codes: frozenset[int] = field(
        default_factory=lambda: frozenset({408, 425, 429, 500, 502, 503, 504})
    )

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        for name in (
            "base_delay_seconds",
            "max_delay_seconds",
            "max_total_delay_seconds",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must not be negative")
        if self.max_delay_seconds < self.base_delay_seconds:
            raise ValueError("max_delay_seconds must be >= base_delay_seconds")
        if not (0 <= self.jitter_ratio <= 1):
            raise ValueError("jitter_ratio must be between 0 and 1")
        if any(not (400 <= code <= 599) for code in self.retry_status_codes):
            raise ValueError("retry_status_codes must contain HTTP error statuses")


@dataclass(frozen=True, slots=True)
class RequestFingerprint:
    """Stable identity for one acquisition request."""

    sha256_hex: str
    canonical_json: str


@dataclass(frozen=True, slots=True)
class DownloadCount:
    calculated_transaction_count: int
    maximum_transaction_limit: int
    transaction_rows_gt_limit: bool
    calculated_count: int
    spending_level: str
    maximum_limit: int
    rows_gt_limit: bool
    messages: tuple[str, ...]
    request_fingerprint: RequestFingerprint


@dataclass(frozen=True, slots=True)
class DownloadJob:
    status_url: str
    file_name: str
    file_url: str
    download_request: Mapping[str, Any]
    request_fingerprint: RequestFingerprint

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "download_request", MappingProxyType(dict(self.download_request))
        )


@dataclass(frozen=True, slots=True)
class DownloadStatus:
    status: str
    file_name: str
    file_url: str | None
    message: str | None
    total_rows: int | None
    total_columns: int | None
    total_size: float | None
    seconds_elapsed: float | None
    checked_at: datetime

    @property
    def terminal(self) -> bool:
        return self.status in {"finished", "failed"}


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise USAspendingProtocolError(
            "request payload must be finite JSON-compatible data"
        ) from exc


def _fingerprint(endpoint: str, payload: Mapping[str, Any]) -> RequestFingerprint:
    canonical = _canonical_json({"endpoint": endpoint, "payload": payload})
    return RequestFingerprint(
        sha256_hex=sha256(canonical.encode("utf-8")).hexdigest(),
        canonical_json=canonical,
    )


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise USAspendingProtocolError(f"{name} must be a JSON object")
    return value


def _required_text(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise USAspendingProtocolError(f"response field {key!r} must be non-blank text")
    return value.strip()


def _optional_text(payload: Mapping[str, Any], key: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise USAspendingProtocolError(f"response field {key!r} must be text or null")
    return value.strip() or None


def _required_int(payload: Mapping[str, Any], key: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise USAspendingProtocolError(f"response field {key!r} must be an integer")
    return value


def _optional_int(payload: Mapping[str, Any], key: str) -> int | None:
    value = payload.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise USAspendingProtocolError(
            f"response field {key!r} must be an integer or null"
        )
    return value


def _required_bool(payload: Mapping[str, Any], key: str) -> bool:
    value = payload.get(key)
    if not isinstance(value, bool):
        raise USAspendingProtocolError(f"response field {key!r} must be boolean")
    return value


def _optional_float(payload: Mapping[str, Any], key: str) -> float | None:
    value = payload.get(key)
    if value is None:
        return None
    if isinstance(value, bool):
        raise USAspendingProtocolError(
            f"response field {key!r} must be numeric or null"
        )
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise USAspendingProtocolError(
            f"response field {key!r} must be numeric or null"
        ) from exc
    if parsed != parsed or parsed in (float("inf"), float("-inf")):
        raise USAspendingProtocolError(f"response field {key!r} must be finite")
    return parsed


class USAspendingClient:
    """Reliable control-plane client for USAspending acquisition.

    Read-safe endpoints use bounded retries. Download-job creation intentionally
    does not auto-retry because a failed client response does not prove that the
    server failed to create the job; blind retries can create duplicate jobs.
    """

    __slots__ = (
        "_transport",
        "_retry_policy",
        "_sleep",
        "_random",
        "_wall_clock",
        "_monotonic",
        "_base_origin",
        "base_url",
        "timeout_seconds",
        "user_agent",
    )

    def __init__(
        self,
        *,
        base_url: str = "https://api.usaspending.gov/api/v2/",
        timeout_seconds: float = 30.0,
        user_agent: str = "ProcureLens/0",
        transport: HTTPTransport | None = None,
        retry_policy: RetryPolicy | None = None,
        sleep: Callable[[float], None] = time.sleep,
        random_source: Callable[[], float] = random.random,
        wall_clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
        monotonic_clock: Callable[[], float] = time.monotonic,
    ) -> None:
        parsed = urlparse(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("base_url must be an absolute HTTP(S) URL")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if not user_agent.strip():
            raise ValueError("user_agent must not be blank")

        normalized_base = base_url.rstrip("/") + "/"
        self.base_url = normalized_base
        self._base_origin = (parsed.scheme.casefold(), parsed.netloc.casefold())
        self.timeout_seconds = float(timeout_seconds)
        self.user_agent = user_agent.strip()
        self._transport = transport or UrllibTransport()
        self._retry_policy = retry_policy or RetryPolicy()
        self._sleep = sleep
        self._random = random_source
        self._wall_clock = wall_clock
        self._monotonic = monotonic_clock

    def count_transactions(
        self,
        filters: Mapping[str, Any],
        *,
        spending_level: str = "transactions",
    ) -> DownloadCount:
        """Preflight the current server-side row limits for a filter population."""

        if spending_level not in {"transactions", "awards", "subawards"}:
            raise ValueError("spending_level must be transactions, awards, or subawards")
        payload = {
            "filters": dict(_require_mapping(filters, "filters")),
            "spending_level": spending_level,
        }
        endpoint = "download/count/"
        fingerprint = _fingerprint(endpoint, payload)
        response = self._request_json(
            method="POST",
            endpoint=endpoint,
            payload=payload,
            retry_safe=True,
        )

        messages_value = response.get("messages", ())
        if messages_value is None:
            messages_value = ()
        if not isinstance(messages_value, Sequence) or isinstance(
            messages_value, (str, bytes)
        ):
            raise USAspendingProtocolError("response field 'messages' must be an array")
        messages: list[str] = []
        for message in messages_value:
            if not isinstance(message, str):
                raise USAspendingProtocolError("download-count messages must be text")
            messages.append(message)

        return DownloadCount(
            calculated_transaction_count=_required_int(
                response, "calculated_transaction_count"
            ),
            maximum_transaction_limit=_required_int(
                response, "maximum_transaction_limit"
            ),
            transaction_rows_gt_limit=_required_bool(
                response, "transaction_rows_gt_limit"
            ),
            calculated_count=_required_int(response, "calculated_count"),
            spending_level=_required_text(response, "spending_level"),
            maximum_limit=_required_int(response, "maximum_limit"),
            rows_gt_limit=_required_bool(response, "rows_gt_limit"),
            messages=tuple(messages),
            request_fingerprint=fingerprint,
        )

    def start_search_download(
        self,
        filters: Mapping[str, Any],
        *,
        columns: Sequence[str] | None = None,
        spending_levels: Sequence[str] = ("transactions",),
        file_format: str = "csv",
        limit: int | None = None,
    ) -> DownloadJob:
        """Start an asynchronous USAspending search-download job.

        This operation is deliberately single-attempt. If the connection fails
        after USAspending accepted the job, an automatic retry could create a
        second job with the same expensive request.
        """

        allowed_levels = {"transactions", "awards", "subawards"}
        levels = tuple(spending_levels)
        if not levels or any(level not in allowed_levels for level in levels):
            raise ValueError(
                "spending_levels must contain transactions, awards, or subawards"
            )
        if len(set(levels)) != len(levels):
            raise ValueError("spending_levels must not contain duplicates")
        if file_format not in {"csv", "tsv", "pstxt"}:
            raise ValueError("file_format must be csv, tsv, or pstxt")
        if limit is not None and (
            isinstance(limit, bool) or not isinstance(limit, int) or limit < 0
        ):
            raise ValueError("limit must be a non-negative integer or None")

        payload: dict[str, Any] = {
            "filters": dict(_require_mapping(filters, "filters")),
            "spending_level": list(levels),
            "file_format": file_format,
        }
        if columns is not None:
            cleaned_columns = []
            seen_columns: set[str] = set()
            for column in columns:
                if not isinstance(column, str) or not column.strip():
                    raise ValueError("columns must contain non-blank strings")
                cleaned = column.strip()
                if cleaned not in seen_columns:
                    cleaned_columns.append(cleaned)
                    seen_columns.add(cleaned)
            payload["columns"] = cleaned_columns
        if limit is not None:
            payload["limit"] = limit

        endpoint = "download/search/"
        fingerprint = _fingerprint(endpoint, payload)
        response = self._request_json(
            method="POST",
            endpoint=endpoint,
            payload=payload,
            retry_safe=False,
        )
        download_request = _require_mapping(
            response.get("download_request"), "download_request"
        )
        return DownloadJob(
            status_url=self._same_origin_url(_required_text(response, "status_url")),
            file_name=_required_text(response, "file_name"),
            file_url=self._resolve_public_url(_required_text(response, "file_url")),
            download_request=download_request,
            request_fingerprint=fingerprint,
        )

    def get_download_status(self, status_url: str) -> DownloadStatus:
        """Read one download-job status response."""

        absolute = self._same_origin_url(status_url)
        response = self._request_json_url(
            method="GET",
            url=absolute,
            payload=None,
            retry_safe=True,
        )

        status = _required_text(response, "status").casefold()
        if status not in {"ready", "running", "finished", "failed"}:
            raise USAspendingProtocolError(
                f"unknown USAspending download status: {status!r}"
            )

        return DownloadStatus(
            status=status,
            file_name=_required_text(response, "file_name"),
            file_url=(
                self._resolve_public_url(value)
                if (value := _optional_text(response, "file_url")) is not None
                else None
            ),
            message=_optional_text(response, "message"),
            total_rows=_optional_int(response, "total_rows"),
            total_columns=_optional_int(response, "total_columns"),
            total_size=_optional_float(response, "total_size"),
            seconds_elapsed=_optional_float(response, "seconds_elapsed"),
            checked_at=self._aware_now(),
        )

    def wait_for_download(
        self,
        job: DownloadJob,
        *,
        timeout_seconds: float = 900.0,
        initial_poll_seconds: float = 1.0,
        max_poll_seconds: float = 15.0,
        poll_multiplier: float = 1.5,
        jitter_ratio: float = 0.1,
    ) -> DownloadStatus:
        """Poll a job serially with bounded adaptive intervals until terminal."""

        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if initial_poll_seconds <= 0 or max_poll_seconds < initial_poll_seconds:
            raise ValueError("poll intervals are invalid")
        if poll_multiplier < 1:
            raise ValueError("poll_multiplier must be >= 1")
        if not (0 <= jitter_ratio <= 1):
            raise ValueError("jitter_ratio must be between 0 and 1")

        started = self._monotonic()
        delay = initial_poll_seconds

        while True:
            status = self.get_download_status(job.status_url)
            if status.terminal:
                return status

            elapsed = self._monotonic() - started
            remaining = timeout_seconds - elapsed
            if remaining <= 0:
                raise USAspendingHTTPError(
                    f"download job did not finish within {timeout_seconds:g} seconds"
                )

            jitter = delay * jitter_ratio * ((self._random() * 2.0) - 1.0)
            sleep_for = max(0.0, min(delay + jitter, remaining))
            self._sleep(sleep_for)
            delay = min(max_poll_seconds, delay * poll_multiplier)

    def _aware_now(self) -> datetime:
        value = self._wall_clock()
        if not isinstance(value, datetime):
            raise USAspendingProtocolError("wall_clock must return datetime")
        if value.tzinfo is None or value.utcoffset() is None:
            raise USAspendingProtocolError("wall_clock must return timezone-aware datetime")
        return value.astimezone(timezone.utc)

    def _request_json(
        self,
        *,
        method: str,
        endpoint: str,
        payload: Mapping[str, Any] | None,
        retry_safe: bool,
    ) -> Mapping[str, Any]:
        return self._request_json_url(
            method=method,
            url=urljoin(self.base_url, endpoint.lstrip("/")),
            payload=payload,
            retry_safe=retry_safe,
        )

    def _request_json_url(
        self,
        *,
        method: str,
        url: str,
        payload: Mapping[str, Any] | None,
        retry_safe: bool,
    ) -> Mapping[str, Any]:
        body = None if payload is None else _canonical_json(payload).encode("utf-8")
        headers = {
            "accept": "application/json",
            "user-agent": self.user_agent,
        }
        if body is not None:
            headers["content-type"] = "application/json"

        policy = self._retry_policy
        total_delay = 0.0
        last_response: TransportResponse | None = None
        last_network_error: USAspendingHTTPError | None = None

        for attempt in range(1, policy.max_attempts + 1):
            try:
                response = self._transport.request(
                    method=method,
                    url=url,
                    headers=headers,
                    body=body,
                    timeout_seconds=self.timeout_seconds,
                )
                last_response = response
                last_network_error = None
            except USAspendingHTTPError as exc:
                last_network_error = exc
                if not retry_safe or attempt >= policy.max_attempts:
                    raise
                delay = self._bounded_retry_delay(
                    attempt=attempt,
                    headers={},
                    total_delay=total_delay,
                )
                total_delay += delay
                self._sleep(delay)
                continue

            if 200 <= response.status_code < 300:
                try:
                    decoded = json.loads(response.body.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise USAspendingProtocolError(
                        "USAspending returned a non-JSON success response"
                    ) from exc
                return _require_mapping(decoded, "USAspending response")

            retryable = (
                retry_safe
                and response.status_code in policy.retry_status_codes
                and attempt < policy.max_attempts
            )
            if retryable:
                delay = self._bounded_retry_delay(
                    attempt=attempt,
                    headers=response.headers,
                    total_delay=total_delay,
                )
                total_delay += delay
                self._sleep(delay)
                continue

            message = self._error_message(response)
            raise USAspendingHTTPError(
                message,
                status_code=response.status_code,
                response_body=response.body,
            )

        if last_network_error is not None:
            raise last_network_error
        if last_response is not None:
            raise USAspendingRetryExhausted(
                "USAspending retry attempts were exhausted",
                status_code=last_response.status_code,
                response_body=last_response.body,
            )
        raise USAspendingRetryExhausted("USAspending retry attempts were exhausted")

    def _bounded_retry_delay(
        self,
        *,
        attempt: int,
        headers: Mapping[str, str],
        total_delay: float,
    ) -> float:
        policy = self._retry_policy
        retry_after = self._retry_after_seconds(headers)
        if retry_after is None:
            exponential = min(
                policy.max_delay_seconds,
                policy.base_delay_seconds * (2 ** max(0, attempt - 1)),
            )
            jitter = exponential * policy.jitter_ratio * (
                (self._random() * 2.0) - 1.0
            )
            delay = max(0.0, exponential + jitter)
        else:
            # Never retry earlier than the server requested.
            delay = retry_after

        if total_delay + delay > policy.max_total_delay_seconds:
            raise USAspendingRetryExhausted(
                "retry delay budget exhausted before another safe attempt"
            )
        return delay

    def _retry_after_seconds(self, headers: Mapping[str, str]) -> float | None:
        value = headers.get("retry-after")
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

    @staticmethod
    def _error_message(response: TransportResponse) -> str:
        detail: str | None = None
        try:
            decoded = json.loads(response.body.decode("utf-8"))
            if isinstance(decoded, Mapping):
                for key in ("detail", "message", "error"):
                    value = decoded.get(key)
                    if isinstance(value, str) and value.strip():
                        detail = value.strip()
                        break
        except (UnicodeDecodeError, json.JSONDecodeError):
            pass
        suffix = f": {detail}" if detail else ""
        return f"USAspending HTTP {response.status_code}{suffix}"

    def _same_origin_url(self, value: str) -> str:
        absolute = urljoin(self.base_url, value)
        parsed = urlparse(absolute)
        origin = (parsed.scheme.casefold(), parsed.netloc.casefold())
        if origin != self._base_origin:
            raise USAspendingProtocolError(
                "USAspending status URL unexpectedly changed origin"
            )
        return absolute

    def _resolve_public_url(self, value: str) -> str:
        # File URLs are allowed to use USAspending's dedicated file host.
        parsed = urlparse(value)
        if parsed.scheme and parsed.netloc:
            if parsed.scheme not in {"http", "https"}:
                raise USAspendingProtocolError("file_url must use HTTP(S)")
            return value
        return urljoin(self.base_url, value)
