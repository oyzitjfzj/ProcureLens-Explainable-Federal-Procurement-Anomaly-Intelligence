from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from types import SimpleNamespace

import pytest

import procurelens.pipeline.run as run_module
import procurelens.pipeline.usaspending_live as live_module
from procurelens.runtime.publication import (
    PublicationError,
    PublicationRunKind,
    publish_live_usaspending_run,
)


def test_live_publication_preserves_source_preparation_and_analysis_provenance(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeAnalysis:
        def __init__(self) -> None:
            self.evidence_sha256 = "a" * 64
            self.manifest = _Manifest()
            payload = b'{"records":[]}\n'
            export = SimpleNamespace(
                payload_bytes=payload,
                payload_sha256=sha256(payload).hexdigest(),
                format=SimpleNamespace(value="json"),
                media_type="application/json",
            )
            self.model_review = SimpleNamespace(serialized_exports=(export,))

    class _FakeLive:
        def __init__(self, analysis) -> None:
            self.analysis = analysis
            self.prepared = _Prepared()
            self.evidence_sha256 = "d" * 64

        def as_dict(self):
            return {
                "prepared_dataset_sha256": "e" * 64,
                "analysis_sha256": self.analysis.evidence_sha256,
                "evidence_sha256": self.evidence_sha256,
            }

    monkeypatch.setattr(run_module, "ProcureLensAnalysisRun", _FakeAnalysis)
    monkeypatch.setattr(live_module, "LiveUSAspendingAnalysisRun", _FakeLive)

    run = _FakeLive(_FakeAnalysis())
    receipt = publish_live_usaspending_run(
        run,
        tmp_path / "runs",
        bundle_name="live-source-review",
    )

    assert receipt.run_kind is PublicationRunKind.LIVE_USASPENDING
    assert receipt.run_evidence_sha256 == run.evidence_sha256
    assert receipt.analysis_evidence_sha256 == run.analysis.evidence_sha256

    bundle = receipt.bundle_path
    expected = {
        "manifest.json",
        "publication.json",
        "live_run.json",
        "source/live_plan.json",
        "source/prepared_dataset.json",
        "exports/export_00.json",
    }
    observed = {
        path.relative_to(bundle).as_posix()
        for path in bundle.rglob("*")
        if path.is_file()
    }
    assert observed == expected

    publication = json.loads((bundle / "publication.json").read_text("utf-8"))
    assert publication["schema"] == {
        "name": "procurelens_publication_bundle",
        "version": 2,
    }
    assert publication["run_kind"] == "live_usaspending_analysis"
    assert publication["run_evidence_sha256"] == run.evidence_sha256
    assert publication["analysis_evidence_sha256"] == run.analysis.evidence_sha256

    prepared = json.loads(
        (bundle / "source/prepared_dataset.json").read_text("utf-8")
    )
    assert prepared["transaction_count"] == 4
    assert prepared["transaction_population_sha256"] == "f" * 64
    assert "transactions" not in prepared

    assert (bundle / "exports/export_00.json").read_bytes() == b'{"records":[]}\n'

    with pytest.raises(PublicationError, match="destination bundle already exists"):
        publish_live_usaspending_run(
            run,
            tmp_path / "runs",
            bundle_name="live-source-review",
        )


def test_live_publication_never_removes_another_publishers_lock(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeAnalysis:
        def __init__(self) -> None:
            self.evidence_sha256 = "a" * 64
            self.manifest = _Manifest()
            payload = b'{"records":[]}\n'
            export = SimpleNamespace(
                payload_bytes=payload,
                payload_sha256=sha256(payload).hexdigest(),
                format=SimpleNamespace(value="json"),
                media_type="application/json",
            )
            self.model_review = SimpleNamespace(serialized_exports=(export,))

    class _FakeLive:
        def __init__(self, analysis) -> None:
            self.analysis = analysis
            self.prepared = _Prepared()
            self.evidence_sha256 = "d" * 64

        def as_dict(self):
            return {
                "prepared_dataset_sha256": "e" * 64,
                "analysis_sha256": self.analysis.evidence_sha256,
                "evidence_sha256": self.evidence_sha256,
            }

    monkeypatch.setattr(run_module, "ProcureLensAnalysisRun", _FakeAnalysis)
    monkeypatch.setattr(live_module, "LiveUSAspendingAnalysisRun", _FakeLive)

    root = tmp_path / "runs"
    root.mkdir()
    lock = root / ".live-source-review.publish.lock"
    sentinel = b"other-publisher-owns-this-lock\n"
    lock.write_bytes(sentinel)

    with pytest.raises(
        PublicationError,
        match="publication lock already exists for this bundle name",
    ):
        publish_live_usaspending_run(
            _FakeLive(_FakeAnalysis()),
            root,
            bundle_name="live-source-review",
        )

    assert lock.read_bytes() == sentinel
    assert not (root / "live-source-review").exists()


@dataclass(frozen=True)
class _Manifest:
    evidence_sha256: str = "b" * 64

    def as_dict(self, *, include_sha: bool = True):
        payload = {
            "run_name": "live-source-review",
            "recipe_sha256": "c" * 64,
        }
        if include_sha:
            payload["evidence_sha256"] = self.evidence_sha256
        return payload


@dataclass(frozen=True)
class _Plan:
    sha256_hex: str = "1" * 64

    def as_dict(self, *, include_sha: bool = True):
        payload = {
            "name": "live-source-review",
            "filters": {
                "time_period": [
                    {"start_date": "2026-01-01", "end_date": "2026-01-31"}
                ]
            },
        }
        if include_sha:
            payload["sha256"] = self.sha256_hex
        return payload


@dataclass(frozen=True)
class _Prepared:
    plan: _Plan = _Plan()

    def as_dict(self, *, include_sha: bool = True):
        payload = {
            "plan_sha256": self.plan.sha256_hex,
            "artifact": {"sha256": "2" * 64},
            "load_report": {"complete": True, "transactions_emitted": 4},
            "quality_gate": {"status": "ready", "allowed": True},
            "transaction_count": 4,
            "transaction_population_sha256": "f" * 64,
            "analysis_allowed": True,
            "analysis_block_reason": None,
        }
        if include_sha:
            payload["evidence_sha256"] = "e" * 64
        return payload
