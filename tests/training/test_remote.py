"""Tests for :mod:`oplm.training.remote` (Tasks 4.1 + 4.2).

Task 4.1's tests exercise :class:`RemoteStore`, the fsspec-based checkpoint mirror,
over a ``file://`` filesystem (fast, no network): upload+finalize commits a
checkpoint via the manifest-last discipline, an uncommitted (no-manifest) checkpoint
directory is invisible to discovery, a downloaded file whose size disagrees with the
manifest raises, and rotation mirrors
``oplm.training.checkpoint._rotate_checkpoints``'s permanent/rolling semantics
exactly. A slow, optional moto-based ``s3://`` smoke test is included but skips
gracefully when ``moto`` is not installed.

Task 4.2's tests cover the trainer integration: :class:`~oplm.training.remote.
UploadManager`'s in-flight/queue/drop serialization and bounded drain (fake store,
fast, no real Trainer), and the end-to-end wiring -- a real pilot ``Trainer`` run
with ``train.remote_checkpoint_uri`` set mirrors its checkpoints to a ``file://``
store, a wiped local ``output_dir`` (simulated NVMe loss) is recovered purely from
the remote mirror via ``auto_resume``, remote rotation honors ``keep_every_n_steps``,
and an unset URI is a total no-op (the zero-behavior-change contract).
"""

from __future__ import annotations

import json
import shutil
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from oplm.training.remote import RemoteStore


def _write_local_checkpoint(local_dir: Path, files: dict[str, str]) -> list[Path]:
    """Write ``files`` (relpath -> content) under ``local_dir``; return relpaths."""
    local_dir.mkdir(parents=True, exist_ok=True)
    relpaths = []
    for relpath, content in files.items():
        path = local_dir / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        relpaths.append(Path(relpath))
    return relpaths


def test_upload_and_finalize_round_trip(tmp_path: Path) -> None:
    """Uploading files then finalizing writes a manifest that ``latest_committed`` sees."""
    local_dir = tmp_path / "local" / "checkpoint-100"
    relpaths = _write_local_checkpoint(local_dir, {"a.txt": "hello", "sub/b.txt": "world!!"})
    remote_root = tmp_path / "remote"
    store = RemoteStore(f"file://{remote_root}")

    store.upload_checkpoint(local_dir, files=relpaths, permanent=False, write_manifest=True)

    result = store.latest_committed()
    assert result is not None
    name, manifest = result
    assert name == "checkpoint-100"
    assert manifest["permanent"] is False
    assert manifest["files"] == {"a.txt": 5, "sub/b.txt": 7}

    manifest_path = remote_root / "checkpoint-100" / "manifest.json"
    assert manifest_path.is_file()
    on_disk = json.loads(manifest_path.read_text())
    assert on_disk == manifest


def test_upload_without_manifest_then_finalize_separately(tmp_path: Path) -> None:
    """``write_manifest=False`` uploads files without committing; a later ``finalize`` does."""
    local_dir = tmp_path / "local" / "checkpoint-200"
    relpaths = _write_local_checkpoint(local_dir, {"a.txt": "abc"})
    remote_root = tmp_path / "remote"
    store = RemoteStore(f"file://{remote_root}")

    store.upload_checkpoint(local_dir, files=relpaths, permanent=True, write_manifest=False)
    assert store.latest_committed() is None

    store.finalize("checkpoint-200", permanent=True)

    result = store.latest_committed()
    assert result is not None
    name, manifest = result
    assert name == "checkpoint-200"
    assert manifest["permanent"] is True
    assert manifest["files"] == {"a.txt": 3}


def test_finalize_twice_is_safe_and_excludes_its_own_manifest(tmp_path: Path) -> None:
    """Re-finalizing rewrites the manifest without ever listing itself as a file.

    Guards the ``relpath == _MANIFEST_NAME`` exclusion in ``finalize``'s listing --
    without it, a second ``finalize`` call would fold the previous ``manifest.json``
    into its own ``files`` dict.
    """
    local_dir = tmp_path / "local" / "checkpoint-600"
    relpaths = _write_local_checkpoint(local_dir, {"a.txt": "hello"})
    remote_root = tmp_path / "remote"
    store = RemoteStore(f"file://{remote_root}")

    store.upload_checkpoint(local_dir, files=relpaths, permanent=False, write_manifest=True)
    first_result = store.latest_committed()
    assert first_result is not None
    assert first_result[1]["files"] == {"a.txt": 5}

    # Upload one more file, then re-finalize (e.g. re-committing after an incremental
    # upload) as permanent this time.
    extra_relpaths = _write_local_checkpoint(local_dir, {"b.txt": "more!!"})
    store.upload_checkpoint(local_dir, files=extra_relpaths, permanent=True, write_manifest=False)
    store.finalize("checkpoint-600", permanent=True)

    result = store.latest_committed()
    assert result is not None
    name, manifest = result
    assert name == "checkpoint-600"
    assert manifest["permanent"] is True
    # manifest.json itself must never appear as one of its own checkpoint's files,
    # on either the first or a subsequent finalize.
    assert manifest["files"] == {"a.txt": 5, "b.txt": 6}


def test_checkpoint_without_manifest_is_invisible_to_latest_committed(tmp_path: Path) -> None:
    """A ``checkpoint-<step>/`` dir with files but no manifest.json is not discovered."""
    local_dir = tmp_path / "local" / "checkpoint-300"
    relpaths = _write_local_checkpoint(local_dir, {"a.txt": "x"})
    remote_root = tmp_path / "remote"
    store = RemoteStore(f"file://{remote_root}")

    store.upload_checkpoint(local_dir, files=relpaths, permanent=False, write_manifest=False)

    assert store.latest_committed() is None
    # The directory and file did land remotely -- just uncommitted.
    assert (remote_root / "checkpoint-300" / "a.txt").is_file()
    assert not (remote_root / "checkpoint-300" / "manifest.json").exists()


def test_latest_committed_picks_highest_numeric_step(tmp_path: Path) -> None:
    """Discovery orders by numeric step, not lexicographically (9000 < 10000)."""
    remote_root = tmp_path / "remote"
    store = RemoteStore(f"file://{remote_root}")

    for step in (9000, 10000):
        local_dir = tmp_path / "local" / f"checkpoint-{step}"
        relpaths = _write_local_checkpoint(local_dir, {"a.txt": "x"})
        store.upload_checkpoint(local_dir, files=relpaths, permanent=False, write_manifest=True)

    name, _ = store.latest_committed()  # type: ignore[misc]
    assert name == "checkpoint-10000"


def test_latest_committed_returns_none_on_empty_store(tmp_path: Path) -> None:
    """An empty (or nonexistent) remote root has no committed checkpoints."""
    store = RemoteStore(f"file://{tmp_path / 'does-not-exist-yet'}")
    assert store.latest_committed() is None


def test_download_checkpoint_round_trips_files(tmp_path: Path) -> None:
    """Downloading a committed checkpoint reproduces its files under the committed name."""
    local_dir = tmp_path / "local" / "checkpoint-400"
    relpaths = _write_local_checkpoint(local_dir, {"a.txt": "hello", "sub/b.txt": "world!!"})
    remote_root = tmp_path / "remote"
    store = RemoteStore(f"file://{remote_root}")
    store.upload_checkpoint(local_dir, files=relpaths, permanent=False, write_manifest=True)

    dest = tmp_path / "download"
    result = store.download_checkpoint("checkpoint-400", dest)

    assert result == dest / "checkpoint-400"
    assert not (dest / "checkpoint-400.tmp").exists()
    assert (dest / "checkpoint-400" / "a.txt").read_text() == "hello"
    assert (dest / "checkpoint-400" / "sub" / "b.txt").read_text() == "world!!"


def test_download_checkpoint_size_mismatch_raises(tmp_path: Path) -> None:
    """A file whose downloaded size disagrees with the manifest raises RuntimeError."""
    local_dir = tmp_path / "local" / "checkpoint-500"
    relpaths = _write_local_checkpoint(local_dir, {"a.txt": "hello"})
    remote_root = tmp_path / "remote"
    store = RemoteStore(f"file://{remote_root}")
    store.upload_checkpoint(local_dir, files=relpaths, permanent=False, write_manifest=True)

    # Corrupt the manifest to claim a size that disagrees with the real remote file.
    manifest_path = remote_root / "checkpoint-500" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"]["a.txt"] = 999
    manifest_path.write_text(json.dumps(manifest))

    dest = tmp_path / "download"
    with pytest.raises(RuntimeError):
        store.download_checkpoint("checkpoint-500", dest)


def test_rotate_keeps_permanent_and_newest_k(tmp_path: Path) -> None:
    """rotate(limit=1) over {100 perm, 150, 200 perm, 250} deletes only 150.

    Mirrors the local-rotation reference case (checkpoint.py's
    ``_rotate_checkpoints``): permanent checkpoints (here, manifest ``permanent: true``)
    are excluded from both the rolling count and deletion; 250 is the newest rolling
    checkpoint, so only 150 is removed.
    """
    remote_root = tmp_path / "remote"
    store = RemoteStore(f"file://{remote_root}")

    permanent_steps = {100, 200}
    for step in (100, 150, 200, 250):
        local_dir = tmp_path / "local" / f"checkpoint-{step}"
        relpaths = _write_local_checkpoint(local_dir, {"a.txt": "x"})
        store.upload_checkpoint(
            local_dir,
            files=relpaths,
            permanent=step in permanent_steps,
            write_manifest=True,
        )

    store.rotate(save_total_limit=1, keep_every_n_steps=None)

    assert (remote_root / "checkpoint-100").exists()
    assert not (remote_root / "checkpoint-150").exists()
    assert (remote_root / "checkpoint-200").exists()
    assert (remote_root / "checkpoint-250").exists()


def test_rotate_respects_keep_every_n_steps(tmp_path: Path) -> None:
    """``keep_every_n_steps`` also marks a step-boundary checkpoint permanent."""
    remote_root = tmp_path / "remote"
    store = RemoteStore(f"file://{remote_root}")

    for step in (100, 150, 200, 250):
        local_dir = tmp_path / "local" / f"checkpoint-{step}"
        relpaths = _write_local_checkpoint(local_dir, {"a.txt": "x"})
        store.upload_checkpoint(local_dir, files=relpaths, permanent=False, write_manifest=True)

    store.rotate(save_total_limit=1, keep_every_n_steps=100)

    assert (remote_root / "checkpoint-100").exists()
    assert not (remote_root / "checkpoint-150").exists()
    assert (remote_root / "checkpoint-200").exists()
    assert (remote_root / "checkpoint-250").exists()


def test_rotate_noop_when_limit_nonpositive(tmp_path: Path) -> None:
    """A ``save_total_limit <= 0`` disables rotation entirely, mirroring the local rule."""
    remote_root = tmp_path / "remote"
    store = RemoteStore(f"file://{remote_root}")

    for step in (100, 200):
        local_dir = tmp_path / "local" / f"checkpoint-{step}"
        relpaths = _write_local_checkpoint(local_dir, {"a.txt": "x"})
        store.upload_checkpoint(local_dir, files=relpaths, permanent=False, write_manifest=True)

    store.rotate(save_total_limit=0, keep_every_n_steps=None)

    assert (remote_root / "checkpoint-100").exists()
    assert (remote_root / "checkpoint-200").exists()


@pytest.mark.slow
def test_s3_smoke_via_moto(tmp_path: Path) -> None:
    """Optional smoke test: the same round trip works against a mocked S3 bucket."""
    pytest.importorskip("moto")
    pytest.importorskip("s3fs")
    from moto import mock_aws

    with mock_aws():
        import boto3

        boto3.client("s3", region_name="us-east-1").create_bucket(Bucket="oplm-test-bucket")

        store = RemoteStore("s3://oplm-test-bucket/checkpoints")
        local_dir = tmp_path / "local" / "checkpoint-100"
        relpaths = _write_local_checkpoint(local_dir, {"a.txt": "hello"})

        store.upload_checkpoint(local_dir, files=relpaths, permanent=False, write_manifest=True)

        result = store.latest_committed()
        assert result is not None
        name, manifest = result
        assert name == "checkpoint-100"
        assert manifest["files"] == {"a.txt": 5}

        dest = tmp_path / "download"
        downloaded = store.download_checkpoint("checkpoint-100", dest)
        assert (downloaded / "a.txt").read_text() == "hello"


# --- Task 4.2: UploadManager + trainer integration -------------------------------


class _FakeAccelerator:
    """Stand-in for Accelerate's Accelerator: only the attributes UploadManager /
    build_upload_job actually read.
    """

    def __init__(
        self,
        *,
        is_main_process: bool = True,
        num_processes: int = 1,
        process_index: int = 0,
        local_process_index: int = 0,
    ) -> None:
        self.is_main_process = is_main_process
        self.num_processes = num_processes
        self.process_index = process_index
        self.local_process_index = local_process_index


class _SlowFakeStore:
    """Fake RemoteStore: ``upload_checkpoint`` sleeps; every call is recorded in order."""

    def __init__(self, sleep_seconds: float = 0.1) -> None:
        self.sleep_seconds = sleep_seconds
        self.uploaded: list[str] = []
        self.finalized: list[str] = []
        self.rotated: list[tuple[int, int | None]] = []
        self._lock = threading.Lock()

    def upload_checkpoint(
        self, local_dir: Path, *, files: list[Path], permanent: bool, write_manifest: bool
    ) -> None:
        time.sleep(self.sleep_seconds)
        with self._lock:
            self.uploaded.append(local_dir.name)

    def finalize(self, name: str, *, permanent: bool) -> None:
        with self._lock:
            self.finalized.append(name)

    def rotate(self, save_total_limit: int, keep_every_n_steps: int | None) -> None:
        with self._lock:
            self.rotated.append((save_total_limit, keep_every_n_steps))


def _job(local_dir: Path) -> Any:
    from oplm.training.remote import UploadJob

    return UploadJob(
        local_dir=local_dir,
        files=[],
        shared_files=None,
        permanent=False,
        save_total_limit=3,
        keep_every_n_steps=None,
    )


def test_upload_manager_serializes_and_drops_stale_queued_jobs(tmp_path: Path) -> None:
    """One in-flight upload at a time; a superseded queued job is dropped, not run.

    Submits three jobs in quick succession: the first starts uploading immediately
    (idle -> busy), the second is queued behind it, and the third -- arriving while
    the first is still uploading and a job is already queued -- drops the second
    (superseded) and takes its place in the single queue slot. Only the first and
    third ever reach the fake store.
    """
    from oplm.training.remote import UploadManager

    store = _SlowFakeStore(sleep_seconds=0.2)
    manager = UploadManager(store, _FakeAccelerator())

    manager.submit(_job(tmp_path / "checkpoint-100"))
    time.sleep(0.02)  # let the first upload actually start (idle -> busy)
    manager.submit(_job(tmp_path / "checkpoint-200"))
    manager.submit(_job(tmp_path / "checkpoint-300"))  # drops 200, takes its slot

    manager.drain(timeout=5.0)

    assert store.uploaded == ["checkpoint-100", "checkpoint-300"]
    assert store.finalized == ["checkpoint-100", "checkpoint-300"]
    assert store.rotated == [(3, None), (3, None)]


def test_upload_manager_drain_bounded_by_timeout(tmp_path: Path) -> None:
    """``drain`` returns at its timeout instead of blocking for the full upload."""
    from oplm.training.remote import UploadManager

    store = _SlowFakeStore(sleep_seconds=2.0)
    manager = UploadManager(store, _FakeAccelerator())
    manager.submit(_job(tmp_path / "checkpoint-1"))

    start = time.monotonic()
    manager.drain(timeout=0.1)
    elapsed = time.monotonic() - start

    assert elapsed < 1.0
    assert store.uploaded == []  # the 2s upload had not finished yet

    # Let the background upload actually finish before the test ends, so no live
    # thread outlives it.
    manager.drain(timeout=5.0)
    assert store.uploaded == ["checkpoint-1"]


def test_remote_rotation_via_upload_manager_honors_keep_every_n_steps(tmp_path: Path) -> None:
    """The UploadManager finalize -> rotate tail (Task 4.2) honors ``keep_every_n_steps``.

    Drives ``build_upload_job`` + ``UploadManager.submit``/``drain`` directly against
    manually constructed single-rank checkpoint dirs, draining after each submission
    so the sequence is deterministic (no race between a live Trainer's save cadence
    and upload speed -- see the docstring on the wipe/resume e2e test below for why
    that combination is deliberately not used here). Complements ``RemoteStore``'s
    own ``test_rotate_respects_keep_every_n_steps`` (Task 4.1) at the level Task 4.2
    actually adds: the trainer's own wiring from a committed checkpoint through
    ``build_upload_job`` and ``UploadManager`` to ``RemoteStore.rotate``.
    """
    from oplm.training.remote import UploadManager, build_upload_job

    remote_root = tmp_path / "remote"
    manager = UploadManager(RemoteStore(f"file://{remote_root}"), _FakeAccelerator())

    for step in (2, 4, 6, 8):
        local_dir = tmp_path / "local" / f"checkpoint-{step}"
        local_dir.mkdir(parents=True)
        (local_dir / "__0_0.distcp").write_text("shard")
        (local_dir / "rng_state_0.pt").write_text("rng")
        (local_dir / "trainer_state.json").write_text("{}")
        (local_dir / "config.yaml").write_text("cfg")

        job = build_upload_job(
            local_dir,
            _FakeAccelerator(),
            permanent=False,
            save_total_limit=1,
            keep_every_n_steps=4,
        )
        manager.submit(job)
        manager.drain(timeout=5.0)

    remote_names = sorted(p.name for p in remote_root.iterdir() if p.is_dir())
    # Rotation runs after every commit (mirroring the trainer's real cadence), not
    # once at the end over the full final set: checkpoint-6 survives because when it
    # committed, rotate(limit=1) only had to trim the rolling set (then {2, 6}) down
    # to 1, dropping 2 -- by the time checkpoint-8 (permanent) commits, the rolling
    # set is already just {6}, well within the limit, so nothing more is removed.
    # checkpoint-4 and checkpoint-8 are permanent (keep_every_n_steps=4 boundary).
    assert remote_names == ["checkpoint-4", "checkpoint-6", "checkpoint-8"]


def test_consult_remote_resume_candidate_noop_when_uri_unset(tmp_path: Path) -> None:
    """No ``remote_checkpoint_uri`` -> the helper is a pure passthrough (no fsspec touch)."""
    from oplm.training.trainer import _consult_remote_resume_candidate
    from tests.training.conftest import tiny_train_cfg

    output_dir = tmp_path / "output"
    cfg = tiny_train_cfg(output_dir, tmp_path / "unused.parquet")

    local = output_dir / "checkpoint-5"
    assert _consult_remote_resume_candidate(local, output_dir, cfg, status=None) == local
    assert _consult_remote_resume_candidate(None, output_dir, cfg, status=None) is None


def test_consult_remote_resume_candidate_prefers_higher_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Remote wins (and downloads) only when its step exceeds the local candidate's.

    Patches ``validate_checkpoint_for_resume`` to a no-op: this test isolates the
    local-vs-remote step comparison, not pre-load validation of a real DCP checkpoint
    (covered elsewhere, e.g. ``tests/training/test_commit_protocol.py`` and the
    real-``Trainer`` e2e test in this file) -- the fake remote checkpoint here has no
    real ``.metadata``/DCP shards to validate.
    """
    import oplm.training.checkpoint as checkpoint_module
    from oplm.training.trainer import _consult_remote_resume_candidate
    from tests.training.conftest import tiny_train_cfg

    monkeypatch.setattr(checkpoint_module, "validate_checkpoint_for_resume", lambda *a, **k: None)

    remote_root = tmp_path / "remote"
    remote_uri = f"file://{remote_root}"
    store = RemoteStore(remote_uri)

    local_dir = tmp_path / "local" / "checkpoint-100"
    relpaths = _write_local_checkpoint(local_dir, {"a.txt": "x"})
    store.upload_checkpoint(local_dir, files=relpaths, permanent=False, write_manifest=True)

    output_dir = tmp_path / "output"
    cfg = tiny_train_cfg(output_dir, tmp_path / "unused.parquet", remote_checkpoint_uri=remote_uri)

    # No local candidate at all: the remote checkpoint (step 100) is downloaded and wins.
    result = _consult_remote_resume_candidate(None, output_dir, cfg, status=None)
    assert result == output_dir / "checkpoint-100"
    assert result.is_dir()

    # A higher-step local candidate beats the (lower-step) remote one: no download.
    higher_local = output_dir / "checkpoint-200"
    higher_local.mkdir(parents=True)
    result2 = _consult_remote_resume_candidate(higher_local, output_dir, cfg, status=None)
    assert result2 == higher_local


@pytest.mark.slow
def test_unset_remote_uri_is_zero_behavior_change(training_parquet: Path, tmp_path: Path) -> None:
    """``remote_checkpoint_uri=None`` never builds an upload manager or touches remote.py."""
    from oplm.training.trainer import Trainer
    from tests.training.conftest import tiny_train_cfg

    cfg = tiny_train_cfg(tmp_path, training_parquet, max_steps=2, batch_size=4, save_every=2)
    trainer = Trainer(cfg, callbacks=[])
    assert trainer._remote_upload_manager is None

    trainer.train()
    assert (tmp_path / "checkpoint-2").is_dir()


@pytest.mark.slow
def test_remote_e2e_upload_wipe_and_auto_resume(training_parquet: Path, tmp_path: Path) -> None:
    """A real pilot run mirrors checkpoints remotely; a wiped local dir recovers from it.

    Drives a real ``Trainer`` with ``train.remote_checkpoint_uri`` set to a ``file://``
    store: the periodic (async) checkpoints get mirrored via the background
    ``UploadManager`` and the natural end-of-training drain (Task 4.2) guarantees the
    final checkpoint's mirror has landed by the time ``train()`` returns. The local
    ``output_dir`` is then wiped entirely (simulating NVMe loss on a requeue to a
    fresh node) and a brand-new ``Trainer`` with ``auto_resume=True`` recovers purely
    from the remote mirror -- downloading it to ``output_dir`` (becoming the local
    committed copy) before resuming.
    """
    from oplm.training.trainer import Trainer
    from tests.training.conftest import tiny_train_cfg

    run_dir = tmp_path / "run"
    remote_root = tmp_path / "remote"
    remote_uri = f"file://{remote_root}"

    cfg = tiny_train_cfg(
        run_dir,
        training_parquet,
        max_steps=6,
        batch_size=4,
        save_every=2,
        save_total_limit=3,
        save_final=False,
        remote_checkpoint_uri=remote_uri,
    )
    Trainer(cfg, callbacks=[]).train()

    store = RemoteStore(remote_uri)
    result = store.latest_committed()
    assert result is not None
    name, manifest = result
    assert name == "checkpoint-6"
    assert "trainer_state.json" in manifest["files"]
    assert "config.yaml" in manifest["files"]
    assert any(f.startswith("hf/") for f in manifest["files"])
    assert any(f.endswith(".distcp") for f in manifest["files"])

    # Simulate NVMe loss on requeue: the local output_dir is gone entirely.
    shutil.rmtree(run_dir)

    resume_cfg = tiny_train_cfg(
        run_dir,
        training_parquet,
        max_steps=8,
        batch_size=4,
        save_every=2,
        save_total_limit=3,
        save_final=False,
        auto_resume=True,
        remote_checkpoint_uri=remote_uri,
    )
    assert resume_cfg.train.resume_from is None
    resumed = Trainer(resume_cfg, callbacks=[])
    assert resumed.global_step == 6
    assert (run_dir / "checkpoint-6").is_dir()  # downloaded, now the local committed copy

    resumed.train()
    assert resumed.global_step == 8
