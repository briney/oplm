"""Tests for :class:`oplm.training.remote.RemoteStore` (Task 4.1).

Exercises the fsspec-based checkpoint mirror over a ``file://`` filesystem (fast,
no network): upload+finalize commits a checkpoint via the manifest-last discipline,
an uncommitted (no-manifest) checkpoint directory is invisible to discovery, a
downloaded file whose size disagrees with the manifest raises, and rotation mirrors
``oplm.training.checkpoint._rotate_checkpoints``'s permanent/rolling semantics
exactly. A slow, optional moto-based ``s3://`` smoke test is included but skips
gracefully when ``moto`` is not installed.
"""

from __future__ import annotations

import json
from pathlib import Path

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
