"""fsspec-based remote checkpoint mirror (``s3://``, ``gs://``, ``file://``, ...).

:class:`RemoteStore` mirrors ``output_dir``'s on-disk checkpoint layout onto an
fsspec filesystem: ``<uri>/checkpoint-<step>/<files>``. It follows the exact same
committed-only discipline as :mod:`oplm.training.checkpoint`'s local tmp-dir +
rename protocol, adapted to object stores (which have no atomic rename across
prefixes): ``manifest.json`` is the remote commit marker and is always written
*last*, after every checkpoint file has landed. A ``checkpoint-<step>/`` directory
without a ``manifest.json`` is therefore uncommitted -- indistinguishable from a
torn upload -- and is invisible to :meth:`RemoteStore.latest_committed` and
:meth:`RemoteStore.rotate`.

This module is pure storage logic (Task 4.1); wiring it into the trainer's
background upload thread is Task 4.2.
"""

from __future__ import annotations

import json
import logging
import shutil
from typing import TYPE_CHECKING, Any

from fsspec.core import url_to_fs

if TYPE_CHECKING:
    from pathlib import Path

    from fsspec.spec import AbstractFileSystem

logger = logging.getLogger(__name__)

_CHECKPOINT_PREFIX = "checkpoint-"
_TMP_SUFFIX = ".tmp"
_MANIFEST_NAME = "manifest.json"


def _is_permanent(manifest: dict[str, Any], step: int, keep_every_n_steps: int | None) -> bool:
    """Mirror ``checkpoint._is_permanent_checkpoint``'s exemption rule for a manifest.

    A checkpoint is permanent if its manifest was written with ``"permanent": true``
    (the remote equivalent of the local ``KEEP`` marker -- see :func:`RemoteStore.finalize`)
    or if its step falls on the ``keep_every_n_steps`` boundary.
    """
    if bool(manifest.get("permanent", False)):
        return True
    return keep_every_n_steps is not None and step % keep_every_n_steps == 0


class RemoteStore:
    """Checkpoint mirror on an fsspec filesystem (``s3://``, ``gs://``, ``file://``...).

    Layout mirrors ``output_dir``: ``<uri>/checkpoint-<step>/<files>`` +
    ``manifest.json`` written LAST -- a ``checkpoint-<step>/`` without
    ``manifest.json`` is uncommitted and invisible.
    """

    def __init__(self, uri: str) -> None:
        """Resolve ``uri`` to an fsspec filesystem + root path.

        Args:
            uri: An fsspec URI (e.g. ``s3://bucket/prefix``, ``gs://bucket/prefix``,
                ``file:///abs/path``). Credentials, when needed, come from standard
                environment/config conventions for the target filesystem (e.g.
                ``AWS_*`` env vars for ``s3://``) -- never from this class.
        """
        self.uri = uri
        self._fs: AbstractFileSystem
        self._fs, self._root = url_to_fs(uri)

    def _join(self, *parts: str) -> str:
        """Join ``parts`` onto the store's root path."""
        return "/".join([self._root.rstrip("/"), *parts])

    def _read_manifest(self, manifest_path: str) -> dict[str, Any]:
        with self._fs.open(manifest_path, "r") as f:
            manifest: dict[str, Any] = json.load(f)
        return manifest

    def _list_committed(self) -> list[tuple[int, str, dict[str, Any]]]:
        """Return committed checkpoints (dir has a ``manifest.json``), ascending by step."""
        if not self._fs.exists(self._root):
            return []
        try:
            entries = self._fs.ls(self._root, detail=True)
        except FileNotFoundError:
            return []

        candidates: list[tuple[int, str, dict[str, Any]]] = []
        for entry in entries:
            if entry.get("type") != "directory":
                continue
            entry_name = entry["name"].rstrip("/").rsplit("/", 1)[-1]
            if not entry_name.startswith(_CHECKPOINT_PREFIX):
                continue
            suffix = entry_name.removeprefix(_CHECKPOINT_PREFIX)
            if not suffix.isdigit():
                continue
            manifest_path = self._join(entry_name, _MANIFEST_NAME)
            if not self._fs.exists(manifest_path):
                continue
            candidates.append((int(suffix), entry_name, self._read_manifest(manifest_path)))

        candidates.sort(key=lambda item: item[0])
        return candidates

    def upload_checkpoint(
        self,
        local_dir: Path,
        *,
        files: list[Path],
        permanent: bool,
        write_manifest: bool,
    ) -> None:
        """Upload ``files`` (relative to ``local_dir``) into ``<uri>/<local_dir.name>/``.

        Args:
            local_dir: Local committed checkpoint directory (``checkpoint-<step>/``);
                only its name is used remotely, its files are read from here.
            files: Paths relative to ``local_dir`` to upload.
            permanent: Passed through to :meth:`finalize` when ``write_manifest`` is
                ``True``; ignored otherwise (the caller is expected to call
                :meth:`finalize` itself later with the real value).
            write_manifest: When ``True``, calls :meth:`finalize` immediately after
                every file has uploaded, committing the checkpoint. When ``False``,
                the checkpoint remains uncommitted (invisible to
                :meth:`latest_committed`/:meth:`rotate`) until a later
                :meth:`finalize` call -- useful for a caller that wants to upload
                incrementally before deciding when to commit.
        """
        name = local_dir.name
        for rel_path in files:
            local_path = local_dir / rel_path
            remote_path = self._join(name, rel_path.as_posix())
            remote_parent = remote_path.rsplit("/", 1)[0]
            self._fs.makedirs(remote_parent, exist_ok=True)
            logger.info("Uploading checkpoint file %s -> %s", local_path, remote_path)
            self._fs.put_file(str(local_path), remote_path)

        if write_manifest:
            self.finalize(name, permanent=permanent)

    def finalize(self, name: str, *, permanent: bool) -> None:
        """Write ``manifest.json``, committing ``<uri>/<name>/`` as a resume candidate.

        Discovers every file already uploaded under ``<uri>/<name>/`` (excluding any
        prior ``manifest.json``) and records each one's remote size. Written LAST --
        after this call, and only after this call, is ``<uri>/<name>/`` visible to
        :meth:`latest_committed` and :meth:`rotate`. Calling ``finalize`` again on an
        already-committed checkpoint (e.g. after uploading additional files) is safe
        and simply rewrites the manifest from a fresh listing -- the prior
        ``manifest.json`` is excluded from that listing by name, so it is never
        recorded as one of its own checkpoint's files.

        **Hard precondition -- call only once every writer has confirmed that every
        file it is contributing has fully uploaded and is remotely visible.** In the
        trainer flow (Task 4.2), that means calling this only *after* the all-nodes
        upload barrier, once every rank has reported its uploads done -- never
        speculatively, and never from a single rank before the others have finished.
        This method has no way to know which files *should* eventually exist; it only
        records what it can currently list. **Violating this precondition does not
        raise.** A file uploaded a moment too late (or never) is simply absent from
        the manifest, silently, with no error here or anywhere else in this class --
        :meth:`download_checkpoint` only ever iterates the manifest's own entries, so
        it will happily "successfully" download an incomplete checkpoint missing one
        or more shards, with nothing to indicate the omission short of whatever fails
        much later when that checkpoint is actually loaded/resumed from.

        Args:
            name: The checkpoint directory name (e.g. ``"checkpoint-100"``).
            permanent: Recorded in the manifest; ``True`` exempts this checkpoint from
                :meth:`rotate`'s deletion regardless of ``keep_every_n_steps``.
        """
        remote_dir = self._join(name)
        prefix_len = len(remote_dir.rstrip("/")) + 1

        files: dict[str, int] = {}
        for path in self._fs.find(remote_dir):
            relpath = path[prefix_len:]
            if relpath == _MANIFEST_NAME:
                continue
            files[relpath] = self._fs.info(path)["size"]

        manifest = {"files": files, "permanent": permanent}
        manifest_path = self._join(name, _MANIFEST_NAME)
        logger.info(
            "Finalizing remote checkpoint %s (permanent=%s, %d file(s))",
            name,
            permanent,
            len(files),
        )
        with self._fs.open(manifest_path, "w") as f:
            json.dump(manifest, f)

    def latest_committed(self) -> tuple[str, dict[str, Any]] | None:
        """Return the highest-step committed checkpoint's ``(name, manifest)``, or ``None``.

        Mirrors :func:`oplm.training.checkpoint.latest_checkpoint`'s numeric-suffix
        ordering (never lexicographic, so ``checkpoint-10000`` outranks
        ``checkpoint-9000``), restricted to directories that carry a ``manifest.json``.
        """
        committed = self._list_committed()
        if not committed:
            return None
        _, name, manifest = committed[-1]
        return name, manifest

    def download_checkpoint(self, name: str, dest: Path) -> Path:
        """Download committed checkpoint ``name`` into ``dest/<name>``.

        Downloads into a local ``dest/<name>.tmp`` staging directory first, verifying
        each downloaded file's size against the manifest, then renames it onto
        ``dest/<name>`` only once every file has verified -- reusing the same local
        tmp-dir + rename commit convention as
        :func:`oplm.training.checkpoint.save_checkpoint`, so a torn download is never
        left at the committed name and is never mistaken for a resume candidate.

        Args:
            name: The checkpoint directory name to download (e.g. ``"checkpoint-100"``).
            dest: Local directory to download into (created if missing).

        Returns:
            The committed local path, ``dest / name``.

        Raises:
            RuntimeError: A downloaded file's size disagrees with the manifest -- i.e.
                the file exists remotely but came down corrupted/truncated/wrong-sized.
            OSError: A file the manifest lists does not exist remotely at all (e.g. it
                was deleted out from under this store after ``finalize`` ran). This is
                *not* translated into ``RuntimeError`` -- it surfaces as whatever
                native exception the underlying fsspec filesystem's ``get_file``
                raises for a missing path (``FileNotFoundError``, a subclass of
                ``OSError``, on local/S3-like filesystems).
        """
        manifest = self._read_manifest(self._join(name, _MANIFEST_NAME))

        tmp_dir = dest / f"{name}{_TMP_SUFFIX}"
        final_dir = dest / name
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir(parents=True, exist_ok=True)

        for relpath, expected_size in manifest["files"].items():
            remote_path = self._join(name, relpath)
            local_path = tmp_dir / relpath
            local_path.parent.mkdir(parents=True, exist_ok=True)
            logger.info("Downloading checkpoint file %s -> %s", remote_path, local_path)
            self._fs.get_file(remote_path, str(local_path))

            actual_size = local_path.stat().st_size
            if actual_size != expected_size:
                raise RuntimeError(
                    f"Downloaded file {relpath!r} for checkpoint {name!r} has size "
                    f"{actual_size}, expected {expected_size} per manifest.json "
                    f"(remote path: {remote_path})"
                )

        if final_dir.exists():
            shutil.rmtree(final_dir)
        tmp_dir.rename(final_dir)
        logger.info("Committed downloaded checkpoint to %s", final_dir)
        return final_dir

    def rotate(self, save_total_limit: int, keep_every_n_steps: int | None) -> None:
        """Delete non-permanent committed checkpoints beyond ``save_total_limit``.

        Mirrors :func:`oplm.training.checkpoint._rotate_checkpoints` exactly:
        permanent checkpoints (manifest ``"permanent": true``, or a step on the
        ``keep_every_n_steps`` boundary -- see :func:`_is_permanent`) are excluded
        from both the rolling count and deletion; only the oldest rolling checkpoints
        beyond the newest ``save_total_limit`` are removed. Uncommitted (no-manifest)
        directories are never counted or deleted here.

        Args:
            save_total_limit: Maximum number of rolling checkpoints to keep. A
                non-positive value disables rotation entirely (no-op), matching the
                local rule.
            keep_every_n_steps: When set, checkpoints whose step is a multiple of this
                value are permanent and excluded from rotation.
        """
        if save_total_limit <= 0:
            return

        committed = self._list_committed()
        rolling = [
            (step, name)
            for step, name, manifest in committed
            if not _is_permanent(manifest, step, keep_every_n_steps)
        ]

        while len(rolling) > save_total_limit:
            _, name = rolling.pop(0)
            remote_dir = self._join(name)
            logger.info("Removing old remote checkpoint: %s", remote_dir)
            self._fs.rm(remote_dir, recursive=True)
