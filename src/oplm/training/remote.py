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

Task 4.1 built :class:`RemoteStore` as pure storage logic. Task 4.2 adds the
trainer-facing wiring on top of it, kept in this module so ``trainer.py``'s own
integration stays thin (per-checkpoint calls, no threading/collective details):

- :func:`build_upload_group` -- a dedicated GLOO subgroup of this run's node
  leaders (``accelerator.local_process_index == 0``), created once at trainer
  init. The background upload thread barriers on this group, never on the
  trainer's own (typically NCCL) default process group.
- :func:`build_upload_job` -- partitions a committed checkpoint directory's files
  into the files *this node's* ranks wrote (DCP shard files are named
  ``__<rank>_<n>.distcp``; each rank also has its own ``rng_state_<rank>.pt``) plus,
  on global rank 0 only, the shared artifacts (``.metadata``, ``trainer_state.json``,
  ``config.yaml``, ``scaler.pt``, ``KEEP``, ``hf/``).
- :class:`UploadManager` -- serializes uploads to a single background daemon
  thread (one in flight; a new commit while uploading replaces at most one queued
  job, dropping the superseded one), barriers node leaders once their local upload
  lands, and -- on the global leader only -- finalizes and rotates the remote
  manifest.
"""

from __future__ import annotations

import json
import logging
import shutil
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fsspec.core import url_to_fs

if TYPE_CHECKING:
    from accelerate import Accelerator
    from fsspec.spec import AbstractFileSystem
    from torch.distributed import ProcessGroup

logger = logging.getLogger(__name__)

_CHECKPOINT_PREFIX = "checkpoint-"
_TMP_SUFFIX = ".tmp"
_MANIFEST_NAME = "manifest.json"

# Mirrors checkpoint.py's own (private) per-rank RNG sidecar naming -- duplicated
# rather than imported since it's an implementation detail of that module's file
# layout that this one needs to recognize, not a shared public constant.
_RNG_SIDECAR_PREFIX = "rng_state_"

# Shared artifacts uploaded once, by global rank 0 only, alongside the per-node DCP
# shard files (see build_upload_job). "KEEP" is included so a downloaded checkpoint
# that was locally marked permanent (checkpoint.mark_permanent) stays exempt from
# local rotation after a remote-recovered resume, too.
_SHARED_ARTIFACT_NAMES = (".metadata", "trainer_state.json", "config.yaml", "scaler.pt", "KEEP")


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


def build_upload_group(accelerator: Accelerator) -> ProcessGroup | None:
    """Create a dedicated GLOO subgroup of this run's node leaders (local rank 0).

    Must be called collectively, exactly once, by **every** rank in the default
    process group -- ``torch.distributed.new_group`` requires this even for ranks
    that end up excluded from the returned group; a rank that skips this call
    (e.g. behind an ``if remote_checkpoint_uri is not None`` guard that isn't
    identical on every rank) hangs every other rank at this line forever. The
    caller (``Trainer.__init__``) is responsible for reaching it identically on
    every rank whenever a remote URI is configured.

    This group exists so :class:`UploadManager`'s background-thread barrier never
    touches the trainer's own default process group -- typically NCCL, which is not
    safe to drive collectives on from a thread other than the one owning the
    training loop. GLOO has no such restriction.

    Args:
        accelerator: The trainer's Accelerator, called right after it is
            constructed and before any checkpoint save.

    Returns:
        The new GLOO ``ProcessGroup`` for a node-leader rank (``local_process_index
        == 0``); a non-member sentinel (``torch.distributed.GroupMember.
        NON_GROUP_MEMBER``) for every other rank -- callers must never construct an
        :class:`UploadManager` from a non-leader rank's return value, only leader
        ranks do; or ``None`` when there is no process group at all (a
        single-process run), which callers must treat as "no barrier needed".
    """
    import torch.distributed as dist

    if accelerator.num_processes <= 1 or not dist.is_initialized():
        return None

    local_ranks: list[int | None] = [None] * accelerator.num_processes
    dist.all_gather_object(local_ranks, accelerator.local_process_index)
    leader_ranks = sorted(rank for rank, local_rank in enumerate(local_ranks) if local_rank == 0)
    return dist.new_group(ranks=leader_ranks, backend="gloo")


@dataclass(frozen=True)
class UploadJob:
    """One committed checkpoint's upload request, already partitioned by writer.

    Built by :func:`build_upload_job` from a committed local checkpoint directory
    and handed to :meth:`UploadManager.submit`.

    Attributes:
        local_dir: The committed local checkpoint directory (``checkpoint-<step>/``);
            only its name is used remotely (see ``RemoteStore.upload_checkpoint``).
        files: Paths relative to ``local_dir`` that *this node's* ranks wrote (DCP
            shard files + RNG sidecars). Every node's leader uploads this list.
        shared_files: Paths relative to ``local_dir`` for the artifacts written once,
            by the main process, regardless of which node it lives on (``.metadata``,
            ``trainer_state.json``, ``config.yaml``, ``scaler.pt``, ``KEEP``, ``hf/``).
            ``None`` on every rank except the global leader, which alone uploads them.
        permanent: Whether this checkpoint is exempt from ``save_total_limit``
            rotation (mirrors the local ``keep_every_n_steps``/``keep_every_n_hours``
            decision already made by the trainer at save time).
        save_total_limit: Passed through to ``RemoteStore.rotate`` after finalize.
        keep_every_n_steps: Passed through to ``RemoteStore.rotate`` after finalize.
    """

    local_dir: Path
    files: list[Path]
    shared_files: list[Path] | None
    permanent: bool
    save_total_limit: int
    keep_every_n_steps: int | None


def _local_world_size(accelerator: Accelerator) -> int:
    """Best-effort per-node rank count, for partitioning DCP shard files by rank.

    Assumes ranks are laid out contiguously per node (rank ``0..k-1`` on node 0,
    ``k..2k-1`` on node 1, ...) -- true for the torchrun/Slurm homogeneous
    allocations this codebase's Slurm generator produces (docs/SLURM.md), but not a
    universal guarantee for every possible launcher. Falls back to
    ``accelerator.num_processes`` (i.e. "everyone is on one node") when the
    environment doesn't expose ``LOCAL_WORLD_SIZE`` (torchrun/Accelerate set it; a
    launcher that doesn't degrades to treating the whole job as one node, which is
    simply wrong -- not silently incorrect in a way that drops files -- for a
    multi-node job under such a launcher: see ``build_upload_job``'s docstring).
    """
    import os

    raw = os.environ.get("LOCAL_WORLD_SIZE")
    if raw is not None and raw.isdigit() and int(raw) > 0:
        return int(raw)
    return accelerator.num_processes


def build_upload_job(
    checkpoint_dir: Path,
    accelerator: Accelerator,
    *,
    permanent: bool,
    save_total_limit: int,
    keep_every_n_steps: int | None,
) -> UploadJob:
    """Partition a committed checkpoint dir into this node's files + rank 0's shared files.

    Single-process (``accelerator.num_processes <= 1``) degenerates to "upload
    everything" -- there is only one node and nothing to partition.

    Otherwise: this node's global-rank range is computed from
    ``accelerator.process_index`` and :func:`_local_world_size` (see its docstring
    for the contiguous-layout assumption), and every rank in that range contributes
    its DCP shard files (``__<rank>_*.distcp``) and its ``rng_state_<rank>.pt``
    sidecar, if present. **Caveat:** if the contiguous-layout assumption is ever
    wrong for a given launcher, this only ever *undercounts* a node's files (ranks
    it doesn't think belong to it are simply never globbed for), never
    double-uploads another node's files -- the failure mode is a checkpoint missing
    shards remotely (caught the same way an incomplete manifest always is, see
    ``RemoteStore.finalize``'s hard precondition), not silent corruption.

    The global leader (``accelerator.is_main_process``) additionally collects the
    shared, once-only artifacts: ``.metadata``, ``trainer_state.json``,
    ``config.yaml``, ``scaler.pt``, ``KEEP`` (whichever of these exist -- ``scaler.pt``
    and ``KEEP`` are conditional even locally), and every file under ``hf/``.

    Args:
        checkpoint_dir: The committed local checkpoint directory.
        accelerator: The trainer's Accelerator.
        permanent: Recorded on the returned job; see :class:`UploadJob`.
        save_total_limit: Recorded on the returned job; see :class:`UploadJob`.
        keep_every_n_steps: Recorded on the returned job; see :class:`UploadJob`.

    Returns:
        The partitioned :class:`UploadJob` for this rank to hand to
        :meth:`UploadManager.submit`.
    """
    if accelerator.num_processes <= 1:
        files = sorted(
            p.relative_to(checkpoint_dir) for p in checkpoint_dir.rglob("*") if p.is_file()
        )
        return UploadJob(
            local_dir=checkpoint_dir,
            files=files,
            shared_files=None,
            permanent=permanent,
            save_total_limit=save_total_limit,
            keep_every_n_steps=keep_every_n_steps,
        )

    local_world_size = _local_world_size(accelerator)
    node_index = accelerator.process_index // local_world_size
    node_start = node_index * local_world_size
    node_end = min(node_start + local_world_size, accelerator.num_processes)

    node_files: list[Path] = []
    for rank in range(node_start, node_end):
        node_files.extend(
            sorted(p.relative_to(checkpoint_dir) for p in checkpoint_dir.glob(f"__{rank}_*.distcp"))
        )
        sidecar = checkpoint_dir / f"{_RNG_SIDECAR_PREFIX}{rank}.pt"
        if sidecar.is_file():
            node_files.append(sidecar.relative_to(checkpoint_dir))

    shared_files: list[Path] | None = None
    if accelerator.is_main_process:
        shared_files = []
        for name in _SHARED_ARTIFACT_NAMES:
            candidate = checkpoint_dir / name
            if candidate.is_file():
                shared_files.append(Path(name))
        hf_dir = checkpoint_dir / "hf"
        if hf_dir.is_dir():
            shared_files.extend(
                sorted(p.relative_to(checkpoint_dir) for p in hf_dir.rglob("*") if p.is_file())
            )

    return UploadJob(
        local_dir=checkpoint_dir,
        files=node_files,
        shared_files=shared_files,
        permanent=permanent,
        save_total_limit=save_total_limit,
        keep_every_n_steps=keep_every_n_steps,
    )


class UploadManager:
    """Background, serialized remote upload of committed checkpoints.

    One upload in flight at a time. A checkpoint submitted while another is
    uploading is queued (at most one slot); if a third arrives before the queued
    one's turn, the queued one is dropped (logged) and replaced -- the newer
    checkpoint always supersedes an older, not-yet-started one. This bounds the
    background thread to ever uploading at most two checkpoints "behind" the
    trainer's own commit cadence: the one currently in flight, plus the one queued.

    Each call to :meth:`submit` uploads ``job.files`` (this node's own shard files)
    and, when ``job.shared_files`` is not ``None`` (the global leader only), the
    shared artifacts too -- then, if this manager was built with an
    ``upload_group`` (multi-node), barriers on it so no leader finalizes before
    every node's upload has landed. Only the global leader
    (``accelerator.is_main_process`` at construction time) then calls
    ``store.finalize`` and ``store.rotate``. This is exactly the ordering
    ``RemoteStore.finalize``'s hard precondition requires: finalize only after
    every writer's files are confirmed uploaded.

    Single-process degradation: ``upload_group=None`` and (necessarily, since
    there is only one process) ``is_global_leader=True`` -- upload, skip the
    barrier entirely, finalize, rotate; exactly Task 4.1's own direct
    ``upload_checkpoint(..., write_manifest=True)`` flow.
    """

    def __init__(
        self,
        store: RemoteStore,
        accelerator: Accelerator,
        *,
        upload_group: ProcessGroup | None = None,
    ) -> None:
        """Build an upload manager for one rank (only ever constructed on node leaders).

        Args:
            store: The remote mirror to upload to and finalize/rotate on.
            accelerator: The trainer's Accelerator -- only ``is_main_process`` is
                read (at construction time, not per-upload), to decide whether this
                rank is the *global* leader that finalizes and rotates.
            upload_group: The GLOO subgroup from :func:`build_upload_group`, or
                ``None`` for a single-process run (no barrier needed).
        """
        self._store = store
        self._is_global_leader = accelerator.is_main_process
        self._upload_group = upload_group
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._pending: UploadJob | None = None

    def submit(self, job: UploadJob) -> None:
        """Enqueue ``job`` for background upload (non-blocking).

        Starts uploading immediately if idle; otherwise queues ``job`` behind the
        in-flight upload, dropping (and logging) whatever was previously queued.
        """
        with self._lock:
            if self._thread is not None:
                if self._pending is not None:
                    logger.warning(
                        "Dropping queued checkpoint upload for %s; superseded by %s",
                        self._pending.local_dir.name,
                        job.local_dir.name,
                    )
                self._pending = job
                return
            thread = threading.Thread(
                target=self._run, args=(job,), daemon=True, name="oplm-checkpoint-upload"
            )
            self._thread = thread
        thread.start()

    def _run(self, job: UploadJob) -> None:
        """Upload ``job``, then chain to whatever was queued by the time it finished."""
        try:
            self._upload_one(job)
        except Exception:  # noqa: BLE001 -- a failed upload must never kill the trainer
            logger.exception("Background checkpoint upload of %s failed", job.local_dir)

        next_job: UploadJob | None
        next_thread: threading.Thread | None = None
        with self._lock:
            next_job = self._pending
            self._pending = None
            if next_job is not None:
                next_thread = threading.Thread(
                    target=self._run, args=(next_job,), daemon=True, name="oplm-checkpoint-upload"
                )
                self._thread = next_thread
            else:
                self._thread = None
        if next_thread is not None:
            next_thread.start()

    def _upload_one(self, job: UploadJob) -> None:
        """Upload ``job``'s files, barrier (if configured), then finalize + rotate.

        See :meth:`RemoteStore.finalize`'s hard precondition -- finalize only runs
        after the barrier confirms every node leader's upload for this exact job has
        landed (or immediately, single-process, where there is nothing to wait for).
        """
        self._store.upload_checkpoint(
            job.local_dir, files=job.files, permanent=job.permanent, write_manifest=False
        )
        if job.shared_files is not None:
            self._store.upload_checkpoint(
                job.local_dir, files=job.shared_files, permanent=job.permanent, write_manifest=False
            )

        if self._upload_group is not None:
            import torch.distributed as dist

            dist.barrier(group=self._upload_group)

        if self._is_global_leader:
            self._store.finalize(job.local_dir.name, permanent=job.permanent)
            self._store.rotate(job.save_total_limit, job.keep_every_n_steps)

    def drain(self, timeout: float = 600.0) -> None:
        """Block (bounded by ``timeout`` seconds total) until nothing is in flight.

        Waits across the *entire* chain -- the currently in-flight upload plus
        whatever was queued behind it, which may itself chain again -- not just a
        single ``Thread.join``, since a single join on the current thread would
        return the moment it finishes even if a queued job immediately takes its
        place. Logs (and returns) if ``timeout`` elapses with an upload still in
        flight -- the caller's local checkpoint already exists and is a valid
        resume target regardless of whether the remote mirror ever finishes.
        """
        deadline = time.monotonic() + timeout
        while True:
            with self._lock:
                thread = self._thread
            if thread is None:
                return
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                logger.error(
                    "Checkpoint upload still in flight after %.0fs timeout; proceeding "
                    "without waiting further (the local checkpoint is unaffected).",
                    timeout,
                )
                return
            thread.join(remaining)
