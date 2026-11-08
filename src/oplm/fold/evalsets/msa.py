"""ColabFold MSA server client and MSA preprocessing for the coevolution eval.

The client follows ColabFold's own: POST a ticket, poll it, download the tar.gz of a3m files.
"""

from __future__ import annotations

import io
import json
import tarfile
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable

import torch
from torch import Tensor

from oplm import __version__
from oplm.eval.metrics.categorical_jacobian import CANONICAL_AMINO_ACIDS

__all__ = [
    "PUBLIC_COLABFOLD_HOST",
    "Fetch",
    "encode_msa",
    "fetch_msa",
    "filter_gap_columns",
    "neff",
    "parse_a3m",
]

PUBLIC_COLABFOLD_HOST = "https://api.colabfold.com"

# (url, POST body or None for GET) -> response body.
Fetch = Callable[[str, bytes | None], bytes]

_GAP = len(CANONICAL_AMINO_ACIDS)
_RETRY = {"RATELIMIT", "UNKNOWN", "PENDING", "RUNNING", "MAINTENANCE"}
_MAX_WAIT = 600.0
_MAX_ATTEMPTS = 8
_TRANSIENT_HTTP = {429, 500, 502, 503, 504}


def _urllib_fetch(url: str, body: bytes | None) -> bytes:
    request = urllib.request.Request(url, data=body, headers={"User-Agent": f"oplm/{__version__}"})
    with urllib.request.urlopen(request, timeout=300) as response:
        return response.read()


def fetch_msa(
    sequence: str,
    host: str,
    *,
    fetch: Fetch | None = None,
    mode: str = "env",
    poll_seconds: float = 5.0,
    sleep: Callable[[float], None] = time.sleep,
) -> str:
    """The combined a3m (UniRef then environmental hits, query once) for ``sequence``.

    Raises:
        RuntimeError: The server reports ``ERROR`` for this sequence.
    """
    raw_fetch = fetch or _urllib_fetch
    host = host.rstrip("/")
    body = urllib.parse.urlencode({"q": f">101\n{sequence}", "mode": mode}).encode()

    def fetch(url: str, data: bytes | None) -> bytes:
        """``raw_fetch`` retried with back-off on 429/5xx, URLError and timeouts."""
        delay = poll_seconds
        for attempt in range(_MAX_ATTEMPTS):
            try:
                return raw_fetch(url, data)
            except urllib.error.HTTPError as exc:
                if exc.code not in _TRANSIENT_HTTP or attempt == _MAX_ATTEMPTS - 1:
                    raise
            except (urllib.error.URLError, TimeoutError):
                if attempt == _MAX_ATTEMPTS - 1:
                    raise
            sleep(delay)
            delay = min(delay * 2, _MAX_WAIT)
        raise AssertionError("unreachable")

    def submit() -> dict:
        return json.loads(fetch(f"{host}/ticket/msa", body))

    wait = poll_seconds
    reply = submit()
    while True:
        if reply["status"] == "ERROR":
            raise RuntimeError(f"ColabFold MSA server returned ERROR for sequence {sequence}")
        if reply["status"] not in _RETRY:
            break
        ratelimited = reply["status"] == "RATELIMIT"
        if not ratelimited:
            wait = poll_seconds
        sleep(wait)
        if ratelimited:
            wait = min(wait * 2, _MAX_WAIT)
        if ratelimited or "id" not in reply:  # no ticket yet: resubmit, as ColabFold does
            reply = submit()
        else:
            reply = json.loads(fetch(f"{host}/ticket/{reply['id']}", None))
    ticket = reply["id"]
    tar_bytes = fetch(f"{host}/result/download/{ticket}", None)
    texts: list[str] = []
    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:gz") as tf:
        for name in ("uniref.a3m", "bfd.mgnify30.metaeuk30.smag30.a3m"):
            member = tf.extractfile(name)
            if member is not None:
                texts.append(member.read().decode().replace("\x00", ""))
    combined = texts[0]
    for text in texts[1:]:
        combined += text.split("\n", 2)[2]  # drop the repeated query record
    return combined


def parse_a3m(text: str) -> list[str]:
    """Aligned rows of an a3m (insertions, the lowercase letters, removed), query first."""
    rows: list[str] = []
    for line in text.splitlines():
        if line.startswith(">"):
            rows.append("")
        elif rows:
            rows[-1] += "".join(c for c in line.strip() if not c.islower())
    return rows


_LOOKUP = torch.full((256,), _GAP, dtype=torch.int8)
for _i, _aa in enumerate(CANONICAL_AMINO_ACIDS):
    _LOOKUP[ord(_aa)] = _i


def encode_msa(rows: list[str], max_seqs: int = 20_000) -> Tensor:
    """``(N, L)`` int8 in 0..20 (gap and non-canonical = 20) from the first ``max_seqs`` rows."""
    codes = [
        torch.frombuffer(bytearray(r.encode("latin-1")), dtype=torch.uint8) for r in rows[:max_seqs]
    ]
    return _LOOKUP[torch.stack(codes).long()]


def filter_gap_columns(msa: Tensor, max_gap_fraction: float = 0.25) -> tuple[Tensor, Tensor]:
    """The MSA kept to columns gapped in <= ``max_gap_fraction`` of rows, and their indices."""
    keep = ((msa == _GAP).float().mean(dim=0) <= max_gap_fraction).nonzero().squeeze(1)
    return msa[:, keep], keep


def neff(msa: Tensor, identity: float = 0.8, chunk: int = 4096) -> tuple[float, Tensor]:
    """Effective sequence count and per-sequence weights (1 / number of rows at >= ``identity``)."""
    n, length = msa.shape
    onehot = torch.nn.functional.one_hot(msa.long(), _GAP + 1).reshape(n, -1).float()
    weights = torch.empty(n)
    for start in range(0, n, chunk):
        similar = onehot[start : start + chunk] @ onehot.T / length >= identity
        weights[start : start + chunk] = 1.0 / similar.sum(dim=1)
    return float(weights.sum()), weights
