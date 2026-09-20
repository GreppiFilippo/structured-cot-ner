from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class HFCheckpointSync:
    """Backs up the local output shard to a Hugging Face dataset repo, and
    can recover it back if the local file is missing (e.g. a fresh node
    after the previous one died mid-run).

    Not a per-row push — over tens of thousands of rows that would be slow
    and likely to hit HF Hub rate limits. Instead, pushes the whole
    accumulated file every `push_every` new results, plus a forced push at
    the end of the run and on SIGTERM (see main.py).
    """

    def __init__(self, repo_id: str, token: Optional[str], push_every: int = 200) -> None:
        self._repo_id = repo_id
        self._token = token
        self._push_every = push_every
        self._since_last_push = 0
        self._api = None
        if repo_id:
            try:
                from huggingface_hub import HfApi

                self._api = HfApi(token=token)
                self._ensure_repo_exists()
            except Exception as error:
                # A transient network/auth issue here must not take down the
                # whole job before it has even started generating — fall
                # back to local-file-only instead of crashing.
                logger.warning("Could not initialize remote backup (%s) — continuing with local file only.", error)
                self._api = None

    @property
    def enabled(self) -> bool:
        return self._api is not None

    def _ensure_repo_exists(self) -> None:
        from huggingface_hub import create_repo

        # private by default: this is an intermediate checkpoint, not the
        # curated dataset you'd eventually want to publish
        create_repo(self._repo_id, repo_type="dataset", token=self._token, exist_ok=True, private=True)

    def try_recover(self, local_path: Path) -> None:
        """Pulls the last known checkpoint from the Hub if no local shard
        exists yet. Never raises — a missing remote checkpoint just means
        starting fresh, which is the normal case on the very first run."""
        if local_path.exists() or not self.enabled:
            return
        try:
            from huggingface_hub import hf_hub_download

            remote_file = hf_hub_download(
                repo_id=self._repo_id, repo_type="dataset", filename=local_path.name, token=self._token
            )
            local_path.parent.mkdir(parents=True, exist_ok=True)
            Path(remote_file).replace(local_path)
            logger.info("Recovered checkpoint from %s: %s", self._repo_id, local_path)
        except Exception as error:
            logger.info("No remote checkpoint to recover (%s) — starting fresh.", error)

    def should_push(self) -> bool:
        """Call once per result appended to the local shard. Returns True
        every `push_every` calls (and resets the counter) — cheap, in-memory
        bookkeeping only, no I/O. Separated from push_now() so the caller
        can decide *how* to run the actual (blocking) upload — see main.py,
        which offloads it to a thread instead of calling it inline and
        freezing the async event loop for the duration of the upload."""
        if not self.enabled:
            return False
        self._since_last_push += 1
        if self._since_last_push >= self._push_every:
            self._since_last_push = 0
            return True
        return False

    def push_now(self, local_path: Path) -> None:
        """Force a push regardless of the counter. Synchronous/blocking —
        call via asyncio.to_thread from async code (see main.py) unless
        you're in a context (like a signal handler) where blocking briefly
        before the process exits is fine."""
        if not self.enabled or not local_path.exists():
            return
        try:
            self._api.upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=local_path.name,
                repo_id=self._repo_id,
                repo_type="dataset",
                commit_message=f"checkpoint: {local_path.name}",
            )
            logger.info("Pushed checkpoint to %s (%s)", self._repo_id, local_path.name)
        except Exception as error:  # a failed backup must never crash the actual run
            logger.warning("Checkpoint push to HF Hub failed, will retry later: %s", error)
