"""Single-host run ownership. Kernel locks disappear when a worker is killed.

New turns use new thread IDs. Busy runs reject concurrent input/approval; SIGINT
cancels execution without granting a replacement writer access prematurely.
"""

import fcntl
import inspect
import uuid
from functools import wraps

from market_analyst.runtime.workspace import ensure_thread_workspace


class RunBusy(RuntimeError):
    pass


def serialized_run(function):
    """Serialize every public workflow mutation on the shared workspace volume."""
    signature = inspect.signature(function)

    @wraps(function)
    def wrapped(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        thread_id = bound.arguments.get("thread_id") or str(uuid.uuid4())
        bound.arguments["thread_id"] = thread_id
        workspace = ensure_thread_workspace(thread_id)
        # ponytail: local kernel lock; multiple hosts require DB leases with fencing.
        with (workspace / ".run.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RunBusy("Run already has an active writer") from None
            try:
                return function(*bound.args, **bound.kwargs)
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    return wrapped
