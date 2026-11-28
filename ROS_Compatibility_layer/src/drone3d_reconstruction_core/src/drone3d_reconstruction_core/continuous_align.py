"""Window alignment for the continuous session (S2 step 5): the G4 map backend process.

Windows are placed in the background: each published window gets a pending alignment, and
the session-owned G4 process re-solves the map from the formal records. The interface is
the four things the session needs:

- :meth:`align` - the alignment written into a window's formal record;
- :meth:`on_window_published` - after a window's record is committed: start the G4 process,
  or restart it when it died (journaled), and report a start failure loudly;
- :meth:`status` / :meth:`map_status` - for diagnostics and for the drain before STOP;
- :meth:`detach` - hand the process to the caller to close outside the session lock.

All methods except the returned process's ``close`` are called with the session lock held;
the aligner has no lock of its own.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from .g4.backend_process import BACKEND_G4
from .odom_local import COORDINATE_MODEL_LOCAL_UNALIGNED


class G4WindowAligner:
    def __init__(self, process_factory: Callable[..., Any]) -> None:
        # The factory is looked up through the session module at call time, so tests that
        # patch continuous_session.G4BackendProcess keep working.
        self._process_factory = process_factory
        self.process = None
        self.start_error = ""

    def reset(self) -> None:
        """A new session starts with no recorded start failure."""
        self.start_error = ""

    @staticmethod
    def align() -> dict[str, Any]:
        """G4 places windows in the background; each window waits for a map revision."""
        return {
            "coordinate_state": COORDINATE_MODEL_LOCAL_UNALIGNED,
            "map_id": "",
            "scale": None,
            "rotation": None,
            "translation": None,
            "residual_rms": None,
            "correspondence_count": 0,
            "reference_frame_id": "",
            "unreliable": True,
            "reason": "waiting_for_G4_map_revision",
            "method": BACKEND_G4,
            "pending": True,
            "quality_passed": False,
            "geometry_accepted": False,
        }

    def on_window_published(
        self,
        records_root: Path,
        session_id: str,
        window_index: int,
        profile: Callable[[], dict],
        write_journal: Callable[[dict], None],
    ) -> str:
        """Make sure the G4 process runs after a window was committed.

        A map backend that cannot start, or that died, does not stop capture: the recorded
        windows can be placed again after landing. It must be loud instead (diagnostics
        ERROR, journal), because the window itself still succeeds. Returns the start error
        ('' when the process runs) for the session's last_error.
        """
        try:
            if self.process is not None and self.process.process.poll() is not None:
                exited = self.process.status()
                write_journal({
                    "event": "g4_backend_exited",
                    "session_id": session_id,
                    "window_index": window_index,
                    "exit_code": exited.get("exit_code"),
                    "error": exited.get("error", ""),
                })
                self.process.close()
                self.process = None
            if self.process is None:
                self.process = self._process_factory(records_root, session_id, profile=profile())
                self.start_error = ""
        except Exception as error:
            self.start_error = f"G4 background start failed: {type(error).__name__}: {error}"
            write_journal({
                "event": "g4_backend_start_failed",
                "session_id": session_id,
                "window_index": window_index,
                "error": self.start_error,
            })
        return self.start_error

    def status(self) -> dict[str, Any]:
        """Map backend status for diagnostics; a failed start reports as an error, not as "waiting"."""
        if self.process is not None:
            return self.process.status()
        if self.start_error:
            return {"state": "error", "error": self.start_error}
        return {}

    def map_status(self) -> dict[str, Any] | None:
        """The running process's status, or None when there is no process."""
        return None if self.process is None else self.process.status()

    def detach(self):
        """Take the process out; the caller closes it outside the session lock."""
        process, self.process = self.process, None
        return process


__all__ = ["G4WindowAligner"]
