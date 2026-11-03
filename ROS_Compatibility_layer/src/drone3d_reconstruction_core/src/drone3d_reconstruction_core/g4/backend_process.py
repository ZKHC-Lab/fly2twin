"""Background G4 process owned by ContinuousSession, with no extra ROS node."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

BACKEND_G4 = "global_g4"


def output_root(records_root, session_id):
    return Path(records_root).parent / "g4" / session_id


def profile_path(root):
    return Path(root) / "backend_profile.json"


class G4BackendProcess:
    def __init__(self, records_root, session_id, profile=None):
        self.root = output_root(records_root, session_id)
        self.root.mkdir(parents=True, exist_ok=True)
        self.log = (self.root / "runtime.log").open("ab")
        env = dict(os.environ)
        for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
            env[key] = "1"
        command = [sys.executable, "-m", __name__, "--records-root", str(records_root),
                   "--session", session_id, "--parent-pid", str(os.getpid())]
        if profile is not None:
            # The child must not re-derive the profile from its environment:
            # hand it the exact mapping this session decided on.
            target = profile_path(self.root)
            temporary = target.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(profile, sort_keys=True, indent=2, allow_nan=False) + "\n")
            os.replace(temporary, target)
            command += ["--profile", str(target)]
        try:
            self.process = subprocess.Popen(
                command, stdout=self.log, stderr=subprocess.STDOUT,
                env=env, start_new_session=True,
            )
        except Exception:
            self.log.close()
            raise

    def status(self):
        try:
            data = json.loads((self.root / "status.json").read_text())
        except (OSError, ValueError):
            data = {"state": "starting"}
        if data.get("pid") != self.process.pid:
            data = {"state": "starting"}
        code = self.process.poll()
        if code is not None:
            data.update(state="stopped" if code == 0 else "error", exit_code=code)
            if code != 0 and not data.get("error"):
                data["error"] = "G4 backend exited with code %s; see %s" % (
                    code, self.root / "runtime.log")
        return data

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        self.log.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--records-root", required=True)
    parser.add_argument("--session", required=True)
    parser.add_argument("--parent-pid", type=int)
    parser.add_argument("--profile", help="explicit backend profile written by the owning session")
    args = parser.parse_args()
    if args.parent_pid is not None and sys.platform == "linux":
        import ctypes, signal
        # No orphan optimizer after an abrupt session-node exit. Check the race
        # between spawn and prctl registration as well.
        if ctypes.CDLL(None, use_errno=True).prctl(1, signal.SIGTERM) != 0:
            raise OSError(ctypes.get_errno(), "PR_SET_PDEATHSIG failed")
        if os.getppid() != args.parent_pid:
            return
    from .map_backend import build_backend
    root = output_root(args.records_root, args.session)
    profile = None if not args.profile else json.loads(Path(args.profile).read_text())
    # runtime.log is created by the parent before the backend's ownership file.
    backend = build_backend(args.records_root, args.session, root, profile)
    def status():
        data = {**backend.status(), "updated_unix": time.time(), "pid": os.getpid()}
        tmp = root / ".status.tmp"
        tmp.write_text(json.dumps(data, allow_nan=False))
        os.replace(tmp, root / "status.json")
    try:
        while True:
            try:
                backend._status.update(state="processing")
                status()
                backend.update()
            except Exception as error:
                print(type(error).__name__ + ": " + str(error), flush=True)
            status()
            time.sleep(2)
    finally:
        backend.close()


if __name__ == "__main__":
    main()
