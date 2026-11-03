"""Window grouping plans and their derivation rules.

A plan describes only frames that already have an identity and a timestamp.
It never carries model outputs, placements, point clouds or storage decisions:
the reader owns identity, the model owns depth/intrinsics/extrinsics, the graph
owns placement and component state, and storage owns the atomic revision.

Two grouping shapes are supported:

``total_frames``
    ``window_frames`` counts every model input frame, shared frames included.
    This is the incremental default and is unchanged by this module.
``new_frames``
    ``window_frames`` counts only newly captured frames; each window then feeds
    ``shared + new`` frames to the model. The first window has nothing to share,
    and ``plan_from_counts`` lets a final short window borrow extra shared frames
    so that its model input does not fall below ``minimum_total_frames``.

    The live session cannot know that a window is final, so it derives only the
    steady shape (``window_frames`` new frames plus ``overlap_frames`` shared).
    A short final window therefore exists only in a **registered** plan; when one
    is registered, ``check_plan_shape`` enforces the rule above instead of
    silently accepting a differently-shaped run.

The derivation below is the grouping rule the accepted long-window baseline
used. ``plan_from_records`` additionally accepts an explicitly registered plan
(the frozen candidate list schema) so an offline reproduction can be checked
frame by frame instead of being regenerated from a formula.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

SCHEMA = "drone3d.window_plan.v1"

# Defaults of the accepted long-window grouping. They are explicit inputs, not
# hidden constants: every caller can override each one.
FIRST_WINDOW_FRAMES = 100
STEADY_NEW_FRAMES = 100
STEADY_SHARED_FRAMES = 20
MINIMUM_TOTAL_FRAMES = 100

SEMANTICS_TOTAL = "total_frames"
SEMANTICS_NEW = "new_frames"
SEMANTICS = (SEMANTICS_TOTAL, SEMANTICS_NEW)


class WindowPlanError(ValueError):
    """Raised when a grouping contradicts frame identity or the plan schema."""


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise WindowPlanError(f"{name} must be a positive integer")
    return value


@dataclass(frozen=True)
class WindowSpec:
    """One window: new frames plus the shared prefix reused from earlier frames."""

    index: int
    id: str
    new_start: int
    new_end_exclusive: int
    input_start: int
    input_end_exclusive: int
    shared_prefix: int

    def __post_init__(self):
        if isinstance(self.index, bool) or not isinstance(self.index, int) or self.index < 0:
            raise WindowPlanError("window index must be a nonnegative integer")
        if not isinstance(self.id, str) or not self.id:
            raise WindowPlanError("window id must be a nonempty string")
        for name in ("new_start", "new_end_exclusive", "input_start", "input_end_exclusive", "shared_prefix"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise WindowPlanError(f"{name} must be a nonnegative integer")
        if self.new_end_exclusive <= self.new_start:
            raise WindowPlanError("a window must contain at least one new frame")
        if self.input_end_exclusive != self.new_end_exclusive:
            raise WindowPlanError("a window ends at its last new frame")
        if self.input_start != self.new_start - self.shared_prefix:
            raise WindowPlanError("shared prefix must start exactly shared_prefix frames before the new frames")
        if self.shared_prefix >= self.input_frames:
            raise WindowPlanError("a window needs at least one new frame")

    @property
    def new_frames(self) -> int:
        return self.new_end_exclusive - self.new_start

    @property
    def input_frames(self) -> int:
        return self.input_end_exclusive - self.input_start

    def to_record(self) -> dict:
        return {"id": self.id, "new_start": self.new_start, "new_end_exclusive": self.new_end_exclusive,
                "input_start": self.input_start, "input_end_exclusive": self.input_end_exclusive,
                "input_frames": self.input_frames, "shared_prefix": self.shared_prefix,
                "new_frames": self.new_frames}


@dataclass(frozen=True)
class WindowPlan:
    """An ordered, contiguous grouping of ``total_frames`` identified frames."""

    windows: tuple
    total_frames: int
    source: str = "derived"
    semantics: str = SEMANTICS_NEW
    shared_frames: int = STEADY_SHARED_FRAMES

    def __post_init__(self):
        windows = tuple(self.windows)
        if not windows:
            raise WindowPlanError("a plan needs at least one window")
        _positive_int(self.total_frames, "total_frames")
        if self.semantics not in SEMANTICS:
            raise WindowPlanError(f"semantics must be one of {SEMANTICS!r}")
        ids = [w.id for w in windows]
        if len(set(ids)) != len(ids):
            raise WindowPlanError("window ids must be unique")
        if [w.index for w in windows] != list(range(len(windows))):
            raise WindowPlanError("window indices must be contiguous from zero")
        if windows[0].new_start != 0:
            raise WindowPlanError("the first window must start at frame zero")
        for earlier, later in zip(windows, windows[1:]):
            if later.new_start != earlier.new_end_exclusive:
                raise WindowPlanError("windows must cover the frames contiguously")
            if later.input_start < earlier.new_start:
                raise WindowPlanError("a shared prefix must reference already captured frames")
        if windows[-1].new_end_exclusive != self.total_frames:
            raise WindowPlanError("the last window must end at total_frames")
        object.__setattr__(self, "windows", windows)

    def __len__(self):
        return len(self.windows)

    def __iter__(self):
        return iter(self.windows)

    @property
    def shared_prefixes(self) -> tuple:
        return tuple(w.shared_prefix for w in self.windows)

    @property
    def model_frames(self) -> tuple:
        return tuple(w.input_frames for w in self.windows)

    def spec(self, index) -> WindowSpec:
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(self.windows):
            raise WindowPlanError("window index out of range")
        return self.windows[index]

    def to_records(self) -> list:
        return [w.to_record() for w in self.windows]

    def to_dict(self) -> dict:
        return {"schema": SCHEMA, "source": self.source, "semantics": self.semantics,
                "shared_frames": self.shared_frames, "total_frames": self.total_frames,
                "window_count": len(self.windows), "windows": self.to_records()}

    def digest(self) -> str:
        """Identity of the grouping itself, not of where it was declared.

        A derived rule and a registered list that describe the same windows
        therefore share one digest; the declaration source is recorded
        separately so an equal grouping never needs two identities.
        """
        payload = json.dumps({"semantics": self.semantics, "shared_frames": self.shared_frames,
                              "total_frames": self.total_frames, "windows": self.to_records()},
                             sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def plan_from_counts(total_frames, *, first_new=FIRST_WINDOW_FRAMES, steady_new=STEADY_NEW_FRAMES,
                     shared=STEADY_SHARED_FRAMES, minimum_total=MINIMUM_TOTAL_FRAMES,
                     prefix="F", index_width=3) -> WindowPlan:
    """Derive the long-window grouping from the number of identified frames."""
    _positive_int(total_frames, "total_frames")
    _positive_int(first_new, "first_new")
    _positive_int(steady_new, "steady_new")
    _positive_int(shared, "shared")
    _positive_int(minimum_total, "minimum_total")
    if shared >= minimum_total:
        raise WindowPlanError("shared frames must be smaller than the minimum window length")
    if not isinstance(prefix, str) or not prefix:
        raise WindowPlanError("prefix must be a nonempty string")
    windows, position, index = [], 0, 0
    while position < total_frames:
        remaining = total_frames - position
        # A short window is final by construction: it only appears when fewer
        # new frames are left than a steady window would consume.
        new_frames = min(first_new if index == 0 else steady_new, remaining)
        is_final = new_frames == remaining and remaining < (first_new if index == 0 else steady_new)
        if index == 0:
            shared_prefix = 0
        elif is_final:
            shared_prefix = max(shared, minimum_total - new_frames)
        else:
            shared_prefix = shared
        shared_prefix = min(shared_prefix, position)
        windows.append(WindowSpec(index, f"{prefix}{index:0{index_width}d}", position, position + new_frames,
                                  position - shared_prefix, position + new_frames, shared_prefix))
        position += new_frames
        index += 1
    return WindowPlan(tuple(windows), total_frames, "derived", SEMANTICS_NEW, shared)


def _record_spec(index, record, label):
    if not isinstance(record, Mapping):
        raise WindowPlanError(f"{label}: window {index} must be an object")
    spec_fields = ("id", "new_start", "new_end_exclusive", "input_start", "input_end_exclusive", "shared_prefix")
    for field in spec_fields:
        if field not in record:
            raise WindowPlanError(f"{label}: window {index} lacks {field}")
        value = record[field]
        if field == "id":
            if not isinstance(value, str) or not value:
                raise WindowPlanError(f"{label}: window {index} id must be a nonempty string")
            continue
        if isinstance(value, bool) or not isinstance(value, int):
            raise WindowPlanError(f"{label}: window {index} field {field} must be an integer")
    try:
        return WindowSpec(int(index), record["id"], record["new_start"], record["new_end_exclusive"],
                          record["input_start"], record["input_end_exclusive"], record["shared_prefix"])
    except WindowPlanError:
        raise
    except (TypeError, ValueError) as error:
        raise WindowPlanError(f"{label}: window {index} is inconsistent: {error}") from error


def plan_from_records(records, *, total_frames=None, source="registered", semantics=SEMANTICS_NEW,
                      shared=None) -> WindowPlan:
    """Adopt an explicitly registered plan and check it against the frame count."""
    records = list(records)
    if not records:
        raise WindowPlanError("a registered plan needs at least one window")
    specs = tuple(_record_spec(index, record, source) for index, record in enumerate(records))
    declared = int(records[-1].get("new_end_exclusive", specs[-1].new_end_exclusive))
    if total_frames is None:
        total_frames = declared
    elif int(total_frames) != declared:
        raise WindowPlanError("registered plan does not cover every identified frame")
    if total_frames <= 0:
        raise WindowPlanError("registered plan must cover at least one frame")
    for index, (spec, record) in enumerate(zip(specs, records)):
        for name in ("input_frames", "new_frames"):
            if name in record and int(record[name]) != getattr(spec, name):
                raise WindowPlanError(f"{source}: window {index} {name} disagrees with its frame range")
    if shared is None:
        # A short final window only ever borrows *more* shared frames, so the
        # steady sharing count is the smallest prefix used after the first window.
        shared = min((spec.shared_prefix for spec in specs[1:]), default=STEADY_SHARED_FRAMES)
    return WindowPlan(specs, int(total_frames), source, semantics, int(shared))


def load_plan(path, *, total_frames=None, source=None) -> WindowPlan:
    """Read a registered plan from the frozen candidate-list schema."""
    path = Path(path)
    payload = json.loads(path.read_text())
    if isinstance(payload, Mapping):
        records = payload.get("windows")
        if records is None:
            raise WindowPlanError("plan object must contain a windows list")
        if total_frames is None and "total_frames" in payload:
            total_frames = int(payload["total_frames"])
    else:
        records = payload
    return plan_from_records(records, total_frames=total_frames, source=source or str(path))


def compare_plans(left: WindowPlan, right: WindowPlan) -> dict:
    """Diff two plans field by field; used by the offline reproduction check."""
    fields = ("id", "new_start", "new_end_exclusive", "input_start", "input_end_exclusive", "shared_prefix")
    differences = []
    for index in range(max(len(left), len(right))):
        if index >= len(left) or index >= len(right):
            differences.append({"index": index, "field": "window",
                                "left": None if index >= len(left) else left.spec(index).id,
                                "right": None if index >= len(right) else right.spec(index).id})
            continue
        a, b = left.spec(index), right.spec(index)
        for field in fields:
            if getattr(a, field) != getattr(b, field):
                differences.append({"index": index, "field": field, "left": getattr(a, field),
                                    "right": getattr(b, field)})
    return {"equal": not differences and left.total_frames == right.total_frames,
            "left_windows": len(left), "right_windows": len(right),
            "left_total_frames": left.total_frames, "right_total_frames": right.total_frames,
            "differences": differences}
