"""Bounded RGB backlog used only between consecutive continuous capture goals.

The action server owns the lock. This buffer never stores snapshots or modifies
messages; bytes and eviction counts make overload visible instead of unbounded.
"""
from collections import deque
import re


def stamp_ns(message):
    stamp = message.header.stamp
    sec = getattr(stamp, "secs", getattr(stamp, "sec", 0))
    nsec = getattr(stamp, "nsecs", getattr(stamp, "nsec", 0))
    return int(sec) * 10**9 + int(nsec)


def consecutive(previous, current):
    a = re.fullmatch(r"(cs-s[0-9a-f]{16})-w([0-9]+)", previous or "")
    b = re.fullmatch(r"(cs-s[0-9a-f]{16})-w([0-9]+)", current or "")
    return bool(a and b and a[1] == b[1] and int(b[2]) == int(a[2]) + 1)


class ImageHandoff:
    def __init__(self, max_frames, max_bytes):
        self.max_frames, self.max_bytes = max_frames, max_bytes
        self.queue = deque()
        self.bytes = 0
        self.evicted = 0
        self.replayed = 0
        self.decimated = 0
        # The sampler only takes one image per goal sample period, so storing the
        # full camera rate wastes 2/3 of the capacity at 15 Hz source / 5 Hz goal.
        self.min_interval_ns = 0

    def clear(self):
        self.queue.clear()
        self.bytes = 0

    def push(self, message, front=False):
        if not self.max_frames:
            return
        try:
            stamp = stamp_ns(message)
            size = len(message.data)
        except (AttributeError, TypeError, ValueError):
            return
        if stamp <= 0:
            return
        if not front and self.queue and stamp - self.queue[-1][0] < self.min_interval_ns:
            self.decimated += 1
            return
        item = (stamp, message, size)
        self.queue.appendleft(item) if front else self.queue.append(item)
        self.bytes += size
        while len(self.queue) > self.max_frames or self.bytes > self.max_bytes:
            self.bytes -= self.queue.popleft()[2]
            self.evicted += 1

    def pop_after(self, cutoff):
        while self.queue:
            stamp, message, size = self.queue.popleft()
            self.bytes -= size
            if stamp > cutoff:
                return stamp, message
        return None
