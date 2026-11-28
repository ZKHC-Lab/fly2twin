# SPDX-License-Identifier: Apache-2.0
import threading
import time

import pytest

from recon.io.common import ahead


def test_values_and_order_match_map():
    def slow(i):
        time.sleep(0.002 * (i % 3))
        return i * i

    for workers in (0, 1, 4):
        assert list(ahead(range(50), slow, workers, depth=8)) == [i * i for i in range(50)]


def test_source_is_read_at_most_depth_ahead():
    taken = []

    def source():
        for i in range(40):
            taken.append(i)
            yield i

    results = ahead(source(), lambda i: i, workers=2, depth=5)
    for count in range(1, 21):
        next(results)
        assert len(taken) <= count + 5
    results.close()


def test_loads_run_off_the_calling_thread():
    names = set()

    def load(i):
        names.add(threading.current_thread().name)
        return i

    assert list(ahead(range(8), load, workers=2, depth=4)) == list(range(8))
    assert threading.current_thread().name not in names
    names.clear()
    assert list(ahead(range(8), load, workers=0)) == list(range(8))
    assert names == {threading.current_thread().name}


def test_error_surfaces_at_its_position():
    def load(i):
        if i == 6:
            raise ValueError('frame 6')
        return i

    results = ahead(range(20), load, workers=3, depth=4)
    assert [next(results) for _ in range(6)] == list(range(6))
    with pytest.raises(ValueError, match='frame 6'):
        next(results)
