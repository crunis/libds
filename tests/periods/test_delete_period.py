from libds.periods import delete_period
from copy import deepcopy

# TFTT
p1 = dict(
    days=3,
    periods=2,
    intervals=[0, 1],
    durations=[1,2],
    starts = [0,2],
    ends = [0,3]
)

# TFTTFFT
p2 = dict(
    days=4,
    periods=3,
    intervals=[0, 1, 2],
    durations=[1,2, 1],
    starts = [0,2,6],
    ends = [0,3,6]
)

# TF
p0 = dict(
    days=1,
    periods=1,
    intervals=[0],
    durations=[1],
    starts = [0],
    ends = [0]
)

def test_doesnt_modify_param():
    p1c = deepcopy(p1)
    delete_period(p1c, 0)
    assert p1 == p1c


def test_delete1():
    res = delete_period(p1, 0)

    assert res['periods'] == 1
    assert res['intervals'] == [2]
    assert res['durations'] == [2]
    assert res['starts'] == [2]
    assert res['ends'] == [3]


def test_delete2():
    res = delete_period(p2, 1)

    assert res['periods'] == 2
    assert res['intervals'] == [0, 5]
    assert res['durations'] == [1,1]
    assert res['starts'] == [0,6]
    assert res['ends'] == [0,6]