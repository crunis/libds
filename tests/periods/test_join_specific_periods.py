from libds.periods import join_specific_periods
from copy import deepcopy


# TFTTF
p1 = dict(
    days=3,
    periods=2,
    intervals=[0, 1],
    durations=[1,2],
    starts = [0,2],
    ends = [0,3]
)

p0 = dict(
    days=3,
    periods=1,
    intervals=[0],
    durations=[1],
    starts = [0],
    ends = [0]
)

def test_doesnt_modify_param():
    p1c = deepcopy(p1)
    join_specific_periods(p1c, 0)
    assert p1 == p1c


def test_simple_join():
    res = join_specific_periods(p1, 0)

    assert res['periods'] == 1
    assert res['intervals'] == [0]
    assert res['durations'] == [4]
    assert res['starts'] == [0]
    assert res['ends'] == [3]




def test_out_of_bounds():
    try:
        res = join_specific_periods(p1, 1)
    except IndexError as e:
        assert str(e) == 'period index out of range'

    try:
        res = join_specific_periods(p1, -1)
    except IndexError as e:
        assert str(e) == 'period index out of range'

    try:
        res = join_specific_periods(p0, 0)
    except IndexError as e:
        assert str(e) == 'period index out of range'




    
