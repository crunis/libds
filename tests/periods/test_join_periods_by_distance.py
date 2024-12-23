from libds.periods import compute_periods, join_periods_by_distance

def makebool(s):
    return [ True if x == 'T' else False for x in s ]

def test_is_callable():
    periods = compute_periods(makebool('FTTF'))
    res = join_periods_by_distance(periods, 1)

def test_simple_case():
    periods = compute_periods(makebool('FTTFT'))
    assert periods['days'] == 3
    assert periods['periods'] == 2

    res = join_periods_by_distance(periods, 1)
    assert res['days'] == 3
    assert res['periods'] == 1


def test_several_join_case():
    periods = compute_periods(makebool('FTTFTFFTTFTTTF'))
    assert periods['days'] == 8
    assert periods['periods'] == 4

    res = join_periods_by_distance(periods, 1)
    assert res['days'] == 8
    assert res['periods'] == 2
    assert res['starts'] == [1, 7]
    assert res['ends'] == [4, 12]

    res = join_periods_by_distance(periods, 2)
    assert res['days'] == 8
    assert res['periods'] == 1
    assert res['starts'] == [1]
    assert res['ends'] == [12]

