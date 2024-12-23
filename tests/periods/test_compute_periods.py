from libds.periods import compute_periods


def makebool(s):
    return [ True if x == 'T' else False for x in s ]

def test_is_callable():
    res = compute_periods(makebool('FTTF'))


def test_small_period():
    res = compute_periods(makebool('FTTF'))
    
    assert res['days'] == 2
    assert res['periods'] == 1


def test_two_periods():
    res = compute_periods(makebool('FTTFTTTF'))

    assert res['days'] == 5
    assert res['periods'] == 2


def test_begins_and_ends():
    res = compute_periods(makebool('TFFTTF'))
    assert res['days'] == 3
    assert res['periods'] == 2

    res = compute_periods(makebool('FFTFTT'))
    assert res['days'] == 3
    assert res['periods'] == 2

    res = compute_periods(makebool('TFFTTFT'))
    assert res['days'] == 4
    assert res['periods'] == 3


def test_intervals():
    res = compute_periods(makebool('TFFTTF'))
    assert res['intervals'] == [0, 2]

    res = compute_periods(makebool('FTFFTTFFFT'))
    assert res['intervals'] == [1, 2, 3]


def test_durations():
    res = compute_periods(makebool('FTFFTTFFFTTT'))
    assert res['durations'] == [1, 2, 3]

    res = compute_periods(makebool('FTFFTTFFFTF'))
    assert res['durations'] == [1, 2, 1]

    res = compute_periods(makebool('TTTFFTTFFFTTT'))
    assert res['durations'] == [3, 2, 3]


def test_starts_and_ends():
    res = compute_periods(makebool('TFFTTF'))
    assert res['starts'] == [0, 3]
    assert res['ends'] == [0, 4]

    res = compute_periods(makebool('FTFFTT'))
    assert res['starts'] == [1, 4]
    assert res['ends'] == [1, 5]
    
