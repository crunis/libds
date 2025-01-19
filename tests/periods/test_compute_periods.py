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


def test_prefix():
    res = compute_periods(makebool('TFFTTF'))
    assert list(res.keys()) == ['days', 'periods', 'intervals', 'durations', 'starts', 'ends', 'max_consec_days']

    res = compute_periods(makebool('TFFTTF'), prefix='fever_')
    assert list(res.keys()) == ['fever_days', 'fever_periods', 'fever_intervals',
                          'fever_durations', 'fever_starts', 'fever_ends',
                          'fever_max_consec_days']
    

def test_return_max_consec_days():
    res = compute_periods(makebool('TFFTTF'))
    assert res['max_consec_days'] == 2

    res = compute_periods(makebool('TFFTTTFFTTFFFF'))
    assert res['max_consec_days'] == 3

    res = compute_periods(makebool('FFFFFFFFF'))
    assert res['max_consec_days'] == 0
