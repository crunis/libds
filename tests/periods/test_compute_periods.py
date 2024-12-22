from libds.periods import compute_periods


def test_compute_periods():
    ordinals = [0, 1, 2, 3, 4]
    values = [False, True, False, True, True]

    res = compute_periods(ordinals, values)

    assert res["days"] == 3
    assert res["periods"] == 2
    assert res["max_consec_days"] == 2


def test_compute_periods_prefix():
    ordinals = [0, 1, 2, 3, 4]
    values = [False, True, False, True, True]

    res = compute_periods(ordinals, values, prefix="potato")

    assert res["potato_days"] == 3
    assert res["potato_periods"] == 2
    assert res["potato_max_consec_days"] == 2


def test_compute_periods_w_gaps():
    # If there are gaps, they are filled with True only between Trues.
    ordinals = [0,     1,     2,     7,     9,  13,   15,    16]
    values = [False, True, False, False, True, True, False, True]
    # True: 1,9,10,11,12,13,16

    res = compute_periods(ordinals, values)

    assert res["days"] == 7
    assert res["periods"] == 3
    assert res["max_consec_days"] == 5


def test_compute_periods_extra():
    ordinals = [0, 1, 2, 3, 4, 5, 6, 7]
    values = [False, True, False, True, True, False, False, True]

    res = compute_periods(ordinals, values, extra_stats=True)

    assert res["days"] == 4
    assert res["periods"] == 3
    assert res["max_consec_days"] == 2
    assert res["interval_days_since_last"] == [1, 1, 2]
    assert res["interval_starts"] == [1, 3, 7]
    assert res["interval_ends"] == [1, 4, 7]


def test_days_since_last():
    # Simple
    ordinals = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    values = [True, False, False, False, True, True, True, True, False, True]
    res = compute_periods(ordinals, values, mode='true_between', extra_stats=True)
    assert res["periods"] == 3
    assert res['days'] == 6
    assert res["max_consec_days"] == 4
    assert res["interval_days_since_last"] ==  [0, 3, 1]

   
    # Gaps on true
    ordinals = [0, 1, 2, 3, 4, 7, 8, 9]
    values = [True, False, False, False, True, True, False, True]
    res = compute_periods(ordinals, values, mode='true_between', extra_stats=True)
    assert res["periods"] == 3
    assert res['days'] == 6
    assert res["max_consec_days"] == 4
    assert res["interval_days_since_last"] ==  [0, 3, 1]

    # Gaps on False
    ordinals = [0, 1, 3, 4, 7, 8, 9]
    values = [True, False, False, True, True, False, True]
    res = compute_periods(ordinals, values, mode='true_between', extra_stats=True)
    assert res["periods"] == 3
    assert res['days'] == 6
    assert res["max_consec_days"] == 4
    assert res["interval_days_since_last"] ==  [0, 3, 1]


def test_interval_days():
    # Simple
    ordinals = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    values = [True, False, False, False, True, True, True, True, False, True]
    res = compute_periods(ordinals, values, mode='true_between', extra_stats=True)
    assert res["interval_days"] ==  [1, 4, 1]

   
    # Gaps on true
    ordinals = [0, 1, 2, 3, 4, 7, 8, 9]
    values = [True, False, False, False, True, True, False, True]
    res = compute_periods(ordinals, values, mode='true_between', extra_stats=True)
    assert res["interval_days"] ==  [1,4,1]

    # Gaps on False
    ordinals = [0, 1, 3, 4, 7, 8, 9]
    values = [True, False, False, True, True, False, True]
    res = compute_periods(ordinals, values, mode='true_between', extra_stats=True)
    assert res["interval_days"] ==  [1, 4, 1]


def test_no_periods():
    ordinals = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    values = [False, False, False, False, False, False, False, False, False, False]

    res = compute_periods(ordinals, values, extra_stats=True)
    assert res["days"] == 0
    assert res["periods"] == 0
    assert res["max_consec_days"] == 0
    assert res['interval_starts'] == []
    assert res['interval_ends'] == []
    assert res['interval_days'] == []
    assert res['interval_days_since_last'] == []