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
    ordinals = [0, 1, 2, 7, 9, 13, 15, 16]
    values = [False, True, False, False, True, True, False, True]

    res = compute_periods(ordinals, values)

    # assert res['max_consecutive_days'] == 2
    #
    assert res["days"] == 7
    assert res["periods"] == 3
    assert res["max_consec_days"] == 5
