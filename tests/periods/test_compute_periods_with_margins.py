from libds.periods import compute_periods_with_margins


def test_initial_marging():
    values = [False, False, True, True, True]
    ordinals = [it for it in range(len(values))]

    res = compute_periods_with_margins(ordinals, values, initial_margin=2)

    assert res["days"] == 3
    assert res["periods"] == 1
    assert res["max_consec_days"] == 3

    res = compute_periods_with_margins(ordinals, values, initial_margin=3)

    assert res["days"] == 0
    assert res["periods"] == 0
    assert res["max_consec_days"] == 0


def test_cooloff():
    values = [True, True, False, True]
    ordinals = [it for it in range(len(values))]

    res = compute_periods_with_margins(ordinals, values, cooloff=0)
    assert res["days"] == 3
    assert res["periods"] == 2
    assert res["max_consec_days"] == 2
    assert res["interval_starts"] == [0, 3]
    assert res["interval_ends"] == [1, 3]
    assert res["interval_days"] == [2, 1]
    assert res["interval_days_since_last"] == [0, 1]


    res = compute_periods_with_margins(ordinals, values, cooloff=1)
    assert res["days"] == 3
    assert res["periods"] == 1
    assert res["max_consec_days"] == 2
    assert res["interval_starts"] == [0]
    assert res["interval_ends"] == [3]
    assert res["interval_days"] == [4]
    assert res["interval_days_since_last"] == [0]


def test_cooloff2():
    values = [ True if it == 'T' else False for it in "TTFTTTFFFTTTFFTTTT"]
    ordinals = [it for it in range(len(values))]

    res = compute_periods_with_margins(ordinals, values, cooloff=1)
    assert res["days"] == 12
    assert res["periods"] == 3
    assert res["max_consec_days"] == 4
    assert res["interval_starts"] == [0, 9, 14]
    assert res["interval_ends"] == [5, 11, 17]
    assert res["interval_days"] == [6, 3, 4]
    assert res["interval_days_since_last"] == [0, 3, 2]


    res = compute_periods_with_margins(ordinals, values, cooloff=2)
    assert res["days"] == 12
    assert res["periods"] == 2
    assert res["max_consec_days"] == 4
    assert res["interval_starts"] == [0, 9]
    assert res["interval_ends"] == [5, 17]
    assert res["interval_days"] == [6, 9]
    assert res["interval_days_since_last"] == [0, 3]

    res = compute_periods_with_margins(ordinals, values, cooloff=3)
    assert res["days"] == 12
    assert res["periods"] == 1
    assert res["max_consec_days"] == 4
    assert res["interval_starts"] == [0]
    assert res["interval_ends"] == [17]
    assert res["interval_days"] == [18]
    assert res["interval_days_since_last"] == [0]


def test_noperiods():
    values = [ True if it == 'T' else False for it in "FFFFFFFFFFFFFF"]
    ordinals = [it for it in range(len(values))]

    res = compute_periods_with_margins(ordinals, values, cooloff=2, initial_margin=2)
    assert res["days"] == 0
    assert res["periods"] == 0
    assert res["max_consec_days"] == 0
    assert res["interval_starts"] == []
    assert res["interval_ends"] == []
    assert res["interval_days"] == []
    assert res["interval_days_since_last"] == []
