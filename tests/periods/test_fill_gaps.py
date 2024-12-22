from libds.periods import fill_gaps

def test_returns_2_lists():
    r1, r2 = fill_gaps(list(), list())

    assert (isinstance(r1, list))
    assert (isinstance(r2, list))


def test_basic_filling():
    ordinals = [0, 1, 3]
    values = [ False, False, False]

    o, v = fill_gaps(ordinals, values)

    assert (o == [0, 1, 2, 3])
    assert (v == [False, False, False, False])


def test_unknown_mode():
    # Verify that using an unknown mode raise error
    ordinals = [0, 1, 3]
    values = [False, False, False]

    try:
        fill_gaps(ordinals, values, mode="unknown")
        assert False  # Should not reach here
    except ValueError as e:
        assert "Unknown mode unknown" in str(e)


def test_fill_gaps_true_between():
    ordinals = [0,      1,    3,     4,    5,    8,    9]
    values = [ False, True, False, False, True, True, False]

    o, v = fill_gaps(ordinals, values, mode="true_between")

    assert (o == [it for it in range(0,10)])
    assert (v == [False, True, False, False, False, True, True, True, True, False])

