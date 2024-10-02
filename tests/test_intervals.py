import pandas as pd
from libds.intervals import intervals_select_by_interval, contains_interval


def test_intervals_select_by_interval():
    # fmt: off
    df = pd.DataFrame({
        'pid': [1, 1, 1, 2, 2, 2],
        'start_dt': ['2020-01-01', '2020-01-02', '2020-01-03', '2020-01-01', '2020-01-02', '2020-01-03'],
        'end_dt': ['2020-01-02', '2020-01-03', '2020-01-04', '2020-01-02', '2020-01-03', '2020-01-04'],
    })
    # fmt: on
    df["start_dt"] = pd.to_datetime(df["start_dt"])
    df["end_dt"] = pd.to_datetime(df["end_dt"])

    res = intervals_select_by_interval(
        df, 1, "2020-01-02", "2020-01-03", contained=True
    )
    assert res.shape[0] == 1
    assert res.iloc[0]["start_dt"] == pd.to_datetime("2020-01-02")
    assert res.iloc[0]["end_dt"] == pd.to_datetime("2020-01-03")

    res = intervals_select_by_interval(
        df, 1, "2020-01-02", "2020-01-03", contained=False
    )
    assert res.shape[0] == 3
    assert res.iloc[1]["start_dt"] == pd.to_datetime("2020-01-02")
    assert res.iloc[1]["end_dt"] == pd.to_datetime("2020-01-03")
    assert res.iloc[2]["start_dt"] == pd.to_datetime("2020-01-03")
    assert res.iloc[2]["end_dt"] == pd.to_datetime("2020-01-04")


def test_contains_interval():
    # fmt: off
    df = pd.DataFrame({
        'pid': [1, 1, 1, 2, 2, 2],
        'start_dt': ['2020-01-01', '2020-01-07', '2020-01-10','2020-01-01', '2020-01-01', '2020-01-01'],
        'end_dt':   ['2020-01-03', '2020-01-09', '2020-01-12','2020-01-22', '2020-01-22', '2020-01-22'],
    })
    # fmt: on
    df["start_dt"] = pd.to_datetime(df["start_dt"])
    df["end_dt"] = pd.to_datetime(df["end_dt"])

    res = contains_interval(df, 1, "2020-01-01", "2020-01-04", contained=True)
    assert res == True

    res = contains_interval(df, 1, "2020-01-01", "2020-01-02", contained=False)
    assert res == True

    res = contains_interval(df, 1, "2020-01-01", "2020-01-02", contained=True)
    assert res == False

    res = contains_interval(df, 1, "2020-01-02", "2020-01-04", contained=False)
    assert res == True

    res = contains_interval(df, 1, "2020-01-08", "2020-01-11", contained=True)
    assert res == False

    res = contains_interval(df, 1, "2020-01-08", "2020-01-11", contained=False)
    assert res == True
