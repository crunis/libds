import pandas as pd
from datetime import datetime

from libds.periods import get_closest_event, unify_dates

dates = [datetime(2020, 1, 1), 
         datetime(2020, 1, 11), 
         datetime(2020, 1, 21),
         datetime(2019, 1, 1)]

df = pd.DataFrame({
    "_id": range(len(dates)),
    "pid": [1] * len(dates),
    "admission_id": [1,1,1,2],
    "_dt": dates,
})


def test_get_closest_event():
    print(df)
    res = get_closest_event(df, 1, 1, datetime(2020, 1, 2))

    assert res["_id"] == 0
    assert res["_dt"] == datetime(2020, 1, 1)


# Check admission id usage
def test_get_closest_event_admission_id():
    res = get_closest_event(df, 1, 2, datetime(2020, 3, 1))

    assert res["_id"] == 3
    assert res["_dt"] == datetime(2019, 1, 1)


# Check days_after usage
def test_get_closest_event_days_after():
    res = get_closest_event(df, 1, 1, datetime(2020, 1, 20), days_after=0)

    assert res["_id"] == 1
    assert res["_dt"] == datetime(2020, 1, 11)


    res = get_closest_event(df, 1, 1, datetime(2020, 1, 20), days_after=3)

    assert res["_id"] == 2
    assert res["_dt"] == datetime(2020, 1, 21)


def test_unify_dates():
    dates = ['2020-01-01', '2020-01-02', '2020-01-04', '2020-01-07']
    dates = list(map(pd.to_datetime, dates))

    res = unify_dates(dates, days=1)
    assert res == [datetime(2020, 1, 1), datetime(2020, 1, 4), datetime(2020, 1, 7)]

    res = unify_dates(dates, days=2)
    assert res == [datetime(2020, 1, 1), datetime(2020, 1, 7)]
