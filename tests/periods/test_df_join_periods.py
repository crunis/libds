from libds.periods import df_join_periods
import pandas as pd


def test_simple():
    d1 = pd.to_datetime('2023-01-01 13:00:00')
    d2 = pd.to_datetime('2023-01-02 13:00:00')
    d3 = pd.to_datetime('2023-01-02 13:00:00')
    d4 = pd.to_datetime('2023-01-04 10:00:00')
    d5 = pd.to_datetime('2023-01-05 10:00:00')
    d6 = pd.to_datetime('2023-01-06 10:00:00')

    df = pd.DataFrame([
        {'pid': 1, 'start_date': d1, 'end_date': d2},
        {'pid': 1, 'start_date': d3, 'end_date': d4},
        {'pid': 1, 'start_date': d5, 'end_date': d6},
        {'pid': 2, 'start_date': d1, 'end_date': d2},
        {'pid': 2, 'start_date': d5, 'end_date': d6},
    ])

    # pytest.set_trace()

    res = df_join_periods(df)

    assert list(res.columns) == ['pid', 'start_date', 'end_date']
    assert list(res.pid) == [1, 1, 2, 2]
    assert list(res.start_date) == [d1, d5, d1, d5]
    assert list(res.end_date) == [d4, d6, d2, d6]



