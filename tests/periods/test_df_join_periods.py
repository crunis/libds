from libds.periods import df_join_periods_with_detailed_mapping, join_up_to_a_day
import pandas as pd


def test_simple():
    d1 = pd.to_datetime('2023-01-01 13:00:00')
    d2 = pd.to_datetime('2023-01-02 13:00:00')
    d3 = pd.to_datetime('2023-01-02 13:00:00')
    d4 = pd.to_datetime('2023-01-04 10:00:00')
    d5 = pd.to_datetime('2023-01-05 10:00:00')
    d6 = pd.to_datetime('2023-01-06 10:00:00')

    df = pd.DataFrame([
        {'pid': 1, '_id': 1, 'eid': 10, 'start_dt': d1, 'end_dt': d2},
        {'pid': 1, '_id': 2, 'eid': 11, 'start_dt': d3, 'end_dt': d4},
        {'pid': 1, '_id': 3, 'eid': 12, 'start_dt': d5, 'end_dt': d6},
        {'pid': 2, '_id': 4, 'eid': 13, 'start_dt': d1, 'end_dt': d2},
        {'pid': 2, '_id': 5, 'eid': 14, 'start_dt': d5, 'end_dt': d6},
    ])

    res, _ = df_join_periods_with_detailed_mapping(df)

    assert list(res.columns) == [
        'admission_id',
        'pid',
        'admission_n',
        'start_dt',
        'end_dt',
        'num_episodes',
    ]
    assert list(res.pid) == [1, 1, 2, 2]
    assert list(res.start_dt) == [d1, d5, d1, d5]
    assert list(res.end_dt) == [d4, d6, d2, d6]


def test_fixtures():
    df_episodes = pd.read_parquet('tests/fixtures/df_episodes.parquet')
    df_admissions = pd.read_parquet('tests/fixtures/df_admissions.parquet')
    df_mappings = pd.read_parquet('tests/fixtures/df_mappings.parquet')

    res_admissions, res_mappings = df_join_periods_with_detailed_mapping(
        df_episodes,
        join_condition=join_up_to_a_day)
    assert len(df_admissions.compare(res_admissions)) == 0
    assert len(df_mappings.compare(res_mappings)) == 0
