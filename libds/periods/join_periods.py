
from joblib import Parallel, delayed
from collections import defaultdict
import pandas as pd


def default_join_periods_join_condition(item1, item2):
    return item1[1] == item2[0]


def join_up_to_a_day(period1, period2):
    seconds = (period2[0] - period1[1]).total_seconds()
    if seconds <= 86400:
        return True
    return False


def join_periods(
        periods,
        errors = 'raise',
        join_condition = default_join_periods_join_condition
        ):
    """
    For this to work properly, periods should be sorted by first column
    """
    current_period = None
    joined_periods = list()
    correspondences = list()
    current_correspondence = list()
    stats = defaultdict(int)

    for n, period in enumerate(periods):
        if isinstance(period, tuple):
            period = list(period)

        # First period
        if current_period is None:
            current_period = period
            current_correspondence.append(n)
            continue

        if period[0] < current_period[1]:
            if period[1] <= current_period[1]:
                # subset, ignore
                stats["subset"] += 1
                current_correspondence.append(n)
                continue
            else:
                # overlap
                stats["overlap"] += 1
                current_period[1] = period[1]
                current_correspondence.append(n)
                continue

        # Join periods
        if join_condition(current_period, period):
            current_period[1] = period[1]
            current_correspondence.append(n)
            continue

        # New period
        joined_periods.append(current_period)
        current_period = period
        correspondences.append(current_correspondence)
        current_correspondence = [n]

    joined_periods.append(current_period)
    correspondences.append(current_correspondence)

    return joined_periods, correspondences, dict(stats)


def df_join_periods_loop(pid, df, start_field, end_field, join_condition):
    res = join_periods(
            zip(df[start_field].tolist(), df[end_field].tolist()),
            join_condition=join_condition
            )
    dfres = pd.DataFrame(res[0], columns=[start_field, end_field])
    dfres.insert(0, 'pid', pid)

    return dfres


def df_join_periods(
        df: pd.DataFrame, 
        start_field='start_date', 
        end_field='end_date',
        join_condition = default_join_periods_join_condition,
        n_jobs=None,
        ):
    if n_jobs is None:
        dfs = [
            df_join_periods_loop(pid, data, start_field, end_field, join_condition)
            for pid, data in df.sort_values(start_field).groupby('pid')
        ]
    else:
        dfs = Parallel(n_jobs=n_jobs)(
            delayed(df_join_periods_loop)(pid, data, start_field, end_field, join_condition)
            for pid, data in df.sort_values(start_field).groupby('pid')
        )
    df = pd.concat(dfs)
    
    return df
