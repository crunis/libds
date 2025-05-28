from collections import defaultdict
import pandas as pd
import pytest


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


def df_join_periods(df: pd.DataFrame, start_field='start_date', end_field='end_date'):
    dfs = list()
    for pid, data in df.groupby('pid'):
        data = data.sort_values(start_field)
        res = join_periods(zip(data[start_field].tolist(), data[end_field].tolist()))
        dfres = pd.DataFrame(res[0], columns=[start_field, end_field])
        dfres.insert(0, 'pid', pid)
        dfs.append(dfres)
    df = pd.concat(dfs)
    
    return df



def process_admissions(data, verbose = 0):
    stats = defaultdict(int)
    num_pids = len(set(data.pid))
    assert num_pids == 1, f"{num_pids} pids detected (should be one)"

    data = data.sort_values(['start_date', 'end_date'])
    current_admission = None
    partial_admissions = list()

    for n, row in data.iterrows():
        if verbose>5:
            print(pd.DataFrame([row]))

        # First episode of the list
        if current_admission is None:
            if verbose>5:
                print("First patient admission")
            stats["First patient admission"] += 1
            current_admission = row.drop(columns='_id')
            continue

        # START_DATEs coincide
        assert current_admission.start_date <= row.start_date, "row.start_date is older"
        if current_admission.start_date == row.start_date:
            if verbose>2:
                print("Same start_date")
            stats["Same start_date"] += 1
            assert current_admission.end_date <= row.end_date, "row.end_date is older"
            current_admission.end_date = row.end_date
            current_admission.end_dt   = row.end_dt
            current_admission.type += row.type
            continue

        # previous END_DATE after new START_DATE -> SUBSET OR OVERLAP
        if current_admission.end_date > row.start_date:
            if row.end_date <= current_admission.end_date:
                if verbose>2:
                    print("Subset, adds no information")
                stats["Subset, adds no information"] += 1
                continue
            if verbose>2:
                print("start_date inside previous episode but no subset")
            stats["start_date inside previous episode but no subset"] += 1
            current_admission.end_date = row.end_date
            current_admission.end_dt   = row.end_dt
            current_admission.type += row.type
            continue

        # previous END_DATE == new START_DATE --> CLEAN FOLLOW UP
        if current_admission.end_date == row.start_date:
            if verbose>5:
                print("Clean follow up")
            stats["Clean follow up"] += 1
            current_admission.end_date = row.end_date
            current_admission.end_dt   = row.end_dt
            current_admission.type += row.type
            continue

        # previous END_DATE < new START_DATE --> NEW admission
        if current_admission.end_date < row.start_date:
            if verbose>5:
                print("New admission for current patient")
            stats["New admission for current patient"] += 1
            partial_admissions.append(current_admission)
            current_admission = row
            continue

        # WEIRD CASE
        print("WEIRD, please check:")
        print("pid: ", row.pid)
        assert True == False

    partial_admissions.append(current_admission)

    return partial_admissions