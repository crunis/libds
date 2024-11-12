from .compute_periods import compute_periods
from .dates import dates_to_ordinal, dates_to_ordinal_with_values, unify_dates, dates_fill_period
from .compute_periods_simple import compute_periods_simple, df_w_intervals_compute_periods_simple
from .get_closest_event import get_closest_event

def find_interval_by_date(df, _dt, pid=None, strict=True):
    """
    Find the interval that contains a given date for a given patient.

    Parameters:
    df (pd.DataFrame): DataFrame containing the intervals with columns 'pid', 'start_dt', and 'end_dt'.
    _dt (datetime): The date to find the interval for.
    pid (int or str): Patient ID to search for. If None, all patients are considered. Defaults to None.
    strict (bool): If True, asserts that exactly one interval is found. Defaults to True.

    Returns:
    pd.Series: The row of the DataFrame that contains the interval with the given date for the specified patient.
    """
    if pid is None:
        res = df[(df.start_dt <= _dt) & (df.end_dt >= _dt)]
    else:
        res = df[(df.pid == pid) & (df.start_dt <= _dt) & (df.end_dt >= _dt)]

    if strict:
        assert len(res) == 1
    elif len(res) == 0:
        return None
    
    return res.iloc[0]
