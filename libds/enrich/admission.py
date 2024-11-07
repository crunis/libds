import pandas as pd
from libds.periods import find_interval_by_date

def get_admission(df: pd.DataFrame, pid: int, _dt: pd.DatetimeTZDtype, strict: bool = True) -> pd.Series:
    """
    Find the admission that contains a given date for a given patient.

    Parameters:
    df (pd.DataFrame): DataFrame containing the admission intervals with columns 'pid', 'start_dt', and 'end_dt'.
    pid (int): Patient ID to search for.
    _dt (datetime): The date to find the interval for.
    
    Returns:
    pd.Series: The row of the DataFrame that contains the admission interval with the given date for the specified patient.
    """
    res = find_interval_by_date(df, _dt, pid, strict)

    return res


def get_admission_id(df: pd.DataFrame, pid: int, _dt: pd.DatetimeTZDtype, strict: bool = True) -> pd.Series:
    res = find_interval_by_date(df, _dt, pid, strict)

    if res is None:
        return None

    return res._id