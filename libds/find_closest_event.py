import pandas as pd


def find_closest_event(df, pid, _dt, inclusive=False, prefix=""):
    """
    Find the closest event to the given date.
    Design to be used with df.apply (returns a Series).
    Args:
        df (pd.DataFrame): The DataFrame containing the events.
        pid (int): The patient ID.
        _dt (datetime): The date to compare.
        inclusive (bool): If True, the comparison is inclusive.
        prefix (str): The prefix for the resulting columns.
    Returns:
        pd.Series: A series containing the following columns:
            - f"{prefix}_": True if an event was found, False otherwise.
            - f"{prefix}_days": The number of days between the event and the given date.
    """
    if inclusive:
        date_cond = df._dt <= _dt
    else:
        date_cond = df._dt < _dt
    events = df[(df.pid == pid) & date_cond]

    if len(events) == 0:
        return pd.Series({f"{prefix}_": False, f"{prefix}_days": 1000})

    event_dt = events.sort_values(by="_dt", ascending=False).iloc[0]._dt
    days = (_dt - event_dt).days

    return pd.Series({f"{prefix}_": True, f"{prefix}_days": days})
