import pandas as pd

from .find_closest_event import find_closest_event
from .drop_close_dates import drop_close_dates


def correct_fillna(df, value):
    # In the future fillna wont automatically modify, for example, dtype object -> boolean.
    # This avoids the warning (by forcing the future behaviour) and manually applies the downcast
    with pd.option_context("future.no_silent_downcasting", True):
        df = df.fillna(value).infer_objects()
        
    return df