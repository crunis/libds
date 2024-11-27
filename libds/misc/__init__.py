import pandas as pd
from pandas.errors import OptionError
from unidecode import unidecode

from .find_closest_event import find_closest_event
from .drop_close_dates import drop_close_dates


def correct_fillna(df, value):
    try:
        # Try using the future behaviour option if it exists
        with pd.option_context("future.no_silent_downcasting", True):
            df = df.fillna(value).infer_objects()
    except OptionError:
        df = df.fillna(value).infer_objects()
        
    return df


def flatten_text(text):
    return unidecode(text).lower().strip()