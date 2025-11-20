import pandas as pd
import numpy as np
from typing import List, Union
import warnings


def ch_get_dummies(df, columns=[], prefix=None, extra_na_strings=None, 
                                    process_na=True, convert_to_float=False):
    """
    Converts multiple columns into dummy columns, handling dirty "null" strings.
    
    Args:
        df: Input DataFrame or Series.
        columns: List of column names to encode. If empty, uses all columns.
        prefix: Optional prefix for output columns (not currently implemented in logic below, but kept for API compatibility).
        extra_na_strings: List of additional strings to treat as NaN (e.g. ["Not Applicable"]).
        process_na: Whether to clean and process NA values (default True).
        convert_to_float: If True, returns float (0.0/1.0) instead of boolean (False/True).
    """

    # Define what looks like a "Null" string
    # These are common artifacts in CSVs/Excel files
    null_strings = [
        "None", "none", "NONE", 
        "NaN", "nan", "NAN", 
        "Null", "null", "NULL", 
        "NA", "na", 
        "", " ", "?"
    ]

    if isinstance(df, pd.Series):
        df = df.to_frame()
    elif not isinstance(df, pd.DataFrame):
        raise TypeError("Input 'df' must be a pandas Series or DataFrame.")

    if (columns is None) or len(columns) == 0:
        columns = df.columns
    
    if extra_na_strings:
        null_strings.extend(extra_na_strings)

    if process_na:
        subset = df[columns].copy()
        subset = subset.replace(null_strings, np.nan)

        # Identify rows where EVERYTHING is missing (after cleaning)
        all_na_mask = subset.isna().all(axis=1)

        # Stack (Wide -> Long)
        # stack() drops np.nan automatically
        stacked = subset.stack()
    else:
        stacked = df[columns].stack()

    dummies = pd.get_dummies(stacked, prefix='', prefix_sep='', dtype='int8')

    # Aggregate (Group by original index and take Max)
    dummies = dummies.groupby(level=0).max()

    # Reindex to recover rows dropped by stack()
    dummies = dummies.reindex(df.index, fill_value=0)

    if convert_to_float:
        dummies = dummies.astype(float)
    else:
        dummies = dummies.astype('boolean')
    
    if process_na:
        dummies.loc[all_na_mask] = pd.NA
    
    if prefix:
        dummies = dummies.add_prefix(f"{prefix}_")
    
    return dummies