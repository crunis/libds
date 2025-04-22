import math
import pandas as pd, numpy as np


TRUES  = set(['true', 'si', 'sí', 'yes', 1, '1'])
FALSES = set(['false', 'no', 0, '0'])
TFS = TRUES | FALSES | set([None, np.nan, pd.NA])

def toTrueFalse(value, TF=[1, 0]):
    if isinstance(value, str):
        value2 = value.lower()
    else:
        value2 = value

    if value2 in TRUES:
        return TF[0]

    if value2 in FALSES:
        return TF[1]

    return value


# Return if the column looks like a T/F or 1/0 col
def colIsTrueFalse(values):
    """
    Checks if all non-NaN values in an iterable belong to the set of
    recognized boolean representations (TFS), performing case-insensitive
    comparison for strings.

    Args:
        values: An iterable (e.g., list, Pandas Series) of values.

    Returns:
        True if all non-NaN values are potential booleans/None, False otherwise.
    """
    # Use a generator expression with all() for efficiency
    # Iterate through unique non-NaN values
    return all(
        (v.lower() if isinstance(v, str) else v) in TFS
        for v in set(values)
        # Filter out NaN values correctly before processing
        if not (isinstance(v, float) and math.isnan(v))
    )


def colToTrueFalse(values, TF=[1, 0]):
    """Converts a Pandas Series to True/False if all its values are
    recognized boolean representations.

    Applies the `toTrueFalse` function element-wise using .map() only if
    `colIsTrueFalse` returns True for the Series.

    Args:
        values: A Pandas Series.

    Returns:
        A Pandas Series with values converted to True/False/None,
        or the original Series if conversion is not applicable.
    """
    if colIsTrueFalse(values):
        return values.map(lambda value: toTrueFalse(value, TF))
    
    return values

