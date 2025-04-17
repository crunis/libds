import pandas as pd
import numpy as np
from typing import List, Union
import warnings



def gen_dummies(
    feature_data: Union[pd.DataFrame, pd.Series],
    prefix: str,
    process_na: bool = True,
    convert_to_float: bool = True
) -> pd.DataFrame:
    """
    Generates dummy (indicator) variables for a single feature (Series or 1-col DF).

    Handles NaNs specifically: rows with original NaNs will have NaN values
    across all generated dummy columns if process_na is True.

    Args:
        feature_data: Input pandas Series or single-column DataFrame.
        prefix: String prefix for the new dummy column names.
        process_na: If True, handle NaNs by setting all dummy columns for NaN
                    rows to NaN. Requires convert_to_float=True.
        convert_to_float: If True, convert dummy columns to float dtype. This
                    usually means that from True/False goes to 1/0

    Returns:
        A pandas DataFrame containing the dummy variables for the single feature.

    Raises:
        ValueError: If process_na is True but convert_to_float is False.
        ValueError: If input feature_data is a DataFrame with more than one column.
    """
    if process_na and not convert_to_float:
        raise ValueError("process_na=True requires convert_to_float=True")

    if isinstance(feature_data, pd.DataFrame):
        if feature_data.shape[1] != 1:
            raise ValueError("Input 'feature_data' must be a pandas Series or a single-column DataFrame.")
        # If single-column DF, extract the Series for get_dummies
        feature_series = feature_data.iloc[:, 0]
    elif isinstance(feature_data, pd.Series):
        feature_series = feature_data
    else:
        raise TypeError("Input 'feature_data' must be a pandas Series or DataFrame.")

    if feature_series.empty:
        return pd.DataFrame(index=feature_series.index) # Handle empty Series input

    # Use standard get_dummies, simpler now
    dummy_df = pd.get_dummies(feature_series, prefix=prefix, dummy_na=process_na, prefix_sep='_')

    if convert_to_float:
         dummy_df = dummy_df.astype(float)

    # Apply specific NaN processing AFTER potential type conversion
    if process_na:
        nan_column = f"{prefix}_nan"
        if nan_column in dummy_df.columns:
            nan_rows_mask = dummy_df[nan_column] == 1
            dummy_df.loc[nan_rows_mask, :] = pd.NA
            dummy_df = dummy_df.drop(columns=nan_column)

    return dummy_df


def gen_dummies_from_combined_columns(
    df: pd.DataFrame,
    columns_to_combine: List[str],
    prefix: str,
    prefix_sep: str = '_',
    process_na: bool = True,
    convert_to_float: bool = True
) -> pd.DataFrame:
    """
    Generates one set of dummy variables by treating multiple input columns
    as containing values for the same underlying categorical feature space.

    A dummy variable for a category 'X' (e.g., f"{prefix}{prefix_sep}X")
    will be 1 for a given row if 'X' appears in *any* of the
    'columns_to_combine' for that row.

    Args:
        df: Input DataFrame.
        columns_to_combine: List of column names in 'df' whose values should be
                            combined into a single dummy set.
        prefix: String prefix for the new dummy column names.
        prefix_sep: Separator used between prefix and value in dummy column names.
        process_na: If True, rows where *all* specified 'columns_to_combine'
                    are NaN will result in NaN values across all generated
                    dummy columns for that row. Requires convert_to_float=True.
                    If False, rows with only NaNs in the specified columns
                    will result in all 0s for the dummy variables.
        convert_to_float: If True, convert the resulting dummy columns to float
                          dtype. Defaults to True. Necessary if process_na=True.

    Returns:
        A pandas DataFrame containing the combined dummy variables,
        indexed like the original DataFrame 'df'.

    Raises:
        ValueError: If process_na is True but convert_to_float is False.
        KeyError: If any column in columns_to_combine is not in df.
        TypeError: If df is not a DataFrame or columns_to_combine is not a list.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input 'df' must be a pandas DataFrame.")
    if not isinstance(columns_to_combine, list):
        raise TypeError("Input 'columns_to_combine' must be a list of column names.")
    if not columns_to_combine:
        warnings.warn("'columns_to_combine' is empty. Returning an empty DataFrame.")
        return pd.DataFrame(index=df.index)

    if process_na and not convert_to_float:
        raise ValueError("process_na=True requires convert_to_float=True")

    # Select the relevant subset and check columns exist
    try:
        # Use copy to avoid SettingWithCopyWarning later if modifying df_subset
        df_subset = df[columns_to_combine].copy()
    except KeyError as e:
        raise KeyError(f"One or more columns in {columns_to_combine} not found in DataFrame.") from e

    if df_subset.empty:
         return pd.DataFrame(index=df.index)

    # --- Stack, Dummify, Aggregate ---
    # Stack the data: creates a Series with MultiIndex (original_index, column_name)
    # stack() automatically drops NaNs, which is usually desired here.
    stacked_data = df_subset.stack()

    if stacked_data.empty:
        # No non-NaN data found in the columns to combine.
        # We still need to return a DF with the correct index.
        # If process_na=True, NaNs will be filled later. Otherwise, it's all zeros.
        dummy_df = pd.DataFrame(index=df.index)
    else:
        dummies_stacked = pd.get_dummies(
            stacked_data,
            prefix=prefix,
            prefix_sep=prefix_sep,
            dummy_na=False # NaNs already dropped by stack()
        )

        # Aggregate back to the original index using max.
        # Group by the first level of the MultiIndex (original_index).
        # .max() ensures a 1 if the category appeared in *any* column for that row.
        dummy_df = dummies_stacked.groupby(level=0).max()

    if convert_to_float:
        if not pd.api.types.is_float_dtype(dummy_df.dtypes):
             dummy_df = dummy_df.astype(float)

    # Reindex to ensure all original rows are present.
    # Rows that only had NaNs (or were empty) in the subset will be filled
    #   with 0 or pd.NA depending on process_na
    # This step also ensures the output has the same index as the input df.
    fill_value = pd.NA if process_na else 0
    dummy_df = dummy_df.reindex(df.index, fill_value=fill_value)

    return dummy_df

