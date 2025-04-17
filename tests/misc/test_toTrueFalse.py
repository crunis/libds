# test_toTrueFalse.py

import pytest
import pandas as pd
import math
import numpy as np # Often used with pandas, useful for NaN

# Assuming toTrueFalse.py is in the same directory or accessible via PYTHONPATH
from libds.misc.toTrueFalse import toTrueFalse, colIsTrueFalse, colToTrueFalse

# --- Tests for toTrueFalse (single value conversion) ---

@pytest.mark.parametrize("input_val, expected_output", [
    # Default TF=[1, 0]
    ('true', 1),
    ('True', 1), # Case insensitivity check
    ('TRUE', 1), # Case insensitivity check
    ('si', 1),
    ('Si', 1),
    ('SI', 1),
    ('yes', 1),
    ('Yes', 1),
    ('YES', 1),
    (1, 1),
    ('1', 1),
    ('false', 0),
    ('False', 0),
    ('FALSE', 0),
    ('no', 0),
    ('No', 0),
    ('NO', 0),
    (0, 0),
    ('0', 0),
    # Values that should not be converted
    ('maybe', 'maybe'),
    (' true', ' true'), # Leading space
    ('true ', 'true '), # Trailing space
    (2, 2),
    (-1, -1),
    (1.0, 1), # Note: 1.0 is not in TRUES/FALSES sets
    (0.0, 0), # Note: 0.0 is not in TRUES/FALSES sets
    (None, None),
    ("", ""), # Empty string
])
def test_toTrueFalse_default_tf(input_val, expected_output):
    """Tests toTrueFalse with default TF=[1, 0]"""
    assert toTrueFalse(input_val) == expected_output

# Test NaN separately because NaN != NaN
def test_toTrueFalse_nan_default_tf():
    """Tests toTrueFalse with NaN input and default TF"""
    assert math.isnan(toTrueFalse(float('nan')))

@pytest.mark.parametrize("input_val, tf_param, expected_output", [
    # Custom TF=[True, False]
    ('true', [True, False], True),
    (1, [True, False], True),
    ('false', [True, False], False),
    (0, [True, False], False),
    ('maybe', [True, False], 'maybe'),
    (None, [True, False], None),
    # Custom TF=['Y', 'N']
    ('yes', ['Y', 'N'], 'Y'),
    ('1', ['Y', 'N'], 'Y'),
    ('no', ['Y', 'N'], 'N'),
    ('0', ['Y', 'N'], 'N'),
    (2, ['Y', 'N'], 2),
])
def test_toTrueFalse_custom_tf(input_val, tf_param, expected_output):
    """Tests toTrueFalse with custom TF parameters"""
    assert toTrueFalse(input_val, TF=tf_param) == expected_output

def test_toTrueFalse_nan_custom_tf():
    """Tests toTrueFalse with NaN input and custom TF"""
    assert math.isnan(toTrueFalse(float('nan'), TF=[True, False]))
    assert math.isnan(toTrueFalse(float('nan'), TF=['Y', 'N']))


# --- Tests for colIsTrueFalse (column check) ---

@pytest.mark.parametrize("input_list, expected_bool", [
    # Valid TF columns (should return True)
    (['true', 'false', 'si', 'no', 'yes', 1, 0, '1', '0'], True),
    (['True', 'False', 'SI', 'NO', 'Yes', 1, 0, '1', '0'], True), # Mixed case
    ([1, 0, 1, 1, 0], True),
    (['1', '0', '1', '1', '0'], True),
    (['true', 'false', None], True),
    ([1, 0, np.nan], True),
    ([None, np.nan], True),
    ([], True), # Empty list
    # Invalid TF columns (should return False)
    (['true', 'false', 'maybe'], False),
    ([1, 0, 2], False),
    (['yes', 'no', ''], False), # Contains empty string
    ([1, 0, 1.0], True), # Contains float 1.0
    ([1, 0, 0.0], True), # Contains float 0.0
    (['true', None, 'Invalid'], False),
])
def test_colIsTrueFalse_list_input(input_list, expected_bool):
    """Tests colIsTrueFalse with list inputs"""
    assert colIsTrueFalse(input_list) == expected_bool

@pytest.mark.parametrize("input_list, expected_bool", [
    # Using the same cases as above, but with Pandas Series
    (['true', 'false', 'si', 'no', 'yes', 1, 0, '1', '0'], True),
    (['True', 'False', 'SI', 'NO', 'Yes', 1, 0, '1', '0'], True),
    ([1, 0, 1, 1, 0], True),
    (['1', '0', '1', '1', '0'], True),
    (['true', 'false', None], True),
    ([1, 0, np.nan], True),
    ([None, np.nan], True),
    ([], True),
    (['true', 'false', 'maybe'], False),
    ([1, 0, 2], False),
    (['yes', 'no', ''], False),
    ([1, 0, 1.0], True),
    ([1, 0, 0.0], True),
    (['true', None, 'Invalid'], False),
])
def test_colIsTrueFalse_series_input(input_list, expected_bool):
    """Tests colIsTrueFalse with Pandas Series inputs"""
    series = pd.Series(input_list)
    assert colIsTrueFalse(series) == expected_bool

# --- Tests for colToTrueFalse (column conversion) ---

def test_colToTrueFalse_valid_conversion_default_tf():
    """Tests conversion of a valid TF column with default TF=[1, 0]"""
    input_series = pd.Series(['true', 'False', 1, '0', None, np.nan, pd.NA, 'si', 'NO'], dtype=object)
    expected_series = pd.Series([1, 0, 1, 0, None, np.nan, pd.NA, 1, 0], dtype=object) # Keep object dtype for None/NaN mix
    result_series = colToTrueFalse(input_series)
    print(result_series)
    pd.testing.assert_series_equal(result_series, expected_series, check_dtype=False, check_names=False)

def test_colToTrueFalse_valid_conversion_custom_tf_bool():
    """Tests conversion of a valid TF column with TF=[True, False]"""
    input_series = pd.Series(['true', 'False', 1, '0', None, np.nan, 'yes', 'no'])
    expected_series = pd.Series([True, False, True, False, None, np.nan, True, False], dtype=object)
    result_series = colToTrueFalse(input_series, TF=[True, False])
    pd.testing.assert_series_equal(result_series, expected_series, check_dtype=False, check_names=False)

def test_colToTrueFalse_valid_conversion_custom_tf_str():
    """Tests conversion of a valid TF column with TF=['Y', 'N']"""
    input_series = pd.Series(['true', 'False', 1, '0', None, np.nan, 'yes', 'no'])
    expected_series = pd.Series(['Y', 'N', 'Y', 'N', None, np.nan, 'Y', 'N'], dtype=object)
    result_series = colToTrueFalse(input_series, TF=['Y', 'N'])
    pd.testing.assert_series_equal(result_series, expected_series, check_dtype=False, check_names=False)

def test_colToTrueFalse_invalid_column_no_conversion():
    """Tests that an invalid column is returned unchanged"""
    input_series = pd.Series(['true', 'false', 'maybe', 1, 0])
    # Make a copy to ensure the original is not modified if the function incorrectly modifies in-place
    original_series_copy = input_series.copy()
    result_series = colToTrueFalse(input_series)
    # Should return the original series object if no conversion happens
    pd.testing.assert_series_equal(result_series, original_series_copy)
    # Optional: Check if it's the *same* object (if efficiency matters)
    # assert result_series is input_series

def test_colToTrueFalse_empty_series():
    """Tests conversion with an empty Series"""
    input_series = pd.Series([], dtype=object)
    expected_series = pd.Series([], dtype=object)
    result_series = colToTrueFalse(input_series)
    pd.testing.assert_series_equal(result_series, expected_series)

# def test_colToTrueFalse_all_none_nan_series():
#     """Tests conversion with a Series containing only None and NaN"""
#     input_series = pd.Series([None, np.nan, None], dtype=object)
#     expected_series = pd.Series([None, np.nan, None], dtype=object) # Should remain unchanged
#     result_series = colToTrueFalse(input_series)
#     pd.testing.assert_series_equal(result_series, expected_series)

