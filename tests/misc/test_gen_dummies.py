# /Users/crunis/libds/tests/test_gen_dummies.py

import pytest
import pandas as pd
import numpy as np

from libds.misc.gen_dummies import gen_dummies, gen_dummies_from_combined_columns

# === Fixtures ===
@pytest.fixture
def sample_series():
    """A simple pandas Series for testing gen_dummies."""
    return pd.Series(['A', 'B', 'A', None, 'C', 'B'], name='feature1', index=[10, 20, 30, 40, 50, 60])

@pytest.fixture
def sample_dataframe_single_col():
    """A simple single-column DataFrame for testing gen_dummies."""
    return pd.DataFrame({'feature1': ['A', 'B', 'A', None, 'C', 'B']}, index=[13, 20, 30, 42, 50, 60])

@pytest.fixture
def sample_dataframe_multi_col():
    """DataFrame for testing gen_dummies_from_combined_columns."""
    return pd.DataFrame({
        'id': [1, 2, 3, 4, 5, 6],
        'proc1': ['A', 'B', None, 'C', 'A', None],
        'proc2': ['C', 'A', None, None, 'B', None],
        'proc3': [None, 'B', None, 'C', 'A', None],
        'other_data': [10, 20, 30, 40, 50, 60]
    }, index=[100, 101, 102, 103, 104, 105]) # Use a distinct index

# === Tests for gen_dummies ===

class TestGenDummies:

    def test_basic_series_input_defaults(self, sample_series):
        """Test basic Series input with default options (process_na=True, float=True)."""
        result = gen_dummies(sample_series, prefix='f1')
        expected = pd.DataFrame({
            'f1_A': [1.0, 0.0, 1.0, np.nan, 0.0, 0.0],
            'f1_B': [0.0, 1.0, 0.0, np.nan, 0.0, 1.0],
            'f1_C': [0.0, 0.0, 0.0, np.nan, 1.0, 0.0],
        }, index=sample_series.index, dtype=float)
        pd.testing.assert_frame_equal(result, expected)

    def test_basic_dataframe_input_defaults(self, sample_dataframe_single_col):
        """Test basic single-column DataFrame input with defaults."""
        result = gen_dummies(sample_dataframe_single_col, prefix='f1')
        expected = pd.DataFrame({
            'f1_A': [1.0, 0.0, 1.0, np.nan, 0.0, 0.0],
            'f1_B': [0.0, 1.0, 0.0, np.nan, 0.0, 1.0],
            'f1_C': [0.0, 0.0, 0.0, np.nan, 1.0, 0.0],
        }, index=sample_dataframe_single_col.index, dtype=float)
        pd.testing.assert_frame_equal(result, expected)

    def test_process_na_false(self, sample_series):
        """Test with process_na=False and convert_to_float=False."""
        result = gen_dummies(sample_series, prefix='f1', process_na=False, convert_to_float=False)
        # Expect integer type (often uint8 from get_dummies, but check result)
        expected = pd.DataFrame({
            'f1_A': [1, 0, 1, 0, 0, 0],
            'f1_B': [0, 1, 0, 0, 0, 1],
            'f1_C': [0, 0, 0, 0, 1, 0],
        }, index=sample_series.index).astype(result.dtypes.iloc[0]) # Match the exact integer type returned
        pd.testing.assert_frame_equal(result, expected)

    def test_process_na_false_convert_true(self, sample_series):
        """Test with process_na=False and convert_to_float=True."""
        result = gen_dummies(sample_series, prefix='f1', process_na=False, convert_to_float=True)
        expected = pd.DataFrame({
            'f1_A': [1.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            'f1_B': [0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            'f1_C': [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        }, index=sample_series.index, dtype=float)
        pd.testing.assert_frame_equal(result, expected)

    def test_convert_to_float_false_no_na(self):
        """Test convert_to_float=False with data having no NaNs."""
        series_no_nan = pd.Series(['A', 'B', 'A', 'B', 'C'], index=[1, 2, 3, 4, 5])
        result = gen_dummies(series_no_nan, prefix='f1', process_na=False, convert_to_float=False)
        assert not pd.api.types.is_float_dtype(result.dtypes.iloc[0])
        # Check if it's integer or boolean type
        assert pd.api.types.is_integer_dtype(result.dtypes.iloc[0]) or pd.api.types.is_bool_dtype(result.dtypes.iloc[0])
        expected = pd.DataFrame({
            'f1_A': [1, 0, 1, 0, 0],
            'f1_B': [0, 1, 0, 1, 0],
            'f1_C': [0, 0, 0, 0, 1],
        }, index=series_no_nan.index).astype(result.dtypes.iloc[0]) # Match exact type
        pd.testing.assert_frame_equal(result, expected)

    def test_value_error_na_true_float_false(self, sample_series):
        """Test ValueError is raised for invalid flag combination."""
        with pytest.raises(ValueError, match="process_na=True requires convert_to_float=True"):
            gen_dummies(sample_series, prefix='f1', process_na=True, convert_to_float=False)

    def test_value_error_multi_column_dataframe(self, sample_dataframe_multi_col):
        """Test ValueError is raised for multi-column DataFrame input."""
        with pytest.raises(ValueError, match="Input 'feature_data' must be a pandas Series or a single-column DataFrame."):
            gen_dummies(sample_dataframe_multi_col[['proc1', 'proc2']], prefix='f1') # Pass >1 column

    def test_type_error_invalid_input(self):
        """Test TypeError for non-Series/DataFrame input."""
        with pytest.raises(TypeError, match="Input 'feature_data' must be a pandas Series or DataFrame."):
            gen_dummies([1, 2, 3], prefix='f1') # Pass a list

    def test_empty_input_series(self):
        """Test with an empty Series."""
        s = pd.Series([], dtype=object, index=pd.Index([], name='empty_idx'))
        result = gen_dummies(s, prefix='empty')
        assert result.empty
        assert result.index.equals(s.index)
        assert isinstance(result, pd.DataFrame)

    def test_empty_input_dataframe(self):
        """Test with an empty single-column DataFrame."""
        df = pd.DataFrame({'col': []}, index=pd.Index([], name='empty_idx'))
        result = gen_dummies(df, prefix='empty')
        assert result.empty
        assert result.index.equals(df.index)
        assert isinstance(result, pd.DataFrame)

    # def test_input_with_only_nans_series(self):
    #     """Test Series containing only NaNs."""
    #     s = pd.Series([None, np.nan, None], dtype=object, index=[1, 2, 3])
    #     result = gen_dummies(s, prefix='nan_only', process_na=True)
    #     # Expect an empty DataFrame as no non-NaN categories exist to create columns
    #     # The internal _nan column is created and then dropped.
    #     expected = pd.DataFrame(index=s.index, dtype=float)
    #     pd.testing.assert_frame_equal(result, expected)

    #     result_no_process = gen_dummies(s, prefix='nan_only', process_na=False, convert_to_float=False)
    #     # Expect empty DataFrame as no categories found and dummy_na=False internally
    #     expected_no_process = pd.DataFrame(index=s.index)
    #     pd.testing.assert_frame_equal(result_no_process, expected_no_process, check_dtype=False)


# === Tests for gen_dummies_from_combined_columns ===

class TestGenDummiesFromCombinedColumns:

    def test_basic_combination_defaults(self, sample_dataframe_multi_col):
        """Test combining columns with default options (process_na=True, float=True)."""
        cols = ['proc1', 'proc2', 'proc3']
        result = gen_dummies_from_combined_columns(sample_dataframe_multi_col, cols, prefix='proc')
        expected = pd.DataFrame({
            # Row 102 (index 2): All proc are None -> NaN
            # Row 105 (index 5): All proc are None -> NaN
            'proc_A': [1.0, 1.0, np.nan, 0.0, 1.0, np.nan],
            'proc_B': [0.0, 1.0, np.nan, 0.0, 1.0, np.nan],
            'proc_C': [1.0, 0.0, np.nan, 1.0, 0.0, np.nan],
        }, index=sample_dataframe_multi_col.index, dtype=float)
        pd.testing.assert_frame_equal(result, expected)

    def test_combination_process_na_false_float_false(self, sample_dataframe_multi_col):
        """Test combining columns with process_na=False and convert_to_float=False."""
        cols = ['proc1', 'proc2', 'proc3']
        result = gen_dummies_from_combined_columns(sample_dataframe_multi_col, cols, prefix='proc',
                                                  process_na=False, convert_to_float=False)
        # Expect integer type (e.g., int64 from groupby.max, or uint8/int if converted)
        expected = pd.DataFrame({
            # Row 102 (index 2): All proc are None -> 0
            # Row 105 (index 5): All proc are None -> 0
            'proc_A': [1, 1, 0, 0, 1, 0],
            'proc_B': [0, 1, 0, 0, 1, 0],
            'proc_C': [1, 0, 0, 1, 0, 0],
        }, index=sample_dataframe_multi_col.index).astype(result.dtypes.iloc[0]) # Match exact type
        pd.testing.assert_frame_equal(result, expected)

    def test_combination_process_na_false_float_true(self, sample_dataframe_multi_col):
        """Test combining columns with process_na=False and convert_to_float=True."""
        cols = ['proc1', 'proc2', 'proc3']
        result = gen_dummies_from_combined_columns(sample_dataframe_multi_col, cols, prefix='proc',
                                                  process_na=False, convert_to_float=True)
        expected = pd.DataFrame({
            'proc_A': [1.0, 1.0, 0.0, 0.0, 1.0, 0.0],
            'proc_B': [0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
            'proc_C': [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        }, index=sample_dataframe_multi_col.index, dtype=float)
        pd.testing.assert_frame_equal(result, expected)


    def test_empty_columns_to_combine(self, sample_dataframe_multi_col):
        """Test providing an empty list for columns_to_combine."""
        with pytest.warns(UserWarning, match="'columns_to_combine' is empty"):
            result = gen_dummies_from_combined_columns(sample_dataframe_multi_col, [], prefix='proc')
        assert result.empty
        assert result.index.equals(sample_dataframe_multi_col.index)
        assert isinstance(result, pd.DataFrame)

    def test_columns_with_only_nans_rows(self, sample_dataframe_multi_col):
        """Test rows where the selected columns only contain NaNs."""
        # Row 102 and 105 have only NaNs in proc1, proc2, proc3
        cols = ['proc1', 'proc2', 'proc3']
        result_na_true = gen_dummies_from_combined_columns(sample_dataframe_multi_col, cols, prefix='proc', process_na=True)
        assert result_na_true.loc[102].isnull().all()
        assert result_na_true.loc[105].isnull().all()
        assert not result_na_true.loc[100].isnull().any() # Row 100 should not be all NaN

        result_na_false = gen_dummies_from_combined_columns(sample_dataframe_multi_col, cols, prefix='proc', process_na=False)
        assert (result_na_false.loc[102] == 0).all()
        assert (result_na_false.loc[105] == 0).all()
        assert not (result_na_false.loc[100] == 0).all() # Row 100 should have some 1s

    def test_empty_input_dataframe(self):
        """Test with an empty input DataFrame."""
        df_empty = pd.DataFrame({'proc1': [], 'proc2': []}, index=pd.Index([], name='empty_idx'))
        result = gen_dummies_from_combined_columns(df_empty, ['proc1', 'proc2'], prefix='p')
        assert result.empty
        assert result.index.equals(df_empty.index)
        assert isinstance(result, pd.DataFrame)

    def test_value_error_na_true_float_false(self, sample_dataframe_multi_col):
        """Test ValueError for invalid flag combination."""
        with pytest.raises(ValueError, match="process_na=True requires convert_to_float=True"):
            gen_dummies_from_combined_columns(sample_dataframe_multi_col, ['proc1'], prefix='p',
                                              process_na=True, convert_to_float=False)

    def test_key_error_invalid_column(self, sample_dataframe_multi_col):
        """Test KeyError when a column in columns_to_combine doesn't exist."""
        with pytest.raises(KeyError, match="not found in DataFrame"):
            gen_dummies_from_combined_columns(sample_dataframe_multi_col, ['proc1', 'invalid_col'], prefix='p')

    def test_type_error_df_not_dataframe(self):
        """Test TypeError when df is not a DataFrame."""
        with pytest.raises(TypeError, match="Input 'df' must be a pandas DataFrame."):
            gen_dummies_from_combined_columns(['a'], ['col'], prefix='p')

    def test_type_error_columns_not_list(self, sample_dataframe_multi_col):
        """Test TypeError when columns_to_combine is not a list."""
        with pytest.raises(TypeError, match="Input 'columns_to_combine' must be a list"):
            gen_dummies_from_combined_columns(sample_dataframe_multi_col, 'proc1', prefix='p')

    def test_different_prefix_sep(self, sample_dataframe_multi_col):
        """Test using a different prefix_sep."""
        cols = ['proc1', 'proc2'] # Use subset for simplicity
        result = gen_dummies_from_combined_columns(sample_dataframe_multi_col, cols, prefix='proc', prefix_sep='::')
        # Expected columns based on values in proc1/proc2: A, B, C
        expected_cols = ['proc::A', 'proc::B', 'proc::C']
        assert all(col in result.columns for col in expected_cols)
        assert result.shape[1] == len(expected_cols) # Ensure no extra columns

    def test_no_non_nan_data_in_columns(self):
        """Test case where selected columns exist but contain only NaNs."""
        df = pd.DataFrame({'c1': [None, None], 'c2': [np.nan, np.nan]}, index=[5, 6])
        result_na_true = gen_dummies_from_combined_columns(df, ['c1', 'c2'], prefix='p', process_na=True)
        # Expect empty df with correct index, as no categories found, but NaNs processed
        expected_na_true = pd.DataFrame(index=df.index, dtype=float)
        pd.testing.assert_frame_equal(result_na_true, expected_na_true)

        result_na_false = gen_dummies_from_combined_columns(df, ['c1', 'c2'], prefix='p', process_na=False)
        # Expect empty df with correct index and 0 columns
        expected_na_false = pd.DataFrame(index=df.index)
        pd.testing.assert_frame_equal(result_na_false, expected_na_false, check_dtype=False)
