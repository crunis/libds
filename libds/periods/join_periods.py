from joblib import Parallel, delayed
from collections import defaultdict
import pandas as pd


def default_join_periods_join_condition(item1, item2):
    return item1[1] == item2[0]


def join_up_to_a_day(period1, period2):
    seconds = (period2[0] - period1[1]).total_seconds()
    if seconds <= 86400:
        return True
    return False


def join_periods(
        periods,
        errors = 'raise',
        join_condition = default_join_periods_join_condition
        ):
    """
    Enhanced version that tracks first episode, last episode (by boundary),
    and all constituent episodes

    Returns:
        joined_periods: list of [start, end] periods
        correspondences: list of all episode indices in each admission
        first_episode_indices: list of indices of chronologically first episodes
        last_episode_indices: list of indices of episodes that determine end boundary
        stats: dictionary with joining statistics
    """
    current_period = None
    joined_periods = list()
    correspondences = list()
    first_episode_indices = list()
    last_episode_indices = list()
    current_correspondence = list()
    current_first_idx = None
    current_last_idx = None
    stats = defaultdict(int)

    for n, period in enumerate(periods):
        if isinstance(period, tuple):
            period = list(period)

        # First period
        if current_period is None:
            current_period = period
            current_correspondence.append(n)
            current_first_idx = n
            current_last_idx = n
            continue

        if period[0] < current_period[1]:
            if period[1] <= current_period[1]:
                # subset, ignore for boundary but keep in correspondence
                stats["subset"] += 1
                current_correspondence.append(n)
                # Don't update current_last_idx since this episode doesn't extend the boundary
                continue
            else:
                # overlap - this episode extends the end boundary
                stats["overlap"] += 1
                current_period[1] = period[1]
                current_correspondence.append(n)
                current_last_idx = n  # This episode now determines the end
                continue

        # Join periods
        if join_condition(current_period, period):
            current_period[1] = period[1]
            current_correspondence.append(n)
            current_last_idx = n  # This episode determines the new end
            continue

        # New period - save current and start new
        joined_periods.append(current_period)
        correspondences.append(current_correspondence)
        first_episode_indices.append(current_first_idx)
        last_episode_indices.append(current_last_idx)

        # Start new admission
        current_period = period
        current_correspondence = [n]
        current_first_idx = n
        current_last_idx = n

    # Don't forget the last admission
    joined_periods.append(current_period)
    correspondences.append(current_correspondence)
    first_episode_indices.append(current_first_idx)
    last_episode_indices.append(current_last_idx)

    return joined_periods, correspondences, first_episode_indices, last_episode_indices, dict(stats)


def df_join_periods_loop(
        pid,
        df,
        start_field,
        end_field,
        join_condition,
        fields_from_first=None,
        fields_from_last=None,
        fields_to_concatenate=None
        ):
    """
    Enhanced version with customizable field extraction

    Parameters:
        fields_from_first: list of field names to take from first episode
        fields_from_last: list of field names to take from last episode (boundary-determining)
        fields_to_concatenate: list of field names to concatenate from all episodes
    """
    # Default field configurations
    if fields_from_first is None:
        fields_from_first = ['start_date', 'start_month']
    if fields_from_last is None:
        fields_from_last = ['end_date', 'end_month']
    if fields_to_concatenate is None:
        fields_to_concatenate = ['type']

    # Reset index to ensure we can map back correctly
    df_sorted = df.sort_values(start_field).reset_index(drop=True)

    # Get the joined periods and correspondences
    res = join_periods(
            zip(df_sorted[start_field].tolist(), df_sorted[end_field].tolist()),
            join_condition=join_condition
            )

    joined_periods, correspondences, first_episode_indices, last_episode_indices, stats = res

    # Build the result dataframe with all original fields
    result_rows = []

    for i, (period, episode_indices, first_idx, last_idx) in enumerate(
        zip(joined_periods, correspondences, first_episode_indices, last_episode_indices)
    ):
        # Get all episodes that make up this admission
        constituent_episodes = df_sorted.iloc[episode_indices]
        first_episode = df_sorted.iloc[first_idx]
        last_episode = df_sorted.iloc[last_idx]

        # Create the admission row
        admission_row = {}

        # Basic fields
        admission_row['pid'] = pid
        admission_row['admission_id'] = i + 1

        # Time fields from the joined period (computed boundaries)
        admission_row[start_field] = period[0]
        admission_row[end_field] = period[1]

        # Fields from first episode (chronologically)
        for field in fields_from_first:
            if field in first_episode:
                admission_row[f'first_{field}'] = first_episode[field]

        # Fields from last episode (boundary-determining)
        for field in fields_from_last:
            if field in last_episode:
                admission_row[f'last_{field}'] = last_episode[field]

        # Concatenated fields from all episodes
        for field in fields_to_concatenate:
            if field in constituent_episodes.columns:
                values = constituent_episodes[field].astype(str).tolist()
                admission_row[f'all_{field}'] = '|'.join(values)

        # Metadata about the admission
        admission_row['num_episodes'] = len(episode_indices)
        admission_row['first_episode_id'] = first_episode.get('eid', first_episode.get('_id', first_idx))
        admission_row['last_episode_id'] = last_episode.get('eid', last_episode.get('_id', last_idx))
        admission_row['all_episode_ids'] = '|'.join(constituent_episodes.get('eid', constituent_episodes.get('_id', episode_indices)).astype(str))

        # Indices for debugging/tracing
        admission_row['first_episode_idx'] = first_idx
        admission_row['last_episode_idx'] = last_idx
        admission_row['all_episode_indices'] = '|'.join(map(str, episode_indices))

        result_rows.append(admission_row)

    return pd.DataFrame(result_rows)


def df_join_periods(
        df: pd.DataFrame,
        start_field='start_dt',
        end_field='end_dt',
        join_condition=default_join_periods_join_condition,
        fields_from_first=None,
        fields_from_last=None,
        fields_to_concatenate=None,
        n_jobs=None,
        ):
    """
    Enhanced version that creates admissions with customizable field extraction

    Parameters:
        df: Input DataFrame with episodes
        start_field: Name of start datetime field
        end_field: Name of end datetime field
        join_condition: Function to determine if periods should be joined
        fields_from_first: List of fields to extract from chronologically first episode
        fields_from_last: List of fields to extract from boundary-determining last episode
        fields_to_concatenate: List of fields to concatenate from all constituent episodes
        n_jobs: Number of parallel jobs

    Returns:
        DataFrame with admissions
    """

    if n_jobs is None:
        dfs = [
            df_join_periods_loop(
                pid, data, start_field, end_field, join_condition,
                fields_from_first, fields_from_last, fields_to_concatenate
            )
            for pid, data in df.groupby('pid')
        ]
    else:
        dfs = Parallel(n_jobs=n_jobs)(
            delayed(df_join_periods_loop)(
                pid, data, start_field, end_field, join_condition,
                fields_from_first, fields_from_last, fields_to_concatenate
            )
            for pid, data in df.groupby('pid')
        )

    result_df = pd.concat(dfs, ignore_index=True)

    return result_df


def df_join_periods_with_detailed_mapping(
        df: pd.DataFrame,
        start_field='start_dt',
        end_field='end_dt',
        join_condition=default_join_periods_join_condition,
        fields_from_first=None,
        fields_from_last=None,
        fields_to_concatenate=None,
        n_jobs=None,
        ):
    """
    Returns admissions DataFrame and detailed mapping showing episode roles

    Returns:
        admissions_df: DataFrame with joined admissions
        mapping_df: DataFrame showing which episodes belong to which admissions
                   and their roles (first, last, constituent, subset, etc.)
    """

    def process_patient_with_detailed_mapping(pid, patient_df):
        df_sorted = patient_df.sort_values(start_field).reset_index()

        # Get enhanced joining results
        res = join_periods(
                zip(df_sorted[start_field].tolist(), df_sorted[end_field].tolist()),
                join_condition=join_condition
                )

        joined_periods, correspondences, first_episode_indices, last_episode_indices, stats = res

        # Create admissions dataframe
        admission_rows = []
        mapping_rows = []

        for admission_id, (period, episode_indices, first_idx, last_idx) in enumerate(
            zip(joined_periods, correspondences, first_episode_indices, last_episode_indices)
        ):
            constituent_episodes = df_sorted.iloc[episode_indices]
            first_episode = df_sorted.iloc[first_idx]
            last_episode = df_sorted.iloc[last_idx]

            # Admission row
            admission_row = {
                'pid': pid,
                'admission_n': admission_id + 1,
                start_field: period[0],
                end_field: period[1],
                'num_episodes': len(episode_indices),
#                 'first_episode_idx': first_idx,
#                 'last_episode_idx': last_idx
            }

            # Add custom fields
            for field in (fields_from_first or []):
                if field in first_episode:
                    admission_row[f'{field}'] = first_episode[field]

            for field in (fields_from_last or []):
                if field in last_episode:
                    admission_row[f'{field}'] = last_episode[field]

            for field in (fields_to_concatenate or []):
                if field in constituent_episodes.columns:
                    values = constituent_episodes[field].astype(str).tolist()
                    admission_row[f'{field}'] = '|'.join(values)

            admission_rows.append(admission_row)

            # Detailed mapping rows
            for episode_idx in episode_indices:
                episode = df_sorted.iloc[episode_idx]

                # Determine episode role
                roles = []
                if episode_idx == first_idx:
                    roles.append('first')
                if episode_idx == last_idx:
                    roles.append('last')
                if episode_idx in episode_indices:
                    roles.append('constituent')

                mapping_row = {
                    'pid': pid,
                    'admission_n': admission_id + 1,
                    'episode_idx': episode_idx,
                    'original_index': episode['index'],
                    'eid': episode.get('eid', episode_idx),
#                     '_id': episode.get('_id', episode_idx),
                    'roles': '|'.join(roles),
                    'is_first': episode_idx == first_idx,
                    'is_last': episode_idx == last_idx,
                    'episode_start': episode[start_field],
                    'episode_end': episode[end_field]
                }
                mapping_rows.append(mapping_row)

        return pd.DataFrame(admission_rows), pd.DataFrame(mapping_rows)

    if n_jobs is None:
        results = [
            process_patient_with_detailed_mapping(pid, data)
            for pid, data in df.groupby('pid')
        ]
    else:
        results = Parallel(n_jobs=n_jobs)(
            delayed(process_patient_with_detailed_mapping)(pid, data)
            for pid, data in df.groupby('pid')
        )

    # Separate admissions and mappings
    admission_dfs = [result[0] for result in results]
    mapping_dfs = [result[1] for result in results]

    admissions_df = pd.concat(admission_dfs, ignore_index=True)
    mapping_df = pd.concat(mapping_dfs, ignore_index=True)

    admissions_df.insert(0, 'admission_id', range(len(admissions_df)))
    mapping_df_w_admission_id = pd.merge(admissions_df[['admission_id', 'pid', 'admission_n']], mapping_df, on=['pid', 'admission_n'], how='right')

    return admissions_df, mapping_df_w_admission_id