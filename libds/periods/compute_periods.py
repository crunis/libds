from . import fill_gaps

def compute_periods(ordinals, values, mode="true_between", prefix=None, extra_stats=False):
    """
    Compute the periods and statistics based on the given ordinals and values.
    Args:
        ordinals (list): A list of integers representing the ordinals.
        values (list): A list of booleans representing the values.
    Returns:
        dict: A dictionary containing the following statistics:
            - days (int): The total number of positive days.
            - periods (int): The total number of periods.
            - max_consec_days (int): The maximum number of consecutive positive days.
    
    If there are gaps, they are filled with True only between Trues.
    """

    ordinals, values = fill_gaps(ordinals, values, mode=mode)

    prev_ordinal = ordinals[0] - 1
    days = 0
    periods = 0
    curr_state = False
    current_interval_days = 0
    max_consec_days = 0
    days_since_last_interval = 0

    interval_starts = []
    interval_ends = []
    interval_days = []
    interval_days_since_last = []

    for ordinal, value in zip(ordinals, values):
        if value:
            days += 1
            current_interval_days += 1
            if value != curr_state:
                # SWITCH to Positive
                periods += 1
                interval_starts.append(ordinal)
                interval_days_since_last.append(days_since_last_interval)
                days_since_last_interval = 0
        else:
            days_since_last_interval += 1
            if value != curr_state:
                # SWITCH to Negative
                interval_ends.append(prev_ordinal)
                interval_days.append(current_interval_days)
                current_interval_days = 0

        curr_state = value
        prev_ordinal = ordinal
        max_consec_days = max(max_consec_days, current_interval_days)

    if curr_state:
        interval_ends.append(prev_ordinal)
        interval_days.append(current_interval_days)

    prefix = f"{prefix}_" if prefix else ""

    res = {
            f"{prefix}days": days,
            f"{prefix}periods": periods,
            f"{prefix}max_consec_days": max_consec_days,
            }
    
    if extra_stats:
        res[f"{prefix}interval_starts"] = interval_starts
        res[f"{prefix}interval_ends"] = interval_ends
        res[f"{prefix}interval_days"] = interval_days
        res[f"{prefix}interval_days_since_last"] = interval_days_since_last
        
    return res
