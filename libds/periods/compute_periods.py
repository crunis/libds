def compute_periods(ordinals, values, prefix=None, extra_stats=False):
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
    """
    last_ordinal = ordinals[0] - 1
    days = 0  # total days positive
    periods = 0
    curr_state = False
    consec_days = 0
    max_consec_days = 0
    days_since_last = 0
    interval_days = []
    interval_starts = []
    interval_ends = []

    for ordinal, value in zip(ordinals, values):
        if value:
            if curr_state:
                # Add interval only if previous was True
                days += ordinal - last_ordinal
                consec_days += ordinal - last_ordinal
            else:
                days += 1
                consec_days = 1
                periods += 1
                curr_state = True
                interval_days.append(days_since_last)
                interval_starts.append(ordinal)
                days_since_last = 0
        else:
            days_since_last += 1
            if curr_state:
                interval_ends.append(last_ordinal)
                curr_state = False

        last_ordinal = ordinal
        max_consec_days = max(max_consec_days, consec_days)

    if curr_state:
        interval_ends.append(last_ordinal)

    prefix = f"{prefix}_" if prefix else ""

    res = {
            f"{prefix}days": days,
            f"{prefix}periods": periods,
            f"{prefix}max_consec_days": max_consec_days,
            }
    
    if extra_stats:
        res[f"{prefix}interval_days"] = interval_days
        res[f"{prefix}interval_starts"] = interval_starts
        res[f"{prefix}interval_ends"] = interval_ends

    return res
