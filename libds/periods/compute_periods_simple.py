def compute_periods_simple(ordinals, prefix=""):
    """
    Compute the number of periods, total days, and maximum consecutive days in a list of ordinals.

    Parameters:
    ordinals (list): A list of integers representing ordinals.

    Returns:
    dict: A dictionary containing the following keys:
        - days (int): The total number of days.
        - periods (int): The number of periods.
        - max_consec_days (int): The maximum number of consecutive days.

    Example:
    >>> compute_periods_simple([1, 2, 3, 5, 6, 8, 9])
    {'days': 7, 'periods': 4, 'max_consec_days': 3}
    """
    periods = 0
    days = 0
    max_consec_days = 0
    curr_consec_days = 0
    prev_ordinal = -100

    for i in ordinals:
        if i - prev_ordinal == 1:
            curr_consec_days += 1
        else:
            curr_consec_days = 1
            periods += 1
        days += 1
        prev_ordinal = i
        max_consec_days = max(max_consec_days, curr_consec_days)

    return {f"{prefix}days": days, f"{prefix}periods": periods, f"{prefix}max_consec_days": max_consec_days}
