def week(_dts, _dt):
    """
    Checks if a given date is within a week (7 days) of the last date in a list of dates.

    Args:
        _dts: A list of datetime objects representing a group of dates.
        _dt: A datetime object to check against the last date in the list.

    Returns:
        True if the difference between _dt and the last date in _dts is within 7 days (inclusive), False otherwise.
    """
    return ((_dt - _dts[-1]).days + 1) <= 7


def group(_dts, condition=week):
    """
    Groups a list of items based on a given condition.

    Args:
        _dts: A list of items (e.g., datetime objects, numbers) to be grouped.
        condition: A function that takes a list (representing the current group) and a single item as input. 
                   It returns True if the item should be added to the current group, False otherwise. 
                   Defaults to the 'week' function.

    Returns:
        A list of lists, where each inner list represents a group of items that satisfy the given condition.
    """
    res = list()
    cur = list()

    for _dt in _dts:
        if not cur:
            cur = [_dt]
            continue

        if condition(cur, _dt):
            cur.append(_dt)
            continue

        res.append(cur)
        cur = [_dt]

    if cur:
        res.append(cur)
    
    return res