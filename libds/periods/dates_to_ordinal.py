def dates_to_ordinal(datetimes):
    """
    Converts a list of dates to a list of ordinal values.
    """
    dates = [date.date() for date in datetimes]
    min_date = min(dates)
    ordinals = [(date - min_date).days for date in dates]

    return ordinals


def dates_to_ordinal_with_values(datetimes, values, drop_duplicates=True):
    ordinals = dates_to_ordinal(datetimes)

    seen = set()
    uniq_ordinals = []
    uniq_values = []
    if drop_duplicates:
        for i in range(len(ordinals)):
            if ordinals[i] not in seen:
                seen.add(ordinals[i])
                uniq_ordinals.append(ordinals[i])
                if values:
                    uniq_values.append(values[i])

    return uniq_ordinals, uniq_values
