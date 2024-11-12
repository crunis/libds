from datetime import timedelta


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


def dates_fill_period(start_date, end_date):
    """
    Returns a list of dates between the start_date and end_date, inclusive.
    """
    days = (end_date.date() - start_date.date()).days
    return [start_date + timedelta(days=i) for i in range(days + 1)]


def df_fill_period(df):
    """
    Generates list of dates from start_dt to end_dt from all DataFrame rows
    Avoids duplicates and sorts them
    """
    dates = []
    for _, r in df.iterrows():
        dates += dates_fill_period(r.start_dt, r.end_dt)
    dates = list(set(dates))
    dates.sort()

    return dates


def unify_dates(datetimes, days=1):
    datetimes = datetimes.copy() # Avoid affecting original list
    datetimes.sort()
    dates = list(map(lambda x: x.date(), datetimes)) # Extract dates from datetimes
    # We always keep first date
    conditions = [True] + [ ((dates[i+1] - dates[i]).days>days) for i in range(len(dates)-1) ]
    datetimes_selected = [ datetime for (condition, datetime) in zip(conditions, datetimes) if condition ] 

    return datetimes_selected 
    