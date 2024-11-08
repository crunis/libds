from datetime import timedelta

def get_closest_event(df, pid, admission_id, _dt, days_before=30, days_after=0):
    end_dt = _dt + timedelta(days=days_after)
    # Up to days_before before
    start_dt = _dt - timedelta(days=days_before)

    df_events = df[
        (df.pid == pid)
        & (df._dt <= end_dt )
        & (
            (df.admission_id == admission_id)
            | (df._dt >= start_dt)
        )
    ].copy()

    if len(df_events) == 0:
        return None

    df_events['distance'] = abs((df_events._dt - _dt).dt.total_seconds())
    df_events.sort_values('distance', ascending=True, inplace=True)

    return df_events.iloc[0]