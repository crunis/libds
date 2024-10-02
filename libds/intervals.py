def intervals_select_by_interval(df, pid, start_dt, end_dt, contained=False):
    if contained:
        return df[(df.pid == pid) & (df['start_dt'] >= start_dt) & (df['end_dt'] <= end_dt  )]
    else:
        return df[(df.pid == pid) & (df['start_dt'] <= end_dt  ) & (df['end_dt'] >= start_dt)]


def contains_interval(df, pid, start_dt, end_dt, contained=False):
    res = intervals_select_by_interval(df, pid, start_dt, end_dt, contained)
    return res.shape[0] > 0
