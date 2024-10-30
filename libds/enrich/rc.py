import pandas as pd


RCS_MAX = "TEMP_AXI FC FREC_RESP".split()
RCS_MIN = "PULSIOX".split()

def compute_rc(df_rc, pid, start_dt, end_dt, rc_max, rc_min):
    rc_subset = df_rc[
        (df_rc.pid == pid)
        & (df_rc._dt >= start_dt)
        & (df_rc._dt <= end_dt)
    ]

    rcs_max =   {
                f"{rc}_max": rc_subset.loc[rc_subset.type == rc, 'value'].max() for rc in rc_max
                }
    rcs_min =   {
                f"{rc}_min": rc_subset.loc[rc_subset.type == rc, 'value'].min() for rc in rc_min
                }

    return pd.Series(rcs_max | rcs_min)