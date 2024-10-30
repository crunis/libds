import pandas as pd

from libds.periods import compute_periods, dates_to_ordinal


def internal_compute_periods(df, pid, start_dt, end_dt, threshold, prefix):
    selected_data = df[ (df.pid == pid) & (df._dt<=end_dt) & (df._dt>=start_dt) ]
    if (len(selected_data)==0):
        return dict()

    return compute_periods(
        dates_to_ordinal(selected_data._dt),
        (selected_data.value < threshold),
        prefix = prefix
    )


def compute_all_penias(df, pid, start_dt, end_dt):
    dft = df[df.desc == 'Neutròfils']
    # neutropenia
    res_neutropenia = internal_compute_periods(dft, pid, start_dt, end_dt, 0.5, 'neutropenia')
    # neutropenia_sever
    res_neutropenia_sever = internal_compute_periods(dft, pid, start_dt, end_dt, 0.1, 'neutropenia_sever')

    # limfocitopenia
    dft = df[df.desc == 'Limfòcits']
    res_limfocitopenia = internal_compute_periods(dft, pid, start_dt, end_dt, 1, 'limfopenia')
    
    # limfocitopenia_severa
    dft = df[df.desc == 'Limfòcits']
    res_limfocitopenia_sever = internal_compute_periods(dft, pid, start_dt, end_dt, 0.5, 'limfopenia_sever')

    return pd.Series(res_neutropenia | res_neutropenia_sever | res_limfocitopenia | res_limfocitopenia_sever)
