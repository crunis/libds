import pandas as pd

def add_exitus_info(exitus_dt, ref_dt):
    """Compute days from reference date to exitus date and if it is within 30 or 60 days"""
    if exitus_dt is not None:
        days = (exitus_dt - ref_dt).days
        exitus_30d = (days <= 30)
        exitus_60d = (days <= 60)
        return pd.Series(dict(
            exitus_days = days,
            exitus_30d = exitus_30d,
            exitus_60d = exitus_60d
        ))
    return pd.Series()


def add_age(birth_dt, ref_dt):
    """Compute age in years from birth date to reference date"""
    if (birth_dt is None) or (pd.isna(birth_dt)):
        print("Warning, None passed as birth_date!")
        return pd.Series()

    return pd.Series({'age': int((ref_dt - birth_dt).days/365.2425) })
