import pandas as pd


def prep_diags_df(df_diags):
    df_diags = df_diags.copy()
    df_diags['class'] = df_diags['class'].str.split(',')
    df_diags = df_diags.explode('class').reset_index(drop=True)
    df_diags['class'] = df_diags['class'].str.strip()
    
    return df_diags


def rename_cols(name, suffix):
    MAX_LENGTH = 60
    name2 = name
    if (len(suffix)+len(name)) > MAX_LENGTH:
        name2=name[:(MAX_LENGTH-2-len(suffix))]

    return f"a_{name2}{suffix}"


def get_diags(df_diags, pid, end_date, start_date=None, suffix=""):
    dft = df_diags
    
    cond = (dft.pid == pid) & (dft._dt <= end_date)
    if start_date:
        cond = cond & (dft._dt >= start_date)
    dft = dft[cond]

    result = pd.crosstab(dft['pid'], dft['class']).astype(bool).reset_index()
    if result.empty:
        return pd.Series()
    
    result = result.drop(columns=['pid'])
    
    return result.iloc[0].rename(index=lambda r: rename_cols(r, suffix))

