import pandas as pd

from libds.enrich import add_age

def test_add_age():
    res = add_age(pd.Timestamp("1979-08-22"), pd.Timestamp("2025-02-11"))
    assert (res == pd.Series({'age': 45})).all()


def test_empty_age():
    res = add_age(None, pd.Timestamp("2025-02-11"))
    assert res.empty

    res = add_age(float('Nan'), pd.Timestamp("2025-02-11"))
    assert res.empty
