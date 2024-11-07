import pandas as pd

from libds.enrich import get_admission, get_admission_id

b_admission = pd.read_excel("tests/fixtures/b_admissions.xlsx")

def test_get_admission():
    res = get_admission(b_admission, 13, pd.Timestamp("2021-03-30"))
    
    assert res["pid"] == 13
    assert res["start_date"] == "2021-03-29"
    assert res["_id"] == 1

def test_get_admission_id():
    res = get_admission_id(b_admission, 13, pd.Timestamp("2021-03-30"))
    assert res == 1
    
    res = get_admission_id(b_admission, 13, pd.Timestamp("2019-03-30"), strict=False)
    assert res is None
    