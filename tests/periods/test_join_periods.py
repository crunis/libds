import pandas as pd

from libds.periods import join_periods, join_up_to_a_day


def test_join_periods_simple():
    periods = [ ['2023-01-01', '2023-01-02'], 
                ['2023-01-02', '2023-01-03'] ]

    assert join_periods(periods) == (
        [['2023-01-01', '2023-01-03']],
        [[0, 1]],
        {}
    )

    periods2 = [ ['2023-01-01', '2023-01-02'], 
                ['2023-01-02', '2023-01-03'],
                ['2023-01-03', '2023-01-04'] ]
    
    assert join_periods(periods2) == (
        [['2023-01-01', '2023-01-04']],
        [[0, 1, 2]],
        {}
    )

    periods3 = [ ['2023-01-01', '2023-01-02'], 
                ['2023-01-02', '2023-01-03'],
                ['2023-01-05', '2023-01-07'] ]
    
    assert join_periods(periods3) == (
        [
            ['2023-01-01', '2023-01-03'],
            ['2023-01-05', '2023-01-07']
        ],
        [
            [0, 1],
            [2]
        ],
        {}
    )


def test_join_periods_custom_function():
    d1 = pd.to_datetime('2023-01-01 13:00:00')
    d2 = pd.to_datetime('2023-01-02 13:00:00')
    d3 = pd.to_datetime('2023-01-03 10:00:00')
    d4 = pd.to_datetime('2023-01-04 10:00:00')
    periods = [ [d1, d2], [d3, d4] ]

    assert join_periods(periods, join_condition=join_up_to_a_day) == (
        [[d1, d4]], [[0, 1]], {}
    )

    d1 = pd.to_datetime('2023-01-01 13:00:00')
    d2 = pd.to_datetime('2023-01-02 13:00:00')
    d3 = pd.to_datetime('2023-01-03 15:00:00')
    d4 = pd.to_datetime('2023-01-04 10:00:00')
    periods = [ [d1, d2], [d3, d4] ]

    assert join_periods(periods, join_condition=join_up_to_a_day) == (
        [[d1, d2], [d3,d4]], [[0], [1]], {}
    )


def test_join_overlap():
    d1 = pd.to_datetime('2023-01-01 13:00:00')
    d2 = pd.to_datetime('2023-01-03 13:00:00')
    d3 = pd.to_datetime('2023-01-02 10:00:00')
    d4 = pd.to_datetime('2023-01-04 10:00:00')
    periods = [ [d1, d2], [d3, d4] ]

    res = join_periods(periods) 
    
    assert res[0] == [[d1, d4]]
    assert res[1] == [[0, 1]]
    assert res[2] == {'overlap': 1}


def test_join_subset():
    d1 = pd.to_datetime('2023-01-01 13:00:00')
    d2 = pd.to_datetime('2023-01-04 10:00:00')
    d3 = pd.to_datetime('2023-01-02 13:00:00')
    d4 = pd.to_datetime('2023-01-03 10:00:00')

    periods = [ [d1, d2], [d3, d4] ]

    res = join_periods(periods) 
    
    assert res[0] == [[d1, d2]]
    assert res[1] == [[0, 1]]
    assert res[2] == {'subset': 1}
