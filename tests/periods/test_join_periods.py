import pandas as pd

from libds.periods import join_periods, join_up_to_a_day


def test_join_periods_simple():
    periods = [ ['2023-01-01', '2023-01-02'], 
                ['2023-01-02', '2023-01-03'] ]

    joined_periods, metadata, stats = join_periods(periods)
    
    assert joined_periods == [['2023-01-01', '2023-01-03']]
    assert metadata['episode_idxs'] == [[0, 1]]
    assert metadata['edge_episode_idx'] == [(0, 1)]
    assert stats == {}

    periods2 = [ ['2023-01-01', '2023-01-02'], 
                ['2023-01-02', '2023-01-03'],
                ['2023-01-03', '2023-01-04'] ]
    
    joined_periods, metadata, stats = join_periods(periods2)
    
    assert joined_periods == [['2023-01-01', '2023-01-04']]
    assert metadata['episode_idxs'] == [[0, 1, 2]]
    assert metadata['edge_episode_idx'] == [(0, 2)]
    assert stats == {}

    periods3 = [ ['2023-01-01', '2023-01-02'], 
                ['2023-01-02', '2023-01-03'],
                ['2023-01-05', '2023-01-07'] ]
    
    joined_periods, metadata, stats = join_periods(periods3)
    
    assert joined_periods == [
        ['2023-01-01', '2023-01-03'],
        ['2023-01-05', '2023-01-07']
    ]
    assert metadata['episode_idxs'] == [[0, 1], [2]]
    assert metadata['edge_episode_idx'] == [(0, 1), (2, 2)]
    assert stats == {}


def test_join_periods_custom_function():
    d1 = pd.to_datetime('2023-01-01 13:00:00')
    d2 = pd.to_datetime('2023-01-02 13:00:00')
    d3 = pd.to_datetime('2023-01-03 10:00:00')
    d4 = pd.to_datetime('2023-01-04 10:00:00')
    periods = [ [d1, d2], [d3, d4] ]

    joined_periods, metadata, stats = join_periods(periods, join_condition=join_up_to_a_day)
    
    assert joined_periods == [[d1, d4]]
    assert metadata['episode_idxs'] == [[0, 1]]
    assert metadata['edge_episode_idx'] == [(0, 1)]
    assert stats == {}

    d1 = pd.to_datetime('2023-01-01 13:00:00')
    d2 = pd.to_datetime('2023-01-02 13:00:00')
    d3 = pd.to_datetime('2023-01-03 15:00:00')
    d4 = pd.to_datetime('2023-01-04 10:00:00')
    periods = [ [d1, d2], [d3, d4] ]

    joined_periods, metadata, stats = join_periods(periods, join_condition=join_up_to_a_day)
    
    assert joined_periods == [[d1, d2], [d3, d4]]
    assert metadata['episode_idxs'] == [[0], [1]]
    assert metadata['edge_episode_idx'] == [(0, 0), (1, 1)]
    assert stats == {}


def test_join_overlap():
    d1 = pd.to_datetime('2023-01-01 13:00:00')
    d2 = pd.to_datetime('2023-01-03 13:00:00')
    d3 = pd.to_datetime('2023-01-02 10:00:00')
    d4 = pd.to_datetime('2023-01-04 10:00:00')
    periods = [ [d1, d2], [d3, d4] ]

    joined_periods, metadata, stats = join_periods(periods)
    
    assert joined_periods == [[d1, d4]]
    assert metadata['episode_idxs'] == [[0, 1]]
    assert stats == {'overlap': 1}


def test_join_subset():
    d1 = pd.to_datetime('2023-01-01 13:00:00')
    d2 = pd.to_datetime('2023-01-04 10:00:00')
    d3 = pd.to_datetime('2023-01-02 13:00:00')
    d4 = pd.to_datetime('2023-01-03 10:00:00')

    periods = [ [d1, d2], [d3, d4] ]

    joined_periods, metadata, stats = join_periods(periods)
    
    assert joined_periods == [[d1, d2]]
    assert metadata['episode_idxs'] == [[0, 1]]
    assert stats == {'subset': 1}