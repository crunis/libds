from copy import deepcopy

def compute_periods(states, prefix="", interval_stats=True):
    # State
    prev_state = False
    interval = 0
    duration = 0
    # Results
    days = 0
    periods = 0
    intervals = []
    durations = []
    starts = []
    ends = []


    for i, curr_state in enumerate(states):
        if prev_state:
            if curr_state:
                # Keeps True
                days += 1
                duration += 1
            else:
                # Switch to False
                interval = 1
                durations.append(duration)
                ends.append(i-1)
        else:
            if curr_state:
                # Switch to True
                intervals.append(interval)
                days += 1
                periods += 1
                duration = 1
                starts.append(i)
            else:
                # Keeps False
                interval += 1

        prev_state = curr_state
    
    if curr_state:
        durations.append(duration)
        ends.append(i)

    result = {
        prefix + 'days': days,
        prefix + 'periods': periods,
        prefix + 'max_consec_days': max(durations) if durations else 0,
    }

    if interval_stats:
        result |= {
            prefix + 'intervals': intervals,
            prefix + 'durations': durations,
            prefix + 'starts': starts,
            prefix + 'ends': ends,
        }

    return result


def join_specific_periods(periods, n):
    if (n<0) or (n>len(periods['intervals'])-2):
        raise IndexError('period index out of range')
    
    periods = deepcopy(periods)

    periods['periods'] -= 1
    periods['durations'][n] += periods['durations'][n+1] + periods['intervals'][n+1]
    del periods['intervals'][n+1]
    del periods['durations'][n+1]
    del periods['starts'][n+1]
    del periods['ends'][n]

    return periods


def delete_period(periods, n):
    len_periods = len(periods['intervals'])
    if (n<0) or (n>len_periods-1):
        raise IndexError('period index out of range')
    
    periods = deepcopy(periods)

    if (n+1) < len_periods:
        periods['intervals'][n+1] += periods['intervals'][n] + periods['durations'][n]
    del periods['intervals'][n]
    del periods['durations'][n]
    del periods['starts'][n]
    del periods['ends'][n]
    periods['periods'] -= 1

    return periods



def join_periods_by_distance(periods, distance):
    for i in range(len(periods['intervals'])-1, 0, -1):
        if periods['intervals'][i] <= distance:
            periods = join_specific_periods(periods, i-1)

    return periods