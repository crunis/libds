from . import fill_gaps
from . import compute_periods

import numpy as np

def compute_periods_with_margins(
    ordinals, 
    values, 
    initial_margin=0, 
    cooloff=0):
    data = compute_periods(ordinals, values, extra_stats=True)

    # Initial margin
    if data['interval_days_since_last'][0] < initial_margin:
        # Substract the first interval days from total days
        data['days'] -= data['interval_days'][0]
        data['periods'] -= 1
        # Remove first interval from data
        data['interval_starts'] = data['interval_starts'][1:]
        data['interval_ends'] = data['interval_ends'][1:]
        data['interval_days'] = data['interval_days'][1:]
        data['interval_days_since_last'] = data['interval_days_since_last'][1:]
        
        data['max_consec_days'] = np.max(data['interval_days']) if len(data['interval_days']) > 0 else 0
    

    if cooloff>0:
        for i in range(len(data['interval_days_since_last'])-1, 0, -1):
            if data['interval_days_since_last'][i] <= cooloff:
                data['periods'] -= 1
                data['interval_days'][i-1] += data['interval_days'][i] + data['interval_days_since_last'][i]
                del data['interval_starts'][i]
                del data['interval_ends'][i-1]
                del data['interval_days_since_last'][i]
                del data['interval_days'][i]

    return data 
