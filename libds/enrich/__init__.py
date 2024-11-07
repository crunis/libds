import pandas as pd

from libds.periods import compute_periods, dates_to_ordinal
from .lab import compute_all_penias
from .rc import compute_rc
from .demo import add_exitus_info, add_age
from .admission import get_admission, get_admission_id