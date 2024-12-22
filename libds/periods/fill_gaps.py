# Filling mode: last. true_between, always_false, always_true
def fill_value(prev, curr, mode):
    if mode == 'always_false':
        return False
    elif mode == 'always_true':
        return True
    elif mode == 'last':
        return prev
    elif mode == 'true_between':
        if prev and curr:
            return True
        else:
            return False
    
    raise ValueError(f"Unknown mode {mode}")


def fill_gaps(ordinals, values, mode='true_between'):
    prev_o, prev_v = None, None

    res_o, res_v = [], []
    for o, v in zip(ordinals,values):
        # First element
        if prev_o is None:
            prev_o, prev_v = o, v
            res_o.append(o)
            res_v.append(v)
            continue

        # No gap
        if o == (prev_o + 1):
            res_o.append(o)
            res_v.append(v)
            prev_o, prev_v = o, v
            continue

        # Fill gap with previous values
        els = [el for el in range(prev_o + 1, o)]
        for el in els:
            res_o.append(el)
            res_v.append(fill_value(prev_v, v, mode))

        res_o.append(o)
        res_v.append(v)
        prev_o, prev_v = o, v

    return res_o, res_v