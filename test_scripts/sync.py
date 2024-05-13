import torch
import time


def show_time(func, a, gamma, beta, type='cppcuda', ntest=10):
    times = list()
    res = list()
    # GPU warm up
    for _ in range(10):
        if type == 'torch':
            func(a)
        elif type == 'cppcuda':
            func(a, gamma, beta)
        else:
            raise ValueError(f"Expected \'torch\' or \'cppcuda\', but got {type}")
    for _ in range(ntest):
        # sync the threads to get accurate cuda running time
        torch.cuda.synchronize(device="cuda:0")
        start_time = time.time()
        if type == 'torch':
            r = func(a)
        elif type == 'cppcuda':
            r = func(a, gamma, beta)
        else:
            raise ValueError(f"Expected \'torch\' or \'cppcuda\', but got {type}")
        torch.cuda.synchronize(device="cuda:0")
        end_time = time.time()

        times.append((end_time-start_time)*1e6)
        res.append(r)
    return times, res
