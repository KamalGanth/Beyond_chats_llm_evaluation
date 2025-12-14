import time #for latency measurement

def time_ms(func, *args, **kwargs):
    t0 = time.time()
    out = func(*args, **kwargs)
    t1 = time.time()
    return out, int((t1 - t0) * 1000)
