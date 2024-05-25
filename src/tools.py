import time
import numpy as np

def benchmark(fun, values, n_trials=10):
    benchmark_times = []

    for v in values:
        times = []
        for _ in range(n_trials):
            start = time.time_ns()
            fun(v)
            end = time.time_ns()
            times.append((end - start) / 1e6)

        benchmark_times.append(np.mean(times))

    return benchmark_times