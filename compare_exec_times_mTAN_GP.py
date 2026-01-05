import time
import numpy as np

times = []
for lc in selected_light_curves:  # N=50-100
    lc_times = []
    for _ in range(10):
        start = time.time()
        # [run inference/fitting]
        end = time.time()
        lc_times.append(end - start)
    times.append(np.median(lc_times))  # median more robust to outliers

mean_time = np.mean(times)
std_time = np.std(times)
