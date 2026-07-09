import numpy as np

sigmas = [1.5, 2.5, 2, 2, 2.5]
thrs = [6, 1, 6, 6, 1]
area_sizes = [7, 3, 5, 5, 3]

print("sigma:", np.mean(sigmas))
print("thr:", np.mean(thrs))
print("area_size:", np.mean(area_sizes))