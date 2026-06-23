import numpy as np

sigmas = [1.5, 2.5, 1.5, 3, 2.5]
thrs = [1.5, 2, 1, 5, 1.5]
area_sizes = [6, 10, 8, 8, 4]

print("sigma:", np.mean(sigmas))
print("thr:", np.mean(thrs))
print("area_size:", np.mean(area_sizes))