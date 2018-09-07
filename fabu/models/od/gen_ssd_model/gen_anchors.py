import math

min_dim = 400
min_ratio = 20
max_ratio = 90
step = int(math.floor((max_ratio - min_ratio) / (6 - 2)))
min_sizes = []
max_sizes = []
for ratio in xrange(min_ratio, max_ratio + 1, step):
  min_sizes.append(min_dim * ratio / 100.)
  max_sizes.append(min_dim * (ratio + step) / 100.)
min_sizes = [min_dim * 10 / 100.] + min_sizes
max_sizes = [min_dim * 20 / 100.] + max_sizes

print min_sizes, max_sizes
