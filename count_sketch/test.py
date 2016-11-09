__author__ = 'nikita'
from count_sketch.CountSketch import CountSketch
import numpy as np


n = int(1e8)
number_of_tests = int(1e0)
length_of_stream = int(1e8)
amplitude = 10


eps = 1e-2
delta = 1/n

error_count = 0

print_sketch_info = True

for t in range(0, number_of_tests):

    x = np.array([0]*n)

    sketch = CountSketch(eps, delta, n)
    if print_sketch_info:
        print_sketch_info = False
        print('sketch info: d =', sketch.d, 'w =', sketch.w)

    for i in range(0, length_of_stream):
        index = np.random.randint(0, n)
        value = np.random.randint(-amplitude, 10*amplitude + 1)

        x[index] += value
        sketch.update(index, value)

    x_ = sketch.recover()
    l2_norm = np.linalg.norm(x)
    for i in range(0, n):
        if abs(x[i] - x_[i]) > eps*l2_norm:
            error_count += 1

print('expected error probability:', delta)
print('actual error frequency:', error_count / (number_of_tests*n))
