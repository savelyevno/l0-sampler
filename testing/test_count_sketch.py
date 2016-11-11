__author__ = 'nikita'
from count_sketch.CountSketch import CountSketch
from tools.Timer import Timer
import numpy as np


n = int(1e5)
number_of_tests = int(1e0)
length_of_stream = int(1e3)
amplitude = 10


eps = 1e-2
delta = 1/n

error_count = 0

print_sketch_info = True

timer = Timer()
timer.start()

for t in range(number_of_tests):

    x = np.zeros(n)

    sketch = CountSketch(eps, delta, n, False)
    if print_sketch_info:
        print_sketch_info = False
        print('sketch info: d =', sketch.d, 'w =', sketch.w)

    for i in range(0, length_of_stream):
        index = np.random.randint(0, n)
        value = np.random.randint(-amplitude, 10*amplitude + 1)

        x[index] += value
        sketch.update(index, value)

    # x_ = sketch.recover()
    l2_norm = np.linalg.norm(x)
    for i in range(0, n):
        if abs(x[i] - sketch.recover_by_index(i)) > eps*l2_norm:
            error_count += 1

print('expected error probability:', delta)
print('actual error frequency:', error_count / (number_of_tests*n))
print('running time:', timer.stop())