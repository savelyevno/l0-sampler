__author__ = 'nikita'
from L0Sampler import L0Sampler
from numpy import random
from tools.Timer import Timer
from pympler import asizeof

random.seed(1)

n = int(1e7)
k = int(1e3)
amplitude = 10

timer = Timer()
timer.start()

l0_sampler = L0Sampler(n)
print(asizeof.asizeof(l0_sampler) >> 13, l0_sampler.get_info())

print('build time:', timer.stop())


update_indexes = [i for i in range(k)]
update_values = [random.randint(1, 10*amplitude) for i in range(k)]
random.shuffle(update_indexes)

timer.start()

for i in range(k):
    pos = update_indexes[i]
    value = update_values[i]

    l0_sampler.update(pos, value)
    l0_sampler.get_sample()


print(timer.stop())
