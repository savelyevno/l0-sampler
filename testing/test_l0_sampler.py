from L0Sampler import L0Sampler
import random
from tools.Timer import Timer
from pympler import asizeof


def test1():
    n = int(1e10)
    k = int(1e3)
    amplitude = 10

    timer = Timer()
    timer.start()

    l0_sampler = L0Sampler(n)
    print('size of l0-sampler:', asizeof.asizeof(l0_sampler) >> 13, 'KB')
    print(l0_sampler.get_info())

    print('build time:', timer.stop())
    timer.start()

    for i in range(k):
        pos = random.randint(0, n)
        value = random.randint(-amplitude, 10*amplitude)

        l0_sampler.update(pos, value)

    print('update time:', timer.stop())


def test2():
    a = L0Sampler(2, 0)
    b = L0Sampler(2, 0)

    b.update(0, 10)

    a.add(b)

    print(a.get_sample())


test2()
