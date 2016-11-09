__author__ = 'nikita'
from random import randint
from numpy import log2
from tools.validation import check_type


def is_prime(n):
    """
        Checks whether n is prime.

    :param n:   Value to check primality.
    :type n:    int
    :return:    False  if n is composite, otherwise
                True   (probably prime).
    :rtype:     bool

    Notes
        May return True even though n is composite.
        Probability of such event is not greater than 1 / n**2.

    Time complexity
        O(log(n)**4)

    References
        https://en.wikipedia.org/wiki/Miller–Rabin_primality_test

    """

    check_type(n, int)

    if n == 2 or n == 3:
        return True
    elif n == 1 or n % 2 == 0:
        return False

    d = n - 1
    r = 0
    while d % 2 == 0:   # n - 1 = d * 2**r, d - odd
        r += 1
        d >>= 1

    k = int(log2(n)) + 1
    for i in range(0, k):
        a = randint(2, n - 2)

        x = pow(a, d, n)    # x = (a**d) % n

        if x == 1 or x == n - 1:
            continue

        stop = False
        j = 0
        while j < r - 1 and not stop:
            x = pow(x, 2, n)

            if x == 1:
                return False
            if x == n - 1:
                stop = True

            j += 1

        if not stop:
            return False

    return True


def get_next_prime(n):
    """
        Finds smallest prime greater than n.

    :param n:
    :type n:    int or float
    :return:    p
    :rtype:     int

    Time complexity
        O(log(n)**5)

    """

    check_type(n, int, float)

    p = int(n) + 1
    if p % 2 == 0:
        p += 1

    while not is_prime(p):
        p += 2

    return p
