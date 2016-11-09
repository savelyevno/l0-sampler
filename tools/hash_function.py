__author__ = 'nikita'
from tools.primality_test import get_next_prime
from numpy.random import randint
from tools.validation import check_type
import numpy as np


def pick_k_ind_hash_function(n, w, k):
    """
        Picks random hash function h:{0, ..., n - 1}-->{0, ..., w - 1}, w <= n
        from family of k-independent hash functions.

    :param n:   Defines domain of resulting hash function.
    :type n:    int
    :param w:   Defines image of resulting hash function.
    :type w:    int
    :param k:   Degree of independence of hash function.
    :type k:    int
    :return:    h
    :rtype:     function

    Notes
        First, finds smallest prime p >= n. Then constructs array of
        coefficients a from Z_p, where a[0] != 0. Let
        h(x) = a[0] * x**(k - 1) + ... + a[k - 2] * x + a[k - 1] mod p mod w.

    Complexity
        Constructed hash function's complexity is O(k * log(x)**3).

    References
        https://en.wikipedia.org/wiki/K-independent_hashing

    """

    check_type(n, int)
    check_type(w, int)
    check_type(k, int)

    p = get_next_prime(max(n, w))

    a = np.array([randint(0, p) for i in range(0, k)])
    a[0] = randint(1, p)

    def h(x):
        return (sum((a[i] * pow(x, k - i - 1, p)) % p for i in range(0, k)) % p) % w

    return h
