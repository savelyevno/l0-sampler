from tools.primality_test import prime_getter
import random


def pick_k_ind_hash_function(n, w, k):
    """
        Picks random hash function h:{0, ..., n - 1}-->{0, ..., w - 1}
        from a family of k-independent hash functions.

    :param n:   Defines domain of resulting hash function.
    :type n:    int
    :param w:   Defines image of resulting hash function.
    :type w:    int
    :param k:   Degree of independence of hash function.
    :type k:    int
    :return:    Generated hash function.
    :rtype:     function

    Notes
        Firstly, finds smallest prime p >= n. Then constructs array of
        coefficients a from Z_p, where a[0] != 0. Let
        h(x) = a[0] + ... + a[k - 2] * x^(k - 2) + a[k - 1] * x^(k - 1) mod p mod w.

    Time complexity
        Constructed hash function's complexity is O(k * log(x)).

    References
        https://en.wikipedia.org/wiki/K-independent_hashing

    """

    p = prime_getter.get_next_prime(max(n, w))

    a = tuple(random.randint(i == k - 1, p - 1) for i in range(k))

    def h(x):
        res = 0
        pow_x = 1
        for i in range(k):
            res = (res + pow_x * a[i]) % p
            pow_x = (pow_x * x) % p
        return res % w

    return h
