from tools.primality_test import get_next_prime
from random import randint
from tools.validation import check_type


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
        h(x) = a[0] * x**(k - 1) + ... + a[k - 2] * x + a[k - 1] mod p mod w.

    Time complexity
        Constructed hash function's complexity is O(k * log(x)**2).

    References
        https://en.wikipedia.org/wiki/K-independent_hashing

    """

    check_type(n, int)
    check_type(w, int)
    check_type(k, int)

    p = get_next_prime(max(n, w))

    a = [randint(0, p) for i in range(k)]
    a[0] = randint(1, p)

    def h(x):
        res = 0
        pow_x = 1
        for i in range(k):
            res = (res + pow_x * a[k - i - 1]) % p
            pow_x = (pow_x * x) % p
        return res % w

    return h
