__author__ = 'nikita'
from tools.primality_test import get_next_prime
from numpy.random import randint


class OneSparseRecoverer:
    """
        Exact 1-sparse recovery data structure.

        This structure contains sketched info about some int vector a
        of length n, allowing to make linear updates on it.

        We pick some large prime p and random z in Z_p,
        then maintain several counters
            iota = sum(i*a_i)
            fi = sum(a_i)
            tau = sum(a_i * z_i**i) mod p.
        If tau = fi * z**(iota / fi) returns the only containing a_i,
        otherwise return FAIL.

        If vector contains only a single non-zero element,
        returns its position and value. Otherwise with probability
        greater then 1 - n/p returns that a is > 1-sparse.

        References:
            Cormode, Graham, and Donatella Firmani. "On unifying the space of l0-sampling algorithms."
            Proceedings of the Meeting on Algorithm Engineering & Experiments.
            Society for Industrial and Applied Mathematics, 2013.
            https://pdfs.semanticscholar.org/b0f3/336c82b8a9d9a70d7cf187eea3f6dbfd1cdf.pdf
    """
    def __init__(self, n):
        """

        :param n:   Size of array.
        :type n:    int
        """

        self.p = get_next_prime(n*n)

        self.z = randint(1, self.p)

        self.iota = 0
        self.fi = 0
        self.tau = 0

    def update(self, i, Delta):
        """

        :param i:       Index of an update.
        :type i:        int
        :param Delta:   Value of an update.
        :type Delta:    int
        :return:        Some a_i or None.
        :rtype:         (int, int) or None
        """

        self.iota += (i + 1)*Delta
        self.fi += Delta
        self.tau += Delta * pow(self.z, i + 1, self.p)

    def get(self):
        """

        :return:
        :rtype:
        """

        if self.iota % self.fi == 0 and\
           self.tau == (self.fi * pow(self.z, self.iota/self.fi, self.p)) % self.p:
            return self.iota / self.fi - 1, self.fi
        else:
            return None