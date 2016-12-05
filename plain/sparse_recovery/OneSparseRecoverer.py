from tools.primality_test import prime_getter
from random import randint


class OneSparseRecoverer:
    """
        Exact 1-sparse recovery data structure.

        This structure contains sketched info about some int vector a
        of length n, allowing to make linear updates on it. If a contains
        only single non-zero element, this data structure allows to
        recover it exactly with high probability.

        We pick some large prime p and random z in Z_p,
        then maintain several counters
            iota = sum(i*a_i)
            fi = sum(a_i)
            tau = sum(a_i * z_i**i) mod p.
        If tau = fi * z**(iota / fi) returns the only containing a_i.

        If vector contains only a single non-zero element,
        returns its position and value. Otherwise with probability
        greater then 1 - n/p returns that a is > 1-sparse.

        Space Complexity
            O(log(n))

        References:
            Cormode, Graham, and Donatella Firmani. "On unifying the space of l0-sampling algorithms."
            Proceedings of the Meeting on Algorithm Engineering & Experiments.
            Society for Industrial and Applied Mathematics, 2013.
            https://pdfs.semanticscholar.org/b0f3/336c82b8a9d9a70d7cf187eea3f6dbfd1cdf.pdf
    """

    __slots__ = 'n', 'p', 'z', 'iota', 'fi', 'tau'

    def __init__(self, n):
        """

        Time Complexity
            O(log(n)**4)

        :param n:   Size of array.
        :type n:    int
        """

        self.n = n

        self.p = prime_getter.get_next_prime(n*100)

        self.z = randint(1, self.p - 1)

        self.iota = 0
        self.fi = 0
        self.tau = 0

    def update(self, i, Delta):
        """
            Update of type a_i += Delta.

        Time Complexity
            O(log(n))

        :param i:       Index of an update.
        :type i:        int
        :param Delta:   Value of an update.
        :type Delta:    int
        :return:
        :rtype:         None
        """

        self.iota += (i + 1)*Delta
        self.fi += Delta
        self.tau = (self.tau + Delta * pow(self.z, i + 1, self.p)) % self.p

    def recover(self):
        """
            Attempt to recover an element.

        Time Complexity
            O(log(n))

        :return:    On success return a_i, otherwise FAIL.
        :rtype:     (int, int) or None
        """

        if self.fi != 0 and self.iota % self.fi == 0 and self.iota // self.fi > 0 and \
                self.tau == self.fi * pow(self.z, self.iota // self.fi, self.p) % self.p:
            return self.iota // self.fi - 1, self.fi
        else:
            return None

    def add(self, another_one_sparse_recoverer):
        """
            Combines two 1-sparse recoverers by adding them.

        Time Complexity
            O(1)

        :param another_one_sparse_recoverer:
        :type another_one_sparse_recoverer:     OneSparseRecoverer
        :return:
        :rtype:     None
        """

        if self.z != another_one_sparse_recoverer.z or self.p != another_one_sparse_recoverer.p or\
           self.n != another_one_sparse_recoverer.n:
            print(self)
            print(another_one_sparse_recoverer)
            raise ValueError('1-sparse recoverers are not compatible')

        self.iota += another_one_sparse_recoverer.iota
        self.fi += another_one_sparse_recoverer.fi
        self.tau = (self.tau + another_one_sparse_recoverer.tau) % self.p

    def subtract(self, another_one_sparse_recoverer):
        """
            Combines two 1-sparse recoverers by subtracting them.

        Time Complexity
            O(1)

        :param another_one_sparse_recoverer:
        :type another_one_sparse_recoverer:     OneSparseRecoverer
        :return:
        :rtype:     None
        """

        if self.z != another_one_sparse_recoverer.z or self.p != another_one_sparse_recoverer.p or\
           self.n != another_one_sparse_recoverer.n:
            raise ValueError('1-sparse recoverers are not compatible')

        self.iota -= another_one_sparse_recoverer.iota
        self.fi -= another_one_sparse_recoverer.fi
        self.tau = (self.tau - another_one_sparse_recoverer.tau) % self.p

    def __str__(self):
        return ", ".join(map(str, [self.n, self.p, self.z, self.iota, self.fi, self.tau]))
