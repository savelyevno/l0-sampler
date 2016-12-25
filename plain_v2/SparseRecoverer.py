from math import log
from random import randint

from tools.hash_function import pick_k_ind_hash_function
from tools.primality_test import prime_getter


class SparseRecoverer:
    """
        s-sparse recovery data structure.
        Contains table of 1-sparse recovery data structures.

        Similar to 1-sparse recovery data structure, s-parse recovery
        contains sketched info about some int vector a
        of length n, allowing to make linear updates on it. If a contains
        no more than s non-zero elements, this data structure allows to
        recover them exactly with high probability.

        On update (i, Delta) for every row l selects corresponding
        to i column using hash function
            h_l:{0, .., n - 1}-->{0, .., 2*s - 1}.
        After that updates selected cell.

        Recovering consists of trying to recover values from
        every 1-sparse recoverers and merging them into a single array.
        Returns obtained vector if it is s-sparse. Returns FAIL otherwise.

        Space Complexity
            O(s * log(s / delta) * log(n))

        References:
            Cormode, Graham, and Donatella Firmani. "On unifying the space of l0-sampling algorithms."
            Proceedings of the Meeting on Algorithm Engineering & Experiments.
            Society for Industrial and Applied Mathematics, 2013.
            https://pdfs.semanticscholar.org/b0f3/336c82b8a9d9a70d7cf187eea3f6dbfd1cdf.pdf
    """

    __slots__ = 'n', 'delta', 'sparse_degree', 'columns', 'rows', 'hash_function', 'R', 'p'

    def __init__(self, n, s, delta):
        """
            Initializes log(s/delta) rows and 2*s columns table and
            2-independent hash function for every row.

        Time Complexity
            O(s * log(s / delta) * log(n)**4)

        :param n:       Length of estimating vector.
        :type n:        int
        :param s:       Degree of required sparsity.
        :type s:        int
        :param delta:   Error term parameter.
        :type delta:    float
        """

        self.n = n
        self.delta = delta
        self.sparse_degree = s

        self.columns = 2*s
        self.rows = int(log(self.sparse_degree / self.delta))

        self.hash_function = tuple(pick_k_ind_hash_function(n, self.columns, 2) for i in range(self.rows))

        self.p = prime_getter.get_next_prime(n*100)

        # R[i][j] = [z, iota, fi, tau]
        self.R = tuple(tuple([randint(1, self.p - 1), 0, 0, 0] for j in range(self.columns)) for i in range(self.rows))

    def update(self, i, Delta):
        """
            Iterates through rows and updates corresponding cells.

        Time Complexity
            O(log(s / delta) * log(n))

        :param i:       Index of update.
        :type i:        int
        :param Delta:   Value of update.
        :type Delta:    int
        :return:
        :rtype:         None
        """

        for l in range(self.rows):
            recoverer = self.R[l][self.hash_function[l](i)]
            recoverer[1] += (i + 1)*Delta
            recoverer[2] += Delta
            recoverer[3] = (recoverer[3] + Delta * pow(recoverer[0], i + 1, self.p)) % self.p

    def recover(self):
        """
            Iterates through every cell and tries to recover element from it.
            Recovered elements are merged into a dictionary to avoid duplicates.

        Time Complexity
            O(s * log(s / delta) * log(n))

        :return:    If resulting dictionary is not empty returns it, otherwise
                    returns None.
        :rtype:     dict or None
        """

        result = {}
        for i in range(self.rows):
            for j in range(self.columns):
                recoverer = self.R[i][j]
                z = recoverer[0]
                iota = recoverer[1]
                fi = recoverer[2]
                tau = recoverer[3]

                if fi != 0 and\
                   iota % fi == 0 and iota // fi > 0 and \
                   tau == fi * pow(z, iota // fi, self.p) % self.p:
                    result[iota // fi - 1] = fi

        if result:
            return result
        else:
            return None

    def _get_info(self):
        """

        :return:    Object info.
        :rtype:     dict
        """

        result = {
            's-sparse recoverer rows': self.rows,
            's-sparse recoverer columns': self.columns
        }

        return result

    def add(self, another_s_sparse_recoverer):
        """
            Combines to s-sparse recoverers by adding.

            !Assuming they have the same hash functions.

        Time Complexity
            O(s*log(s / delta))

        :param another_s_sparse_recoverer:  s-sparse recoverer to add.
        :type another_s_sparse_recoverer:   SparseRecoverer
        :return:
        :rtype:
        """

        if self.n != another_s_sparse_recoverer.n or\
           self.sparse_degree != another_s_sparse_recoverer.sparse_degree or\
           self.delta != another_s_sparse_recoverer.delta or\
           self.p != another_s_sparse_recoverer.p:
            raise ValueError('s-sparse recoverers are not compatible')
        else:
            for i in range(self.rows):
                for j in range(self.columns):
                    recoverer = self.R[i][j]
                    another_recoverer = another_s_sparse_recoverer.R[i][j]

                    if recoverer[0] != another_recoverer[0]:
                        raise ValueError('1-sparse recoverers are not compatible')

                    recoverer[1] += another_recoverer[1]
                    recoverer[2] += another_recoverer[2]
                    recoverer[3] = (recoverer[3] + another_recoverer[3]) % self.p

    def subtract(self, another_s_sparse_recoverer):
        """
            Combines to s-sparse recoverers by subtracting.

            !Assuming they have the same hash functions.

        Time Complexity
            O(s*log(s / delta))

        :param another_s_sparse_recoverer:  s-sparse recoverer to add.
        :type another_s_sparse_recoverer:   SparseRecoverer
        :return:
        :rtype:
        """

        if self.n != another_s_sparse_recoverer.n or \
           self.sparse_degree != another_s_sparse_recoverer.sparse_degree or \
           self.delta != another_s_sparse_recoverer.delta or \
           self.p != another_s_sparse_recoverer.p:
            raise ValueError('s-sparse recoverers are not compatible')
        else:
            for i in range(self.rows):
                for j in range(self.columns):
                    recoverer = self.R[i][j]
                    another_recoverer = another_s_sparse_recoverer.R[i][j]

                    if recoverer[0] != another_recoverer[0]:
                        raise ValueError('1-sparse recoverers are not compatible')

                    recoverer[1] -= another_recoverer[1]
                    recoverer[2] -= another_recoverer[2]
                    recoverer[3] = (recoverer[3] - another_recoverer[3]) % self.p
