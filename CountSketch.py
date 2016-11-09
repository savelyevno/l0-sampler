__author__ = 'nikita'
from tools.hash_function import pick_k_ind_hash_function
from tools.validation import *
from numpy import log
import numpy as np


class CountSketch:
    """
        Count-Sketch data structure.
        Approximates vector x of given length n, where
        elements of x are linearly updated over time.

        If x is an exact vector, and x' is its approximation
        then the following holds
        Pr[|x_i - x'_i| >= eps * l_2_norm(x)] <= delta.

        Total space required is O(log(1/delta) / eps).


        Notes:
            Maintains table C with d rows and w columns where
            each row corresponds to unique pair of hash functions
                h:{0, .., n - 1}-->{0, .., w - 1},
                g:{0, .., n - 1}-->{-1, 1}
            and each column is one of the values that index i
            can be hashed to by function h.

            Each C[l][j] contains an estimate of some x_i times g_l(i).

            Recovering x_i consists of acquiring in every row element
            that corresponds to x_i and taking median of d resulting
            values times g_l(i).

        References
            https://courses.engr.illinois.edu/cs598csc/fa2014/Lectures/lecture_6.pdf

    """
    def __init__(self, eps, delta, n):
        """

        :param eps:
        :type eps:      float
        :param delta:
        :type delta:    float
        :param n:
        :type n:        int

        """

        check_type(eps, float)
        check_type(delta, float)
        check_type(n, int)

        self.n = n

        self.d = int(log(1 / delta)) + 1
        self.w = int(3 / eps**2) + 1

        self.h = np.array([pick_k_ind_hash_function(n, self.w, 2)] * self.d)
        self.g = np.array([pick_k_ind_hash_function(n, 2, 2)] * self.d)

        self.C = np.zeros((self.d, self.w), dtype=float)

    def modified_g(self, l, i):
        """

        :param l:   Row number.
        :type l:    int
        :param i:   Index to hash.
        :type i:    int
        :return:    -1 or 1
        :rtype:     int
        """

        check_type(l, int)
        check_type(i, int)
        check_in_range(0, self.d - 1, l)
        check_in_range(0, self.n - 1, i)

        res = self.g[l](i)
        if res == 0:
            res -= 1
        return res

    def update(self, i, Delta):
        """

        :param i:
        :type i:        int
        :param Delta:
        :type Delta:    float or int
        :return:
        :rtype:

        Time Complexity O(d)
        """

        check_type(i, int)
        check_type(Delta, float, int)
        check_in_range(0, self.n - 1, i)

        for l in range(0, self.d):
            self.C[l][self.h[l](i)] += self.modified_g(l, i) * Delta

    def recover_by_index(self, i):
        """

        :param i:   Index of recovering element.
        :type i:    int
        :return:    Approximated value of x_i.
        :rtype:     float

        Time Complexity O(d * log(d))
        """

        check_type(i, int)
        check_in_range(0, self.n - 1, i)

        estimates = np.sort(np.array([self.modified_g(l, i) * self.C[l][self.h[l](i)] for l in range(0, self.d)]))

        return np.median(estimates)

    def recover(self):
        """

        :return: x
        :rtype: np.array

        Time Complexity O(n * d * log(d))
        """

        x = np.array([self.recover_by_index(i) for i in range(0, self.n)])

        return x
