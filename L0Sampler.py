__author__ = 'nikita'
from sparse_recovery.SparseRecoverer import SparseRecoverer
from tools.hash_function import pick_k_ind_hash_function
from numpy import log, log2, ceil
from tools.validation import check_in_range, check_type


class L0Sampler:
    """
        l0-sampler data structure.

        Contains sketched info about integer-valued vector a
        of length n allowing to make linear updates to a and
        get l0-samples from it.

        For info on how this works see References.

        Space Complexity
            O(s * log(s / delta) * log(n)**2), where eps and delta such that
            l0-sampler success probability is at least 1 - delta, output
            distribution is in range
            [(1 - eps)/l0_norm(a) - delta, (1 + eps)/l0_norm(a) + delta].

            We take s = O(log(1 / eps) + log(1 / delta)),
                eps = delta = poly(1/n).
            Thus space complexity is O(log(n)**4).

    References:
            Cormode, Graham, and Donatella Firmani. "On unifying the space of l0-sampling algorithms."
            Proceedings of the Meeting on Algorithm Engineering & Experiments.
            Society for Industrial and Applied Mathematics, 2013.
            https://pdfs.semanticscholar.org/b0f3/336c82b8a9d9a70d7cf187eea3f6dbfd1cdf.pdf
    """
    def __init__(self, n):
        """

        :param n:   Length of vector a.
        :type n:    int
        """

        self.n = n
        self.rows = int(ceil(log2(n)))

        self.eps = 1/n
        self.delta = 1/n

        self.sparse_degree = int(ceil(log(1 / self.eps) + log(1 / self.delta)))
        self.k = int(self.sparse_degree / 2)

        self.hash_function = pick_k_ind_hash_function(n, n ** 3, self.k)

        self.recoverers = [SparseRecoverer(n, self.sparse_degree, self.delta) for i in range(self.rows)]
        
    def update(self, i, Delta):
        """
            Update of type a_i += Delta.

            Time Complexity
                O(log(n)**5)

        :param i:       Index of update.
        :type i:        int
        :param Delta:   Value of update.
        :type Delta:    int
        :return: 
        :rtype:         None
        """

        check_type(i, int)
        check_type(Delta, int)
        check_in_range(0, self.n - 1, i)

        for l in range(self.rows):
            if pow(self.n, 3) << l >= self.hash_function(l):
                self.recoverers[l].update(i, Delta)

    def get_sample(self):
        """
            Get l0-sample.

            Time Complexity
                O(log(n)**6)

        :return:    Return tuple (i, a_i) or FAIL.
        :rtype:     None or (int, int)
        """

        for l in range(self.rows):
            recover_result = self.recoverers[l].recover()

            if isinstance(recover_result, dict):
                arg_min = 0
                res_min = pow(self.n, 3)

                for key in recover_result:
                    hash_value = self.hash_function(key)

                    if hash_value < res_min:
                        res_min = hash_value
                        arg_min = key

                return arg_min, recover_result[arg_min]

        return None
