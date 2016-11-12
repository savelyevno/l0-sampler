from sparse_recovery.OneSparseRecoverer import OneSparseRecoverer
from tools.hash_function import pick_k_ind_hash_function
from math import log
from tools.validation import check_in_range, check_type


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
    def __init__(self, n, s, delta):
        """
            Initializes log(s/delta) rows and 2*s columns table and
            2-independent hash function for every row.

        Time Complexity
            O(s * log(s / delta) * log(n)**5)

        :param n:       Length of estimating vector.
        :type n:        int
        :param s:       Degree of required sparsity.
        :type s:        int
        :param delta:   Error term parameter.
        :type delta:    float
        """

        check_type(n, int)
        check_type(s, int)
        check_type(delta, float)
        check_in_range(1, n, s)

        self.n = n
        self.delta = delta
        self.sparse_degree = s

        self.columns = 2*s
        self.rows = int(log(s / delta))

        self.hash_function = [pick_k_ind_hash_function(n, self.columns, 2) for i in range(self.rows)]

        self.R = [[OneSparseRecoverer(self.n) for j in range(self.columns)] for i in range(self.rows)]

    def update(self, i, Delta):
        """
            Iterates through rows and updates corresponding cells.

        Time Complexity
            O(log(s / delta) * log(n)**3)

        :param i:       Index of update.
        :type i:        int
        :param Delta:   Value of update.
        :type Delta:    int
        :return:
        :rtype:         None
        """

        for l in range(self.rows):
            self.R[l][self.hash_function[l](i)].update(i, Delta)

    def recover(self):
        """
            Iterates through every cell and tries to recover element from it.
            Recovered elements are merged into a dictionary to avoid duplicates.

        Time Complexity
            O(s * log(s / delta) * log(n)**3)

        :return:    If resulting dictionary is not empty returns it, otherwise
                    returns None.
        :rtype:     dict or None
        """

        result = {}
        for i in range(self.rows):
            for j in range(self.columns):
                one_sparse_recovery_result = self.R[i][j].recover()

                if one_sparse_recovery_result is not None:
                    result[one_sparse_recovery_result[0]] = one_sparse_recovery_result[1]

        if len(result) == 0:
            return None
        else:
            return result

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
            O(s*log(s / delta)*log(n)**2)

        :param another_s_sparse_recoverer:  s-sparse recoverer to add.
        :type another_s_sparse_recoverer:   SparseRecoverer
        :return:
        :rtype:
        """

        if self.n != another_s_sparse_recoverer.n or\
           self.sparse_degree != another_s_sparse_recoverer.sparse_degree or\
           self.delta != another_s_sparse_recoverer.delta:
            raise ValueError('s-sparse recoverers are not compatible')
        else:
            self.sum_of_vector += another_s_sparse_recoverer.sum_of_vector
            for i in range(self.rows):
                for j in range(self.columns):
                    self.R[i][j].add(another_s_sparse_recoverer.R[i][j])
