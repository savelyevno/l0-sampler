__author__ = 'nikita'
from sparse_recovery.OneSparseRecoverer import OneSparseRecoverer
from tools.hash_function import pick_k_ind_hash_function
from numpy import log


class SparseRecoverer:
    """


    """
    def __init__(self, n, s, k, delta):
        """
        
        :param n:
        :type n:
        :param s:
        :type s:
        :param k:
        :type k:
        :param delta:
        :type delta:
        """

        self.n = n

        self.r = int(log(s / delta))

        self.g = [pick_k_ind_hash_function(n, 2*s, 2) for i in range(0, self.r)]
