from sparse_recovery.SparseRecoverer import SparseRecoverer
from tools.hash_function import pick_k_ind_hash_function
import random
from math import log, log2, ceil
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
            https://pdfs.semanticscholar.org/b0f3/336c82b8a9d9a70d7cf187eea3f6dbfd1cdf.pdf
    """
    def __init__(self, n, init_seed=None):
        """

        Time Complexity:
            O(log(n)**8)

        :param n:           Length of vector a.
        :type n:            int
        :param init_seed:   Seed for random generator to initialize data structure, optional

                            This is needed for adding sketches together. We can add sketches S1 and S2
                            only if for each level l of S1 and S2, s-sparse recoverers that are stored there
                            have the same hash functions and 1-sparse recoverers that they consist of are
                            initialized with the same random parameters. This holds if we initialize S1 and S2
                            consequentially setting the seed for PRG.
        :type init_seed:    int
        """

        self.init_seed = init_seed
        if init_seed is not None:
            random.seed(init_seed)

        self.n = n
        self.levels = int(ceil(log2(n)))

        self.eps = 1/n
        self.delta = 1/n

        # for more accurate distribution among levels one may want to increase this value
        self.n_hash_power = 1

        self.sparse_degree = int(ceil(log(1 / self.eps) + log(1 / self.delta)))
        self.k = int(self.sparse_degree / 2)

        self.hash_function = pick_k_ind_hash_function(n, n ** self.n_hash_power, self.k)

        self.recoverers = [SparseRecoverer(n, self.sparse_degree, self.delta) for i in range(self.levels)]
        
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

        for l in range(self.levels):
            if (self.n**self.n_hash_power) >> l > self.hash_function(i):
                self.recoverers[l].update(i, Delta)

    def _get_sample_with_min_hash(self):
        """
            Get l0-sample.
            (This one is described in the paper.)

            Time Complexity
                O(log(n)**6)

        :return:    Return tuple (i, a_i) or FAIL.
        :rtype:     None or (int, int)
        """

        for l in range(self.levels):
            recover_result = self.recoverers[l].recover()

            if isinstance(recover_result, dict):
                arg_min = -1
                res_min = pow(self.n, self.n_hash_power)

                for key in recover_result:
                    hash_value = self.hash_function(key)

                    if hash_value < res_min:
                        res_min = hash_value
                        arg_min = key

                return arg_min, recover_result[arg_min]

        return None

    def get_sample(self):
        """
            Get l0-sample.

            Time Complexity
                O(log(n)**6)

        :return:    Return tuple (i, a_i) or FAIL.
        :rtype:     None or (int, int)
        """

        result = {}

        for l in range(self.levels):
            recover_result = self.recoverers[l].recover()

            if isinstance(recover_result, dict):
                key = random.choice(list(recover_result.keys()))
                result[key] = recover_result[key]

        if len(result) > 0:
            key = random.choice(list(result.keys()))
            return key, result[key]
        return None

    def _get_info(self):
        """

        :return:    Object info
        :rtype:     dict
        """
        result = {
            'l0-sampler levels': self.levels,
            'l0-sampler s': self.sparse_degree,
            'l0-sampler k': self.k
        }

        result = {**result, **self.recoverers[0]._get_info()}

        return result

    def add(self, another_l0_sampler):
        """
            Combines two l0-samplers by adding them.

            !Assuming they have the same hash functions. (This should hold
            if they were initialized with the same random bits)

        Time Complexity
            O(log(n)**4)

        :param another_l0_sampler:  l0-sampler to add.
        :type another_l0_sampler:   L0Sampler
        :return:
        :rtype:     None
        """

        if self.n != another_l0_sampler.n:
            raise ValueError('l0-samplers are not compatible')
        if self.init_seed is None or another_l0_sampler.init_seed is None or\
           self.init_seed != another_l0_sampler.init_seed:
            raise ValueError('samplers are not initialized from the same random bits')

        for l in range(self.levels):
            self.recoverers[l].add(another_l0_sampler.recoverers[l])
