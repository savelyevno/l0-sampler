import random
from math import log, log2, ceil

from tools.hash_function import pick_k_ind_hash_function
from tools.primality_test import prime_getter


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

    __slots__ = 'init_seed', 'n_hash_power', 'n', 'n_sq', 'levels', 'eps', 'delta', 'sparse_degree', 'k',\
                'hash_function', 'delta_r', 'recoverers_values', 'recoverers_values_tmp', 'recoverers_rows',\
                'recoverers_columns', 'recoverers_column_hash_params', 'prime_after_n', 'one_sp_rec_p'

    def __init__(self, n, delta=-1, init_seed=-1):
        """

        Time Complexity:
            O(log(n)**4)

        :param n:           Length of vector a.
        :type n:            int
        :param delta:       delta parameter of the l0-sampler. -1 ==> delta = 1/log2(n)
        :type delta:        float
        :param init_seed:   l0-samplers that are initialized with the same value of
                            init_seed will have the same random parameters.
        :type init_seed:    int
        """

        if delta == -1:
            delta = 1/log2(n)

        if init_seed == -1:
            init_seed = random.getrandbits(32)
        random.seed(init_seed)
        self.init_seed = init_seed

        # for more accurate distribution among levels one may want to increase this value
        self.n_hash_power = 1

        self.n = n
        self.n_sq = n*n
        self.levels = ceil(log2(self.n))

        self.eps = delta
        self.delta = delta

        self.k = 4
        # self.sparse_degree = ceil(2*(log2(1 / self.delta)))
        self.sparse_degree = 2*self.k

        self.hash_function = pick_k_ind_hash_function(n, n ** self.n_hash_power, self.k)

        # s-sparse recoverers initialization
        self.delta_r = self.delta
        self.recoverers_values = {}
        self.recoverers_values_tmp = {}
        # self.recoverers_rows = int(log2(self.sparse_degree / self.delta_r))
        self.recoverers_rows = 4
        self.recoverers_columns = 2 * self.sparse_degree

        # parameters to hash function that hashes update index into some column
        self.recoverers_column_hash_params = {}

        # prime larger than n
        self.prime_after_n = prime_getter.get_next_prime(n)

        # prime for ring Z_p from which we choose z in 1-sparse recoverers
        self.one_sp_rec_p = prime_getter.get_next_prime(n*100)

    def get_one_sp_rec_z(self, level, row, column):
        """
            Returns z for 1-sparse recoverer.

        :param level:   l0-sampler level.
        :type level:    int
        :param row:     Row number of s-sparse recoverer.
        :type row:      int
        :param column:  Column of s-sparse recoverer.
        :type column:   int
        :return:        Value z that is used to initialize 1-sparse recoverer.
        :rtype:         int
        """

        random.seed((level*self.n_sq + row*self.n + column) ^ self.init_seed)
        return random.randint(1, self.one_sp_rec_p - 1)

    def get_recoverers_column_hash_params(self, level, row):
        """
            Returns parameters of hash function that will hash indexes of update to
            some column.

        :param level:   l0-sampler level.
        :type level:    int
        :param row:     Row number of s-sparse recoverer.
        :type row:      int
        :return:        (a, b). Later they are used for calculating h(i) = (a*i + b) mod p.
        :rtype:         tuple
        """

        random.seed((level*self.n + row) ^ self.init_seed)
        return random.randint(1, self.prime_after_n - 1), random.randint(0, self.prime_after_n - 1)

    def get_column(self, level, row, i):
        """

        :param level:   l0-sampler level.
        :type level:    int
        :param row:     Row number of s-sparse recoverer.
        :type row:      int
        :param i:       Index of update.
        :type i:        int
        :return:        Corresponding column number.
        :rtype:         int
        """

        a = self.recoverers_column_hash_params[(level, row)]
        return ((i * a[0] + a[1]) % self.prime_after_n) % self.recoverers_columns

    def update(self, i, Delta):
        """
            Update of type a_i += Delta.

            Time Complexity
                O(log(n)**3)

        :param i:       Index of update.
        :type i:        int
        :param Delta:   Value of update.
        :type Delta:    int
        :return:
        :rtype:         None
        """

        hash_value = self.hash_function(i)
        n_copy = self.n ** self.n_hash_power - 1
        max_l = 0
        while n_copy >= hash_value and max_l < self.levels:
            max_l += 1
            n_copy >>= 1

        for level in range(max_l):
            for row in range(self.recoverers_rows):
                if (level, row) not in self.recoverers_column_hash_params:
                    self.recoverers_column_hash_params[(level, row)] = self.get_recoverers_column_hash_params(level, row)
                column = self.get_column(level, row, i)

                if (level, row, column) not in self.recoverers_values:
                    self.recoverers_values[(level, row, column)] = [self.get_one_sp_rec_z(level, row, column), 0, 0, 0]

                recoverer = self.recoverers_values[(level, row, column)]
                recoverer[1] += (i + 1) * Delta
                recoverer[2] += Delta
                recoverer[3] = (recoverer[3] + Delta * pow(recoverer[0], i + 1, self.one_sp_rec_p)) % self.one_sp_rec_p

    def get_sample(self):
        """
            Get l0-sample.

            Time Complexity
                O(log(n)**4)

        :return:    Return tuple (i, a_i) or FAIL.
        :rtype:     None or (int, int)
        """

        samples = self.get_samples()
        if len(samples) == 0:
            return None
        else:
            i = random.choice(list(samples.keys()))
            val = samples[i]

            return i, val

        # result = None
        # result_hash = 2*(self.n**self.n_hash_power)
        #
        # for level in range(self.levels):
        #     for row in range(self.recoverers_rows):
        #         for column in range(self.recoverers_columns):
        #             if (level, row, column) not in self.recoverers_values:
        #                 continue
        #
        #             recoverer = self.recoverers_values[(level, row, column)]
        #
        #             z = recoverer[0]
        #             iota = recoverer[1]
        #             fi = recoverer[2]
        #             tau = recoverer[3]
        #
        #             if fi != 0 and iota % fi == 0 and iota // fi > 0 and \
        #                             tau == fi * pow(z, iota // fi, self.one_sp_rec_p) % self.one_sp_rec_p:
        #                 index = iota // fi - 1
        #                 value = fi
        #                 if self.hash_function(index) < result_hash:
        #                     result = index, value
        #
        # return result

    def get_samples(self):
        """
            Get l0-samples.

            Time Complexity
                O(log(n)**4)

        :return:    Return dict of tuples (i, a_i).
        :rtype:     dict
        """

        result = {}

        for level in range(self.levels):
            for row in range(self.recoverers_rows):
                for column in range(self.recoverers_columns):
                    if (level, row, column) not in self.recoverers_values:
                        continue

                    recoverer = self.recoverers_values[(level, row, column)]

                    z = recoverer[0]
                    iota = recoverer[1]
                    fi = recoverer[2]
                    tau = recoverer[3]

                    if fi != 0 and iota % fi == 0 and iota // fi > 0 and \
                            tau == fi * pow(z, iota // fi, self.one_sp_rec_p) % self.one_sp_rec_p:
                        result[iota // fi - 1] = fi

        return result

    def add(self, another_l0_sampler):
        """
            Combines two l0-samplers by adding them.

            !Assuming they have the same hash functions. (This should hold
            if they were initialized with the same random bits)

        Time Complexity
            O(log(n)**3)

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

        for key, another_recoverer in another_l0_sampler.recoverers_values.items():
            level = key[0]
            row = key[1]
            column = key[2]

            if (level, row, column) not in self.recoverers_values:
                self.recoverers_values[(level, row, column)] = [self.get_one_sp_rec_z(level, row, column), 0, 0, 0]
            recoverer = self.recoverers_values[(level, row, column)]

            if recoverer[0] != another_recoverer[0]:
                raise ValueError('1-sparse recoverers are not compatible')
            recoverer[1] += another_recoverer[1]
            recoverer[2] += another_recoverer[2]
            recoverer[3] = (recoverer[3] + another_recoverer[3]) % self.one_sp_rec_p

    def subtract(self, another_l0_sampler):
        """
            Combines two l0-samplers by subtracting them.

            !Assuming they have the same hash functions. (This should hold
            if they were initialized with the same random bits)

        Time Complexity
            O(log(n)**3)

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

        for (key, another_recoverer) in another_l0_sampler.recoverers_values.items():
            level = key[0]
            row = key[1]
            column = key[2]

            if (level, row, column) not in self.recoverers_values:
                self.recoverers_values[(level, row, column)] = [self.get_one_sp_rec_z(level, row, column), 0, 0, 0]
            recoverer = self.recoverers_values[(level, row, column)]

            if recoverer[0] != another_recoverer[0]:
                raise ValueError('1-sparse recoverers are not compatible')
            recoverer[1] -= another_recoverer[1]
            recoverer[2] -= another_recoverer[2]
            recoverer[3] = (recoverer[3] - another_recoverer[3]) % self.one_sp_rec_p
