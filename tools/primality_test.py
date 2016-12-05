from numpy.random import randint


class PrimeGetter:
    def __init__(self):
        self.cache = {}

    @staticmethod
    def is_prime(n):
        """
            Checks whether n is prime.

        :param n:   Value to check primality.
        :type n:    int
        :return:    False  if n is composite, otherwise
                    True   (probably prime).
        :rtype:     bool

        Notes
            May return True even though n is composite.
            Probability of such event is not greater than 1 / n**2.

        Time complexity
            O(log(n)**3)

        References
            https://en.wikipedia.org/wiki/Miller–Rabin_primality_test

        """

        if n == 2 or n == 3:
            return True
        elif n == 1 or n % 2 == 0:
            return False

        d = n - 1
        r = 0
        while d % 2 == 0:  # n - 1 = d * 2**r, d - odd
            r += 1
            d >>= 1

        for i in range(r):
            a = randint(2, n)

            x = pow(a, d, n)  # x = (a**d) % n

            if x == 1 or x == n - 1:
                continue

            stop = False
            j = 0
            while j < r - 1 and not stop:
                x = pow(x, 2, n)

                if x == 1:
                    return False
                if x == n - 1:
                    stop = True

                j += 1

            if not stop:
                return False

        return True

    def get_next_prime(self, n):
        """
            Finds smallest prime greater than n.

        :param n:   Lower bound of value of prime.
        :type n:    int
        :return:    Prime p >= n.
        :rtype:     int

        Time complexity
            O(log(n)**4)

        """

        if n in self.cache:
            return self.cache[n]

        p = n + 1
        if p % 2 == 0:
            p += 1

        while not self.is_prime(p):
            p += 2

        self.cache[n] = p
        return p


prime_getter = PrimeGetter()
