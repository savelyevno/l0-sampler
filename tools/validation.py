def check_type(var, *types):
    """
        Raises exception if var is not an instance of
        any of types provided in *types.

    :param var: Variable to check.
    :type var:
    :param types:  Types to check.
    :type types:
    :return:
    :rtype:
    """

    success = False
    for tp in types:
        if isinstance(var, tp):
            success = True
            break

    if not success:
        raise TypeError('Variable must be ', types, ', found', type(var))


def check_in_range(a, b, i):
    """
        Checks whether i is in [a, b].

    :param a:   Left boundary.
    :type a:    int
    :param b:   Right boundary.
    :type b:    int
    :param i:   Value to check.
    :type i:    int
    :return:
    :rtype:

    """

    if not (a <= i <= b):
        raise ValueError('element', i, 'is not in range (', a, ',', b, ')')
