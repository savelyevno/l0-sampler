__author__ = 'nikita'


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
        raise TypeError('Variable must be ', str(types), ', found', type(var))
