"""
    Core of the error propagation library
"""
from collections.abc import Iterable

import sympy as sy


def error_propagation(function, input_variables, uncorrelated=False, print_messages=True):
    """
    error_propagation Given a function, and it's variables, computes the propagated error with respect to the input variables

    Args:
        function (sy.expr): The function for which to propagate errors
        input_variables (Iterable[sy.Symbol]): The variables of the function
        uncorrelated (bool): Whether the input variables are uncorrelated or not. If True, the covariance matrix will be diagonal.
    """
    if not isinstance(function, sy.Matrix):
        function = sy.Matrix([function])
    if not isinstance(input_variables, Iterable):
        input_variables = [input_variables]

    assert all(isinstance(var, sy.Symbol) for var in input_variables), "All input variables must be sympy symbols"

    if uncorrelated:
        input_covariance_symbols = [sy.symbols(rf"\sigma_{var}^2") for var in input_variables]
        input_covariance_matrix_symbols = sy.diag(*input_covariance_symbols)
    else:
        input_covariance_symbols = []
        for var_1 in input_variables:
            for var_2 in input_variables:
                input_covariance_symbols.append(sy.symbols(rf"\sigma_{var_1}_{var_2}"))
        input_covariance_matrix_symbols = sy.Matrix(len(input_variables), len(input_variables), input_covariance_symbols)

    jacobian = function.jacobian(input_variables)
    propagated_error = jacobian * input_covariance_matrix_symbols * jacobian.T
    if print_messages:
        print("Error, J, input covariance symbols")
    return propagated_error, jacobian, input_covariance_matrix_symbols


