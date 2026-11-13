
import sympy as sy

from errorprop.core import error_propagation


def main() -> None:
    # init sympy pretty printing
    sy.init_printing(pretty_print=True, use_latex="mathjax")
    
    # Define the variables
    A_1, A_2, delta_1, delta_2 = sy.symbols("A_1 A_2 delta_1 delta_2")
    omega_1, omega_2 = sy.symbols("omega_1 omega_2")
    t = sy.symbols("t")
    ddelta = sy.symbols("Delta_delta")
    domega = sy.symbols("Delta_omega")

    # Define the function
    f = A_1**2 + A_2**2 + 2 * A_1 * A_2 * sy.cos(delta_1 - delta_2 + (omega_1 - omega_2) * t)
    f = f.subs({delta_1: ddelta + delta_2, omega_1: domega + omega_2}).simplify()

    # Call the error propagation function
    propagated_error, jacobian, input_covariance_matrix_symbols = error_propagation(f, [A_1,A_2, ddelta], uncorrelated=True)

    print(f"Function: {f}")


    # Print the results
    print("Propagated Error:")
    print(propagated_error)
    print("\nJacobian:")
    print(jacobian)
    print("\nInput Covariance Matrix Symbols:")
    print(input_covariance_matrix_symbols)

if __name__ == "__main__":
    main()