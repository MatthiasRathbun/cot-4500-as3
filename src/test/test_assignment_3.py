import sys
from pathlib import Path
import numpy as np

parent_dir = str(Path(__file__).parent.parent)
sys.path.append(parent_dir)
from main.assignment_3 import euler_method, runge_kutta, gaussian_elimination, lu_factorization, is_diagonally_dominant, is_positive_definite

def test_euler_method():
    def diff_eq(t, y):
        return t - y**2

    t0 = 0
    y0 = 1
    t_end = 2
    n = 10

    return euler_method(diff_eq, t0, y0, t_end, n)

def test_runge_kutta():
    def diff_eq(t, y):
        return t - y**2

    t0 = 0
    y0 = 1
    t_end = 2
    n = 10

    return runge_kutta(diff_eq, t0, y0, t_end, n)

def test_gaussian_elimination():
    A = np.array([[2, -1, 1],
                [1, 3, 1],
                [-1, 5, 4]])

    b = np.array([6, 0, -3])

    return gaussian_elimination(A, b)

def test_lu_factorization():
    A = np.array([[1, 1, 0, 3],
              [2, 1, -1, 1],
              [3, -1, -1, 2],
              [-1, 2, 3, -1]], dtype=np.float64)
    return lu_factorization(A)

def test_diagonally_dominant():
    A = np.array([[9, 0, 5, 2, 1],
              [3, 9, 1, 2, 1],
              [0, 1, 7, 2, 3],
              [4, 2, 3, 12, 2],
              [3, 2, 4, 0, 8]])
    return is_diagonally_dominant(A)

def test_positive_definite():
    A = np.array([[2, 2, 1],
              [2, 3, 0],
              [1, 0, 2]])
    return is_positive_definite(A)

def test_functions():
    print("Testing Euler method:")
    print(test_euler_method())
    print("Testing Runge-Kutta method:")
    print(test_runge_kutta())
    print("Testing Gaussian elimination:")
    print(test_gaussian_elimination())
    print("Testing LU factorization:")
    L, U = test_lu_factorization()
    determinant = np.prod(np.diag(U))
    print(L)
    print(U)
    print(determinant)
    print("Testing diagonally dominant:")
    print(test_diagonally_dominant())
    print("Testing positive definite:")
    print(test_positive_definite())

if __name__ == "__main__":
    test_functions()