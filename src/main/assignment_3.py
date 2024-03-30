import numpy as np

def euler_method(f, t0, y0, t_end, n):
    h = (t_end - t0) / n
    t = t0
    y = y0

    for _ in range(n):
        y += h * f(t, y)
        t += h

    return y

def runge_kutta(f, t0, y0, t_end, n):
    h = (t_end - t0) / n 
    t = t0
    y = y0

    for _ in range(n):
        k1 = f(t, y)
        k2 = f(t + 0.5 * h, y + 0.5 * h * k1)
        k3 = f(t + 0.5 * h, y + 0.5 * h * k2)
        k4 = f(t + h, y + h * k3)
        
        y += (h / 6) * (k1 + 2*k2 + 2*k3 + k4)
        t += h 

    return y


def backward_substitution(U, b):
    n = len(b)
    x = np.zeros_like(b)

    for i in range(n-1, -1, -1):
        if U[i, i] == 0:
            raise ValueError("Matrix is singular.")
        x[i] = (b[i] - np.dot(U[i, i+1:], x[i+1:])) / U[i, i]
    
    return x

def gaussian_elimination(A, b):
    n = len(b)
    Ab = np.hstack([A, b.reshape(-1, 1)])
    
    for i in range(n):
        for j in range(i+1, n):
            if Ab[i][i] == 0:
                Ab[[i, j]] = Ab[[j, i]]
            ratio = Ab[j][i] / Ab[i][i]
            Ab[j] = Ab[j] - ratio * Ab[i]
    
    return backward_substitution(Ab[:, :-1], Ab[:, -1])


def lu_factorization(A):
    n = A.shape[0]
    L = np.zeros_like(A, dtype=np.double)
    U = np.zeros_like(A, dtype=np.double)
    
    for i in range(n):
        # Upper Triangular
        for k in range(i, n):
            sum_ = sum(L[i][j] * U[j][k] for j in range(i))
            U[i][k] = A[i][k] - sum_
        
        # Lower Triangular
        for k in range(i, n):
            if i == k:
                L[i][i] = 1
            else:
                sum_ = sum(L[k][j] * U[j][i] for j in range(i))
                L[k][i] = (A[k][i] - sum_) / U[i][i]
    
    return L, U

def is_diagonally_dominant(A):
    n = A.shape[0]
    for i in range(n):
        sum_of_row = np.sum(np.abs(A[i])) - np.abs(A[i,i])
        if np.abs(A[i,i]) <= sum_of_row:
            return False
    return True

def is_positive_definite(matrix):
    if not np.allclose(matrix, matrix.T):
        return False
    try:
        eigenvalues = np.linalg.eigvals(matrix)
        return np.all(eigenvalues > 0)
    except np.linalg.LinAlgError:
        return False