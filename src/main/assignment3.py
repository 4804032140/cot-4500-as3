import numpy as np
import scipy as sp

np.set_printoptions(precision=5, suppress=True, linewidth=100)

def function(t: float, y: float):
    return t - (y**2)

def eulers(y, low, high, num_of_iterations):
    h = (high - low) / num_of_iterations
    f_i = y + h * function(low, y)
    t = low + h
    for i in range(1, num_of_iterations):
        f_i = f_i + h * function(t, f_i)
        t = t + h
    return f_i

def runge_kutta(w, low, high, num_of_iterations):
    h = (high - low) / num_of_iterations
    t = low
    for i in range(0, num_of_iterations):
        k1 = h * function(t, w)
        k2 = h * function(t + h / 2, w + k1 / 2)
        k3 = h * function(t + h / 2, w + k2 / 2)
        t = t + h
        k4 = h * function(t, w + k3)
        w = w + (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    return w

def guassian_elimination(matrix):
    n, m = np.shape(matrix)
    for i in range(n):
        for j in range (i + 1, n):
            r = float(matrix[j, i] / matrix[i, i])
            for k in range(n + 1):
                matrix[j, k] = matrix[j, k]- r * matrix[i, k]
    ret = np.zeros(n)
    ret[n - 1] = matrix[n - 1, n] / matrix[n - 1, n - 1]
    for i in range(n - 2, -1, -1):
        ret[i] = matrix[i, n]
        for j in range(i + 1, n):
            ret[i] = ret[i] - matrix[i, j] * ret[j]
        ret[i] = ret[i] / matrix[i, i]
    return ret

def determinate(matrix):
    return np.linalg.det(matrix)

def matrixLU(matrix):
    ret = sp.linalg.lu(matrix)
    return ret[1], ret[2]

def diagnolly_dominant(matrix):
    x, y = matrix.shape
    if (x != y):
        return False
    matrix_diag = np.diag(matrix)
    for i in range(0, x):
        a = 0
        for j in range (0, y):
            if (i == j):
                continue
            a += matrix[i,j]
        if (matrix_diag[i] < a):
            return False
    return True

if __name__ == "__main__":
    euler_approx = eulers(1, 0, 2, 10)
    print("%.5f" % euler_approx, end = "\n\n")

    runge_kutta_approx = runge_kutta(1, 0, 2, 10)
    print("%.5f" % runge_kutta_approx, end = "\n\n")

    matrix1 = np.matrix([[2, -1, 1, 6],[1, 3, 1, 0],[-1, 5, 4, -3]])
    gaussian_result = guassian_elimination(matrix1)
    print(gaussian_result, end = "\n\n")

    matrix2 = np.matrix([[1,1,0,3],[2,1,-1,1],[3,-1,-1,2],[-1,2,3,-1]])
    det1 = determinate(matrix2)
    L, U = matrixLU(matrix2)
    print("%.5f" % det1, end = "\n\n")
    print(L, end = "\n\n")
    print(U, end = "\n\n")

    matrix3 = np.matrix([[9,0,5,2,1],[3,9,1,2,1],[0,1,7,2,3],[4,3,2,12,2],[4,2,3,0,8]])
    is_diag = diagnolly_dominant(matrix3)
    print(is_diag, end = "\n\n")

    matrix4 = np.matrix([[2,2,1],[2,3,0],[1,0,2]])
    det2 = determinate(matrix4)
    if (det2 > 0):
        print(True)
    else:
        print(False)
