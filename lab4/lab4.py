import numpy as np
import numpy.linalg as linalg


# Метод отражений


def solve(A, f):
    size = len(f)
    E = np.eye(size)
    for i in range(size - 1):
        s = np.array([0 if j < i else A[j, i] for j in range(size)])
        e = E[:, i]
        a = np.sqrt(np.dot(s, s))
        k = 1/ np.sqrt(2 * np.dot(s, s - a * e))
        w = np.array(k * (s - a * e))
        V = np.array([[E[i, j] - 2 * w[i] * w[j] for j in range(size)] for i in range(size)])
        A = np.dot(V, A)
        f = np.dot(V, f)
    x = np.zeros(size)
    for i in reversed(range(size)):
        x[i] = (f[i] - sum(A[i, j] * x[j] for j in range(i + 1, size))) / A[i, i]
    return x


file = open("matrix", "r")  # Чтение файла
A, b = [], []
for line in file:
    A.append([float(el) for el in line.split()[:-1]])
    b.append(float(line.split().pop()))
A = np.array(A)
b = np.array(b)
print("Решаем матричное уравнение методом отражений")
print("Основная матрица системы:")
print(A)
print("Свободные члены: ", b)
ans = solve(A, b)
print("Решение системы: ", ans)
print("Невязка: ", (np.dot(A, ans) - b))
