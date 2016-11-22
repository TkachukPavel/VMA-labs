import numpy as np
import numpy.linalg as linalg


# Метод квадратного корня


def solve(A, b):
    size = len(b)
    S = np.zeros((size, size))
    for i in range(size):
        S[i, i] = np.sqrt(A[i, i] - sum(([(S[k, i] ** 2) for k in range(i)])))
        for j in range(i + 1, size):
            S[i, j] = (A[i, j] - sum([S[k, i] * S[k, j] for k in range(i)])) / S[i, i]
    y = np.zeros(size)
    for i in range(size):
        y[i] = (b[i] - sum([S[k, i] * y[k] for k in range(i)])) / S[i, i]
    x = np.zeros(size)
    for i in reversed(range(size)):
        x[i] = (y[i] - sum(S[i, k] * x[k] for k in range(i + 1, size))) / S[i, i]
    return x


# Основная программа

file = open("matrix", "r")  # Чтение файла
A, b = [], []
for line in file:
    A.append([float(el) for el in line.split()[:-1]])
    b.append(float(line.split().pop()))
A = np.array(A)
b = np.array(b)
b = np.dot(np.transpose(A), b)
A = np.dot(np.transpose(A), A)
print("Решаем матричное уравнение методом квадратного корня")
print("Основная матрица системы:")
print(A)
print("Свободные члены:")
print(b)
ans = solve(A, b)
print("Решение системы:")
print(ans)
print("Невязка:")
print(linalg.norm(np.dot(A, ans) - b))
