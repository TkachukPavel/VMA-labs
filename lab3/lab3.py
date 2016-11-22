import numpy as np
import numpy.linalg as linalg


# Метод левой прогонки


def solve(A, f):
    size = len(f)
    b = np.zeros(size)
    a = np.zeros(size)
    c = np.zeros(size)
    for i in range(size):
        c[i] = A[i, i]
        if i != size - 1:
            b[i] = -A[i, i + 1]
        if i != 0:
            a[i] = -A[i, i - 1]
    ksi = np.zeros(size + 1)
    for i in reversed(range(size)):
        ksi[i] = (a[i]) / (c[i] - ksi[i + 1] * b[i])
    eta = np.zeros(size + 1)
    for i in reversed(range(size)):
        eta[i] = (f[i] + b[i] * eta[i + 1]) / (c[i] - ksi[i + 1] * b[i])
    y = np.zeros(size)
    for i in range(size):
        y[i] = ksi[i] * y[i - 1] + eta[i]
    return y


file = open("matrix", "r")  # Чтение файла
A, b = [], []
for line in file:
    A.append([float(el) for el in line.split()[:-1]])
    b.append(float(line.split().pop()))
A = [[A[i][j] if np.abs(i - j) < 2 else 0 for j in range(len(b))] for i in range(len(b))]
A = np.array(A)
b = np.array(b)

print("Решаем матричное уравнение методом квадратного корня")
print("Основная матрица системы:")
print(A)
print("Свободные члены: ", b)
ans = solve(A, b)
print("Решение системы: ", ans)
print("Невязка: ", np.dot(A, ans) - b)
print("Норма невязки: ", linalg.norm(np.dot(A, ans) - b))