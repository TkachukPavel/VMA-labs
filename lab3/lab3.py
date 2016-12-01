import numpy as np
import numpy.linalg as linalg


# Метод левой прогонки


def solve(A, f):
    size = len(f)
    c = np.array([A[i, i] for i in range(size)])
    b = np.array([-A[i, i + 1] for i in range(size - 1)])
    a = np.array([-A[i, i - 1] for i in range(1, size)])
    y = np.zeros(size)
    eta = np.zeros(size + 1)
    ksi = np.zeros(size + 1)
    for i in reversed(range(size - 1)):
        if i == size - 2:
            ksi[i] = a[i] / c[i + 1]
        else:
            ksi[i] = a[i] / (c[i + 1] - b[i + 1] * ksi[i + 1])
    for i in reversed(range(size)):
        if i == size - 1:
            eta[i] = f[i] / c[i]
        else:
            eta[i] = (f[i] + b[i] * eta[i + 1]) / (c[i] - ksi[i + 1] * b[i])
    for i in range(size):
        if i == 0:
            y[i] = eta[i]
        else:
            y[i] =  ksi[i] * y[i - 1] + eta[i]
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
print("Свободные члены:")
print(b)
ans = solve(A, b)
print("Решение системы:")
print(ans)
print(linalg.solve(A, b))
print("Невязка:")
print(linalg.norm(np.dot(A, ans) - b))
