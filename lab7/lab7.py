import numpy as np
import numpy.linalg as linalg


# Метод отражений


def eigen(A):
    size = len(A)
    C = np.eye(size)
    for i in range(1, size):
        C[:, i] = np.dot(A, C[:, i - 1])
    c = np.dot(A, C[:, size - 1])
    C = C[:, ::-1]
    q = linalg.solve(C, c)
    print(q)
    p = [1] + list(q * -1)
    print(p)

    eigvals = np.roots(p)

    beta = np.zeros(size)
    beta[0] = 1
    for i in range(1, size):
        beta[i] = beta[i - 1] * eigvals[0] + p[i]
    return sum([beta[i] * C[:, i] for i in range(size)]), eigvals[0]


file = open("matrix", "r")  # Чтение файла
A, b = [], []
for line in file:
    A.append([float(el) for el in line.split()[:-1]])
    b.append(float(line.split().pop()))
A = np.array(A)
A = np.dot(A, A.transpose())
b = np.array(b)
print("Ищем собственные вектора матрицы")
print("Исходная матрица:")
print(A)
ans = eigen(A)
print("Собственное значение: ", ans[1])
print("Собственный вектор (канонический базис): ", ans[0])
print("Невязка: ", np.dot(A, ans[0]) - ans[0] * ans[1])
print("Норма невязки: ", linalg.norm(np.dot(A, ans[0]) - ans[0] * ans[1]))
