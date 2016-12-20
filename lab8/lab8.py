import numpy as np
import numpy.linalg as linalg


# Поиск собственных векторов методом Крылова


def eigen(matr_A):
    A = matr_A.copy()
    size = len(A)
    S = np.eye(size)
    for i in reversed(range(0, size-1)):
        M = np.eye(size)
        invM = np.eye(size)
        M[i] = [- A[i+1, j] / A[i+1, i] for j in range(size)]
        M[i, i] = 1 / A[i+1, i]
        invM[i] = [A[i+1, j] for j in range(size)]
        A = np.dot(invM, A)
        A = np.dot(A, M)
        S = np.dot(S, M)

    print("Коэфициенты характеристического многочлена\n", A[i])
    print("q_1-Sp(A) = ", abs(matr_A.trace() - A[0, 0]), "q_n-det(A) = " ,abs(linalg.det(matr_A) - A[0, size - 1]))
    p = [1] + list(A[i] * -1)  # Получили коэфициенты собственного многочлена
    eigvals = np.roots(p)
    y = [eigvals[0] ** (size - i - 1) for i in range(size)]
    return np.dot(S, y), eigvals[0]

file = open("matrix", "r")  # Чтение файла
A, b = [], []
for line in file:
    A.append([float(el) for el in line.split()[:-1]])
    b.append(float(line.split().pop()))
A = np.array(A)
A = np.dot(A, A.transpose())
b = np.array(b)
print("Ищем собственные вектор матрицы")
print("Исходная матрица:")
print(A)
ans = eigen(A)
print("Собственное значение: ", ans[1])
print("Собственный вектор (канонический базис): ", ans[0])
print("Невязка: ", np.dot(A, ans[0]) - ans[0] * ans[1])
print("Норма невязки: ", linalg.norm(np.dot(A, ans[0]) - ans[0] * ans[1]))
