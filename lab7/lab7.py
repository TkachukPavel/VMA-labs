import numpy as np
import numpy.linalg as linalg


# Поиск собственных векторов методом Крылова


def eigen(A):
    size = len(A)
    C = np.eye(size)
    for i in range(1, size):
        C[:, i] = np.dot(A, C[:, i - 1])
    c = np.dot(A, C[:, size - 1])
    C = C[:, ::-1]
    q = linalg.solve(C, c)
    print("Коэфициенты характеристического многочлена\n", q)
    print("q_1-Sp(A) = ", abs(A.trace() - q[0]), "q_n-det(A) = " ,abs(linalg.det(A) - q[size - 1]))
    p = [1] + list(q * -1)	# Получили коэфициенты собственного многочлена
    eigvals = np.roots(p)	# Находим его корни			

    beta = np.zeros(size)	# Вычисляем собственный вектор
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
print("Ищем собственные вектор матрицы")
print("Исходная матрица:")
print(A)
ans = eigen(A)
print("Собственное значение: ", ans[1])
print("Собственный вектор (канонический базис): ", ans[0])
print("Невязка: ", np.dot(A, ans[0]) - ans[0] * ans[1])
print("Норма невязки: ", linalg.norm(np.dot(A, ans[0]) - ans[0] * ans[1]))
v = ans[0]
print(v / v[len(v) - 1])
