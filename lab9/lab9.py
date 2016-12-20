import numpy as np
import numpy.linalg as linalg


# Поиск собственных векторов методом Крылова

def leberie(matr_A):
    A = matr_A.copy()
    n = len(A)
    s = np.zeros(n + 1)
    for i in range(1, n + 1):
        s[i] = sum(A[j, j] for j in range(n))
        A = np.dot(A, matr_A)
    p = np.zeros(n + 1)
    for i in range(1, n + 1):
        p[i] = (s[i] - sum(s[j]*p[i-j] for j in range(i))) / i
    print("Коэфициенты характеристического многочлена\n", p[1:])
    print("q_1-Sp(A) = ", abs(A.trace() - p[1]), "q_n-det(A) = ", abs(linalg.det(A) - p[n]))
    return p[1:]

def faddeev(matr_A):
    A = matr_A.copy()
    n = len(A)
    q = np.zeros(n + 1)
    resB = np.eye(n)
    E = np.eye(n)
    for i in range(1, n + 1):
        q[i] = sum(A[i, i] for i in range(n)) / i
        B = A - q[i]*E
        resB[:,i-1] = B[:,0].copy()
        A = np.dot(matr_A, B)
    print("Коэфициенты характеристического многочлена\n", q[1:])
    print("q_1-Sp(A) = ", abs(matr_A.trace() - q[1]), "q_n-det(A) = ", abs(linalg.det(matr_A) - q[n]))
    p = [1] + list(q[1:] * -1)
    eigvals = np.roots(p)
    v = (eigvals[0] ** (n - 1))*E[:, 0]
    for i in range(n - 1):
        v += (eigvals[0] ** i) * resB[:,(n - i - 2)]
    return v, eigvals[0]

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
print("Метод Леверье")
leberie(A)
print("Метод Фаддеева")
ans = faddeev(A)
print("Собственное значение: ", ans[1])
print("Собственный вектор: ", ans[0])
print("Невязка: ", np.dot(A, ans[0]) - ans[0] * ans[1])
print("Норма невязки: ", linalg.norm(np.dot(A, ans[0]) - ans[0] * ans[1]))
