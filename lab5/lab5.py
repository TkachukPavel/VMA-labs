import numpy as np
import numpy.linalg as linalg


def is_solvable(A, b):		# Проверка сходимости
    size = len(A)
    E = np.eye(size)
    B = np.array(E - np.dot(A, np.transpose(A)) / linalg.norm(np.dot(A, np.transpose(A))))
    sums = []
    for i in range(size):
        sums.append(sum(abs(B[i, j]) for j in range(size)))
    return max(sums) < 1


def seidel(A, b, eps):
    n = len(b)
    r = range(n)
    x = b
    count = 0
    converge = False
    while not converge:
        count += 1
        x_new = x.copy()
        for i in r:
            s1 = sum(A[i][j] * x_new[j] for j in range(i))
            s2 = sum(A[i][j] * x[j] for j in range(i + 1, n))
            x_new[i] = (b[i] - s1 - s2) / A[i][i]
        converge = linalg.norm(x - x_new) <= eps
        x = x_new

    return x, count


file = open("matrix", "r")  # Чтение файла
A, b = [], []
for line in file:
    A.append([float(el) for el in line.split()[:-1]])
    b.append(float(line.split().pop()))
A = np.array(A)
b = np.array(b)
print("Решаем матричное уравнение методом Гаусса-Зейделя")
print("Основная матрица системы:")
print(A)
print("Свободные члены: ", b)
if is_solvable(A, b):
    print("Метод сходится")
    ans = seidel(A, b, 0.00001)
    print("Решаем с точность 0.000001")
    print("Решение ", ans[0], "\nПолучено за ", ans[1], " итераций")
    print("Невязка ", np.dot(A, ans[0]) - b)
    print("Норма невязки ", linalg.norm(np.dot(A, ans[0]) - b))
else:
    print("Метод расходится")


