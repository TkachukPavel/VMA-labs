import numpy as np
import numpy.linalg as linalg


def is_solvable(A, b):  # Проверка сходимости
    size = len(A)
    E = np.eye(size)
    B = np.array(E - np.dot(A, np.transpose(A)) / linalg.norm(np.dot(A, np.transpose(A))))
    sums = []
    for i in range(size):
        sums.append(sum(abs(B[i, j]) for j in range(size)))
    return max(sums) < 1


def bottom_relax(A, b, eps):  # Нижняя релаксация
    size = len(A)
    count = 0
    converge = False
    x = b
    w = 0.5
    while not converge:
        count += 1
        x_new = x.copy()
        for i in range(size):
            s1 = sum(A[i][j] * x_new[j] for j in range(i))
            s2 = sum(A[i][j] * x[j] for j in range(i + 1, size))
            x_new[i] = (1 - w) * x[i] + (w * (b[i] - s1 - s2) / A[i][i])
        converge = linalg.norm(x - x_new) <= w * eps
        x = x_new
    return x, count


def grad_descent(A, b, eps):  # Градиентный спуск
    b = np.dot(np.transpose(A), b)
    A = np.dot(np.transpose(A), A)
    x = b
    r = np.dot(A, x) - b
    count = 0
    converge = False
    while not converge:
        count += 1
        r = np.dot(A, x) - b
        x_new = x - np.dot(r, r) * r / np.dot(np.dot(A, r), r)
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
    print("Решаем с точность 0.00001")
    print("Метод градиентного спуска")
    ans = grad_descent(A, b, 0.00001)
    print("Решение ", ans[0], "\nПолучено за ", ans[1], " итераций")
    print("Невязка ", np.dot(A, ans[0]) - b)
    print("Норма невязки ", linalg.norm(np.dot(A, ans[0]) - b))

    print("Метод нижней релаксации")
    ans = bottom_relax(A, b, 0.00001)
    print("Решение ", ans[0], "\nПолучено за ", ans[1], " итераций")
    print("Невязка ", np.dot(A, ans[0]) - b)
    print("Норма невязки ", linalg.norm(np.dot(A, ans[0]) - b))
else:
    print("Метод расходится")
