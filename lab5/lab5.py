import numpy as np
import numpy.linalg as linalg


def is_solvable(matr_A, matr_b): # Проверка сходимости
    A = np.array(matr_A)
    b = np.array(matr_b)
    size = len(A)
    E = np.eye(size)
    B = np.array(E - np.dot(A.transpose(), A) / linalg.norm(np.dot(A.transpose(), A)))
    sums = []
    for i in range(size):
        sums.append(sum(abs(B[i, j]) for j in range(size)))
    return max(sums) < 1

def simple_iteration(matr_A, matr_b, eps):
    A = np.array(matr_A)
    b = np.array(matr_b)
    size = len(b)
    E = np.eye(size)
    B = np.array(E - np.dot(A.transpose(), A) / linalg.norm(np.dot(A.transpose(), A)))
    g = np.array(np.dot(A.transpose(), b) / linalg.norm(np.dot(A.transpose(), A)))
    x = g;
    converge = False;
    count = 0
    while not converge:
        count +=1
        x_new = x.copy()
        x_new = np.dot(B, x) + g
        converge = linalg.norm(x - x_new) <= eps
        x = x_new

    return x, count

def seidel(matr_A, matr_b, eps):
    A = np.array(matr_A)
    b = np.array(matr_b)
    n = len(b)
    x = b.copy();
    count = 0
    converge = False
    while not converge:
        count += 1
        x_new = x.copy()
        for i in range(n):
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
print("Решаем матричное уравнение итерационными методами")
print("Основная матрица системы:")
print(A)
print("Свободные члены: ", b)
if is_solvable(A, b):
    print("Метод сходится")
    print("Метод простых итераций")
    ans = simple_iteration(A, b, 0.00001)
    print("Решаем с точность 0.000001")
    print("Решение ", ans[0], "\nПолучено за ", ans[1], " итераций")
    print("Невязка ", np.dot(A, ans[0]) - b)
    print("Норма невязки ", linalg.norm(np.dot(A, ans[0]) - b))
    print("Метод Зейделя")
    ans = seidel(A, b, 0.00001)
    print("Решаем с точность 0.000001")
    print("Решение ", ans[0], "\nПолучено за ", ans[1], " итераций")
    print("Невязка ", np.dot(A, ans[0]) - b)
    print("Норма невязки ", linalg.norm(np.dot(A, ans[0]) - b))
else:
    print("Метод расходится")


