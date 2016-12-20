import numpy as np
import numpy.linalg as linalg

def eigvals(matr_A, eps):
    A = matr_A.copy()
    n = len(A)
    y = np.array([1 for i in range(n)])
    converge = False
    count = 0
    prev = 1
    while not converge:
        count +=1
        y_new = np.dot(A, y)
        next = sum(y_new[i] / y[i] for i in range(n)) / n
        converge = abs(next - prev) <= eps
        prev = next
        y = y_new / linalg.norm(y_new)
    return y, prev, count

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
ans = eigvals(A, 0.00001)
print("Собственное значение: ", ans[1])
print("Собственный вектор: ", ans[0])
print("Количество итераций: ", ans[2])
print("Невязка: ", np.dot(A, ans[0]) - ans[0] * ans[1])
print("Норма невязки: ", linalg.norm(np.dot(A, ans[0]) - ans[0] * ans[1]))
