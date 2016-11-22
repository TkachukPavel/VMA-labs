import numpy as np
import numpy.linalg as linalg


# Метод Гаусса с выбором главного элеммента по строке

def solve_with_gauss_method(A, b):
    size = len(b)
    for i in range(size):  # Прямой ход метода Гаусса
        j_max = np.argmax([abs(el) for el in A[i]])
        A[:, [i, j_max]] = A[:, [j_max, i]]
        b[i] /= A[i, i]
        A[i] /= A[i, i]
        for j in range(i + 1, size):
            b[j] += b[i] * (- A[j, i])
            A[j] += A[i] * (- A[j, i])
    x = np.zeros(size)
    for i in reversed(range(size)):  # Обратный ход метода Гаусса
        x[i] = b[i] - sum(A[i, j]*x[j] for j in range(i + 1, size))
    return x


# Основная программа

file = open("matrix", "r")  # Чтение файла
A, b = [], []
for line in file:
    A.append([float(el) for el in line.split()[:-1]])
    b.append(float(line.split().pop()))
A = np.array(A)
b = np.array(b)

print("Решаем матричное уравнение методом Гаусса с выбором главного элемента по строке")
print("Основная матрица системы:")
print(A)
print("Свободные члены:")
print(b)
ans = solve_with_gauss_method(A, b)
print("Решение системы:")
print(ans)
print("Невязка:")
print(linalg.norm(np.dot(A, ans) - b))
