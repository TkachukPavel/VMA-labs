import numpy as np
import numpy.linalg as linalg


# Метод Гаусса с выбором главного элеммента по строке

def cond(A):
    B = np.array(A)
    return np.multiply(linalg.norm(A), linalg.norm(inv(B)))

def det(A):
    return solve_with_gauss_method(A, np.zeros(len(A)), det=True)

def inv(matr):
    A = np.array(matr)
    size = len (A)
    invA = np.eye(size)
    for i in range(size):
        invA[:, i] = solve_with_gauss_method(A, invA[:, i])
    return invA

def solve_with_gauss_method(matr, matr_b, det=False):
    tmp = 1
    b = np.array(matr_b)
    A = np.array(matr)
    size = len(b)
    for i in range(size):  # Прямой ход метода Гаусса
        j_max = np.argmax([abs(el) for el in A[i]])
        A[:, [i, j_max]] = A[:, [j_max, i]]
        tmp *= A[i, i]
        b[i] /= A[i, i]
        A[i] /= A[i, i]
        for j in range(i + 1, size):
            b[j] += b[i] * (- A[j, i])
            A[j] += A[i] * (- A[j, i])
    if det:
        return tmp
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
print("Число обусловленности")
print(cond(A))
ans = solve_with_gauss_method(A, b)
print("Решение системы:")
print(ans)
print("Вектор невязки:")
print(np.dot(A, ans) - b)
print("Норма невязки:")
print(linalg.norm(np.dot(A, ans) - b))
B = inv(A)
print("Обратная матрица")
print(B)
print("Невязка обратной матрицы")
print(np.dot(A, B) - np.eye(len(A)))
print("Норма невязки")
print(linalg.norm(np.dot(A, B) - np.eye(len(A))))
print("Определитель матрицы")
print(det(A))