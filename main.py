"""
Семейство линейных систем, представленных в следующем виде зависит от p:

p-29  6 -6 -4 -3 -8 -5   5   x^_1   4p-175
6   -13 -3  5  4  3  1   7   x^_2   133
5    -5 -1  7  2  0  7   1   x^_3   110
5    -5  5  6  4 -7  4   0   x^_4   112
4     4  7 -4  9 -8 -8  -4   x^_5   17
-4    5 -4  1  0 12  0   6   x^_6   32
-3   -2 -4  2 -8 -3 16   4   x^_7   13
7     5  0  2  0 -6  8 -12   x^_8  -18

Решить линейные системы, используя программы DECOMP и SOLVE, при р= 1.0,
0.1, 0.01, 0.0001, 0.000001. Сравнить решение системы А*х_1=b с решением системы
А^Т*А*х_2 =А^Т*b, полученной из исходной, левой трансформацией Гаусса.
Проанализировать связь числа обусловленности cond и величины δ=||x_1-x_2||/||x_1||
"""
import numpy as np
from prettytable import PrettyTable
from scipy.linalg import solve
from numpy.linalg import cond, norm


def get_matrix_A(p):
    """Для получения значения матрицы A по p"""
    return np.array([
        [p - 29, 6, -6, -4, -3, -8, -5, 5],
        [6, -13, -3, 5, 4, 3, 1, 7],
        [5, -5, -1, 7, 2, 0, 7, 1],
        [5, -5, 5, 6, 4, -7, 4, 0],
        [4, 4, 7, -4, 9, -8, -8, -4],
        [-4, 5, -4, 1, 0, 12, 0, 6],
        [-3, -2, -4, 2, -8, -3, 16, 4],
        [7, 5, 0, 2, 0, -6, 8, -12]
    ])


def get_vector_b(p):
    """Для получения значения матрицы b по p"""
    return np.vstack(np.array([4 * p - 175, 133, 110, 112, 17, 32, 13, -18]))


def print_answer(x_1, x_2):
    pt = PrettyTable()
    pt.add_column('x', [f'x_{i + 1}' for i in range(0, len(x_1))])
    pt.add_column('x_1', [x[0] for x in x_1])
    pt.add_column('x_2', [x[0] for x in x_2])
    print(pt)


def main():
    for p in (10 ** -i for i in range(7)):
        # Получение матрицы A
        matrix_A = get_matrix_A(p)
        # Получение транспонированной матрицы A
        matrix_A_transpose = matrix_A.transpose()
        # Получение вектора b
        vector_b = get_vector_b(p)
        # Решение уравнения Ax=b
        result = solve(matrix_A, vector_b)
        # Решение уравнения A_t * A * x = A_t * b
        result_transform = solve(matrix_A_transpose.dot(matrix_A), matrix_A_transpose.dot(vector_b))

        print(f'Обусловленность матрицы A: {cond(matrix_A, p="fro")} {cond(matrix_A_transpose, p="fro")}')
        print_answer(result, result_transform)



if __name__ == '__main__':
    main()
