import numpy as np

A = np.array([[3, 1, 1], [1, 4, 1], [2, 3, 7]])
b = np.array([1, 3, 0])
tolerance = 1e-6
iterations = 50

# --------------------------1------------------------------
def norm(x, xo, tolerance):
    return (max(abs(x - xo))) / (max(abs(xo)) + tolerance)


def gauss_seidel(A, b, tolerance, iterations):
    length = len(b)

    x = np.zeros((length), dtype=np.double)

    k = 1

    while (k <= iterations):

        xo = x.copy()

        for i in range(length):

            first_sum = second_sum = 0

            for j in range(i):
                first_sum += (A[i][j] * x[j])

            for j in range(i + 1, length):
                second_sum += (A[i][j] * (xo[j]))

            x[i] = (1 / A[i][i]) * (-first_sum - second_sum + b[i])

            if (norm(x, xo, tolerance) < tolerance):
                return k

        k += 1

    return k

print(gauss_seidel(A, b, tolerance, iterations))
print()

# --------------------------2------------------------------
def jacobi(A, b, tolerance, iterations):
    length = len(b)

    x = np.zeros((length), dtype=np.double)

    k = 1

    while (k <= iterations):

        xo = x.copy()

        for i in range(length):

            sum = 0

            for j in range(length):

                if j != i:
                    sum += (A[i][j] * xo[j])

            x[i] = (1 / A[i][i]) * (-sum + b[i])

            if (norm(x, xo, tolerance) < tolerance):
                return k

        k += 1

    return k

print(jacobi(A, b, tolerance, iterations))
print()

# --------------------------3------------------------------
def custom_derivative(value):
    return (3 * value * value) - (2 * value)

def newton_raphson(initial_approximation: float, tolerance: float, sequence: str):
    # remember this is an iteration based approach...
    iteration_counter = 0
    # finds f
    x = initial_approximation
    f = eval(sequence)
    # finds f'
    f_prime = custom_derivative(initial_approximation)

    approximation: float = f / f_prime
    while (abs(approximation) >= tolerance):
        # finds f
        x = initial_approximation
        f = eval(sequence)
        # finds f'
        f_prime = custom_derivative(initial_approximation)
        # division operation
        approximation = f / f_prime
        # subtraction property
        initial_approximation -= approximation
        iteration_counter += 1
    return (iteration_counter)

initial_approximation: float = 0.5
tolerance: float = .000001
sequence: str = "x**3 - (x**2) + 2"

print(newton_raphson(initial_approximation, tolerance, sequence))
print()

# --------------------------4------------------------------

def apply_div_diff(matrix):
    size = len(matrix)
    for i in range(2, size):
        for j in range(2, i + 2):

            if j >= len(matrix[i]) or matrix[i][j] != 0:
                continue

            # something get left and diag left
            left = matrix[i][j - 1]
            diag_left = matrix[i - 1][j - 1]
            numerator = left - diag_left

            denominator = matrix[i][0] - matrix[i - j + 1][0]

            operation = numerator / denominator
            matrix[i][j] = operation
    return matrix


def hermite_interpolation(x_points, y_points, slopes):
    # main difference with hermite's method , using instances with x

    num_of_points = len(x_points)
    matrix = np.zeros((num_of_points * 2, num_of_points * 2))

    # populate x values

    index = 0
    for x in range(0, len(matrix), 2):
        matrix[x][0] = x_points[index]
        matrix[x + 1][0] = x_points[index]
        index += 1

    # prepopulate y values
    index = 0
    for x in range(0, len(matrix), 2):
        matrix[x][1] = y_points[index]
        matrix[x + 1][1] = y_points[index]
        index += 1

    # prepopulate with derivatives (every other row)
    index = 0
    for x in range(1, len(matrix), 2):
        matrix[x][2] = slopes[index]
        index += 1

    apply_div_diff(matrix)
    print(matrix)

x_points = [0.0, 1.0, 2.0]
y_points = [1.0, 2.0, 4.0]
slopes = [1.06, 1.23, 1.55]
hermite_interpolation(x_points, y_points, slopes)
print()

# --------------------------5------------------------------

def function(t, y):
    return y - (t ** 3)

def mod_eulers_method(initial_point, point_a, point_b, n):
    h = (point_b - point_a) / n
    t, w = point_a, initial_point

    for i in range(n):
        w = w + ((h / 2) * (function(t, w) + function(t + h, w + (h * function(t, w)))))
        t += h

    return w

initial_point = 0.5
point_a, point_b = 0, 3
n = 100
print(mod_eulers_method(initial_point, point_a, point_b, n))

