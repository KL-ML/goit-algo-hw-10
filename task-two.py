import scipy.integrate as sci
import numpy as np
import matplotlib.pyplot as plt
import numpy as np

# Визначення функції та межі інтегрування
def f(x):
    return x ** 2

a = 0  # Нижня межа
b = 2  # Верхня межа

# Створення діапазону значень для x
x = np.linspace(-0.5, 2.5, 400)
y = f(x)

# Створення графіка
fig, ax = plt.subplots()

# Малювання функції
ax.plot(x, y, 'r', linewidth=2)

# Заповнення області під кривою
ix = np.linspace(a, b)
iy = f(ix)
ax.fill_between(ix, iy, color='gray', alpha=0.3)

# Налаштування графіка
ax.set_xlim([x[0], x[-1]])
ax.set_ylim([0, max(y) + 0.1])
ax.set_xlabel('x')
ax.set_ylabel('f(x)')

# Додавання меж інтегрування та назви графіка
ax.axvline(x=a, color='gray', linestyle='--')
ax.axvline(x=b, color='gray', linestyle='--')
ax.set_title('Графік інтегрування f(x) = x^2 від ' + str(a) + ' до ' + str(b))
plt.grid()
plt.show()


# Визначення функції та межі інтегрування
def f(x):
    return x**2


a = 0  # Нижня межа
b = 2  # Верхня межа


def mc_integral(a, b, num_samples=10000):
    x = np.random.uniform(a, b, num_samples)
    y = np.random.uniform(0, f(b), num_samples)

    points_under_curve = sum(y <= f(x))
    area_ratio = points_under_curve / num_samples

    total_area = (b - a) * f(b)
    return total_area * area_ratio


if __name__ == "__main__":
    print(f"Numeric integral: {sci.quad(f, a, b)[0]}")

    num_samples = (100, 1000, 10000, 100000, 1000000, 10000000)
    for sample in num_samples:
        print(
            f"Monte Carlo integral ({sample}): {mc_integral(a, b, num_samples=sample)}"
        )