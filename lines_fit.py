import os
import numpy as np
import matplotlib.pyplot as plt

# y = 10.1234512345 * x + 9.916238521
a = 2.1234512345
b = 100.916238521
number = 300

def fit_point(noise_ratio):
    points = []

    np.random.seed(999911369)
    noise = np.random.normal(0, 1, (number)) * noise_ratio
    np.random.seed(999911369)
    rand = np.random.randint(0, 300, number)
    for i in range(number):
        points.append([rand[i], a * rand[i] + b + noise[i]])

    points = np.array(points, dtype = np.float64)

    plt.scatter(points[:, 0], points[:, 1], s=1)
    plt.savefig("fig.png")
    plt.xlabel("x")
    plt.ylabel("y")

    return points

def partial_solution(points):
    sxy = 0; sx = 0; sy = 0; sxx = 0;
    for i in range(number):
        sxy += points[i][0] * points[i][1]
        sxx += points[i][0] * points[i][0]
        sx += points[i][0]
        sy += points[i][1]

    Delta = number * sxx - sx * sx
    a_0 = (number * sxy - sx * sy) / Delta
    b_0 = (sy * sxx - sx * sxy) / Delta

    return a_0, b_0

def grad_solution(points):
    a_0 = 0.0001
    b_0 = -0.0001

    learning_rate = 0.00001
    epoch = 100000
    for e in range(epoch):
        if e == epoch - 3:
            learning_rate = learning_rate / 10
        for i in range(number):
            x = points[i][0]
            y = points[i][1]
            tmp = y - (a_0 * x + b_0)
            delta_a = -2 * x * tmp
            delta_b = -2 * tmp

            aa = a_0 - learning_rate * delta_a
            bb = b_0 - learning_rate * delta_b

            if abs(aa - a) < 0.0000001 and abs(bb - b) < 0.0000001:
                return aa, bb
            a_0 = aa
            b_0 = bb
    return a_0, b_0

if __name__ == "__main__":
    print("noise_ratio = 0")
    points = fit_point(0)
    a0, b0 = partial_solution(points)
    sa = abs(a0 - a); sa = f"partial_solution: {a0:.6f} -> {a:.6f}"
    sb = abs(b0 - b); sb = f"partial_solution: {b0:.6f} -> {b:.6f}"
    print(sa, sb)

    a1, b1 = grad_solution(points)
    sa = abs(a1 - a); sa = f"grad_solution: {a1:.6f} -> {a:.6f}"
    sb = abs(b1 - b); sb = f"grad_solution: {b1:.6f} -> {b:.6f}"
    print(sa, sb)
    

    print("noise_ratio = 30")
    points = fit_point(30)
    a0, b0 = partial_solution(points)
    sa = abs(a0 - a); sa = f"partial_solution: {a0:.6f} -> {a:.6f}"
    sb = abs(b0 - b); sb = f"partial_solution: {b0:.6f} -> {b:.6f}"
    print(sa, sb)
    
    a1, b1 = grad_solution(points)
    sa = abs(a1 - a); sa = f"grad_solution: {a1:.6f} -> {a:.6f}"
    sb = abs(b1 - b); sb = f"grad_solution: {b1:.6f} -> {b:.6f}"
    print(sa, sb)
