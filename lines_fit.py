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

def plot_ab(aaCol, bbCol, noise_ratio):
    global a, b
    with open(f"a_b_{noise_ratio}.txt", "w") as f:
        length = min(10000, len(aaCol))
        for i in range(10000):
            f.write(f"{i} {aaCol[i]:.6f} {bbCol[i]:.6f}\n")
    plt.close("all")
    plt.figure(figsize=(10, 6))
    plt.scatter(np.arange(len(aaCol)), aaCol, s=1, color="blue", label="a")
    plt.scatter(np.arange(len(bbCol)), bbCol, s=1, color="red", label="b")
    plt.xlabel("epoch")
    plt.ylabel("a/b")
    plt.title(f"a={a:.6f} b={b:.6f} (noise_ratio = {noise_ratio})")
    plt.legend()
    plt.savefig(f"a_b_{noise_ratio}.png")
    plt.close("all")

    plt.close("all")
    plt.figure(figsize=(10, 6))
    aaCol = aaCol[:1000]
    bbCol = bbCol[:1000]
    plt.scatter(np.arange(len(aaCol)), aaCol, s=1, color="blue", label="a")
    plt.scatter(np.arange(len(bbCol)), bbCol, s=1, color="red", label="b")
    plt.xlabel("epoch")
    plt.ylabel("a/b")
    plt.title(f"a={a:.6f} b={b:.6f} (noise_ratio = {noise_ratio})")
    plt.legend()
    plt.savefig(f"1000epoch_a_b_{noise_ratio}.png")
    plt.close("all")

def grad_solution(points):
    global a, b
    a_0 = 0.0001
    b_0 = -0.0001

    learning_rate = 0.00001
    epoch = 200000
    aaCol = []
    bbCol = []
    for e in range(epoch):
        if e == 190000:
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
                return aa, bb, np.array(aaCol), np.array(bbCol)
            a_0 = aa
            b_0 = bb

        aaCol.append(aa)
        bbCol.append(bb)
    return a_0, b_0, np.array(aaCol), np.array(bbCol)

if __name__ == "__main__":
    noise_ratio0 = 0
    print(f"noise_ratio = {noise_ratio0}")
    points = fit_point(noise_ratio0)
    a0, b0 = partial_solution(points)
    sa = abs(a0 - a); sa = f"partial_solution: {a0:.6f} -> {a:.6f}"
    sb = abs(b0 - b); sb = f"partial_solution: {b0:.6f} -> {b:.6f}"
    print(sa, sb)

    a1, b1, aaCol, bbCol = grad_solution(points)
    sa = abs(a1 - a); sa = f"grad_solution: {a1:.6f} -> {a:.6f}"
    sb = abs(b1 - b); sb = f"grad_solution: {b1:.6f} -> {b:.6f}"
    print(sa, sb)
    
    noise_ratio1 = 30
    print(f"noise_ratio = {noise_ratio1}")
    points = fit_point(noise_ratio1)
    a0, b0 = partial_solution(points)
    sa = abs(a0 - a); sa = f"partial_solution: {a0:.6f} -> {a:.6f}"
    sb = abs(b0 - b); sb = f"partial_solution: {b0:.6f} -> {b:.6f}"
    print(sa, sb)

    plot_ab(aaCol, bbCol, noise_ratio0)
    a1, b1, aaCol, bbCol = grad_solution(points)
    plot_ab(aaCol, bbCol, noise_ratio1)
    sa = abs(a1 - a); sa = f"grad_solution: {a1:.6f} -> {a:.6f}"
    sb = abs(b1 - b); sb = f"grad_solution: {b1:.6f} -> {b:.6f}"
    print(sa, sb)
