import math
import matplotlib
import numpy as np
from prettytable import PrettyTable
import matplotlib.pyplot as plt

def renumber():

    global x_ren_L1, y_ren_L1, x_ren_L3, y_ren_L3, x_ren_L6, y_ren_L6
    x_ren_L1 = []
    y_ren_L1 = []
    x_ren_L3 = []
    y_ren_L3 = []
    x_ren_L6 = []
    y_ren_L6 = []

    for i in range(2):
        x_ren_L1.append(x[i+k])
        y_ren_L1.append(y[i+k])
    for i in range(4):
        x_ren_L3.append(x[i+k])
        y_ren_L3.append(y[i+k])
    for i in range(7):
        x_ren_L6.append(x[i+k])
        y_ren_L6.append(y[i+k])

    print("\nТаблица перенумерованных узлов (k=2)")

    print("x(k):[", end="")
    for i in range(len(x_ren_L1)):
        print('{:.2f}'.format(x_ren_L1[i]), end="")
        if i != len(x_ren_L1)-1:
            print(', ', end="")
    print(']')
    print("y(k):[", end="")
    for i in range(len(y_ren_L1)):
        print('{:.2f}'.format(y_ren_L1[i]), end="")
        if i != len(y_ren_L1)-1:
            print(', ', end="")
    print(']')
    print()

    print("x(k):[", end="")
    for i in range(len(x_ren_L3)):
        print('{:.2f}'.format(x_ren_L3[i]), end="")
        if i != len(x_ren_L3)-1:
            print(', ', end="")
    print(']')
    print("y(k):[", end="")
    for i in range(len(y_ren_L3)):
        print('{:.2f}'.format(y_ren_L3[i]), end="")
        if i != len(y_ren_L3)-1:
            print(', ', end="")
    print(']')
    print()

    print("x(k):[", end="")
    for i in range(len(x_ren_L6)):
        print('{:.2f}'.format(x_ren_L6[i]), end="")
        if i != len(x_ren_L6)-1:
            print(', ', end="")
    print(']')
    print("y(k):[", end="")
    for i in range(len(y_ren_L6)):
        print('{:.2f}'.format(y_ren_L6[i]), end="")
        if i != len(y_ren_L6)-1:
            print(', ', end="")
    print(']')
    print()

    return 0

def Aitkens_scheme():

    x_new = np.around(list(np.linspace(-2, 2, 51)), 2)
    y_new = []
    for i in range(len(x_new)):
        y_new.append(np.around((pow(x_new[i]**2, 1/3)-1), 3))

    l1 = []
    l3 = []
    l6 = []
    count = 0

    for q in range(len(x_new)):
        l = y_ren_L1.copy()
        for i in range(1, len(x_ren_L1)):
            for j in range(i, len(x_ren_L1)):
                matrix_help = np.matrix(
                    [[l[i-1], (x_ren_L1[i-1] - x_new[q])], [l[j], (x_ren_L1[j] - x_new[q])]])
                l[j] = np.around(
                    (np.linalg.det(matrix_help)/(x_ren_L1[j] - x_ren_L1[i-1])), 3)
        l1.append(np.around((l[j]), 3))

    for q in range(len(x_new)):
        l = y_ren_L3.copy()
        for i in range(1, len(x_ren_L3)):
            for j in range(i, len(x_ren_L3)):
                count = count+1
                matrix_help = np.matrix(
                    [[l[j-1], (x_ren_L3[count-1] - x_new[q])], [l[j], (x_ren_L3[j] - x_new[q])]])
                l[j-1] = np.around(
                    (np.linalg.det(matrix_help)/(x_ren_L3[j] - x_ren_L3[count-1])), 3)
            count = 0
            l_help = l.copy()

            for t in range(i, len(x_ren_L3)):
                l[t] = l_help[t-1]
        l3.append(np.around((l[j]), 3))

    for q in range(len(x_new)):
        l = y_ren_L6.copy()
        for i in range(1, len(x_ren_L6)):
            for j in range(i, len(x_ren_L6)):
                count = count+1
                matrix_help = np.matrix(
                    [[l[j-1], (x_ren_L6[count-1] - x_new[q])], [l[j], (x_ren_L6[j] - x_new[q])]])
                l[j-1] = np.around(
                    (np.linalg.det(matrix_help)/(x_ren_L6[j] - x_ren_L6[count-1])), 3)
            count = 0
            l_help = l.copy()

            for t in range(i, len(x_ren_L6)):
                l[t] = l_help[t-1]
        l6.append(np.around((l[j]), 3))

    table = PrettyTable()
    table.field_names = ['x', 'y(x)', 'L1(x)', 'L3(x)', 'L6(x)']
    for i in range(len(x_new)):
        table.add_row([x_new[i], y_new[i], l1[i], l3[i], l6[i]])
    print(table)
    plt.subplot(2, 2, 1)
    plt.xlim([-2, 2])
    plt.ylim([-2, 2])
    plt.grid(color='green')
    plt.plot(x_new, y_new, 'b-', label='y(x)')
    plt.plot(x_new, l1, '-.k', label='$L_{1}$(x)')
    plt.plot(x_ren_L1, y_ren_L1, 'ro', label='$y_{k}$')
    plt.legend(bbox_to_anchor=(-0.10, 0.6))
    plt.subplot(2, 2, 2)
    plt.xlim([-2, 2])
    plt.ylim([-2, 2])
    plt.grid(color='green')
    plt.plot(x_new, y_new, 'b-', label='y(x)')
    plt.plot(x_new, l3, '-.k', label='$L_{3}$(x)')
    plt.plot(x_ren_L3, y_ren_L3, 'ro', label='$y_{k}$')
    plt.legend(bbox_to_anchor=(1.25, 0.6))
    plt.subplot(2, 2, 3)
    plt.xlim([-2, 2])
    plt.ylim([-2, 2])
    plt.grid(color='green')
    plt.plot(x_new, y_new, 'b-', label='y(x)')
    plt.plot(x_new, l6, '-.k', label='$L_{6}$(x)')
    plt.plot(x_ren_L6, y_ren_L6, 'ro', label='$y_{k}$')
    plt.legend(bbox_to_anchor=(-0.10, 0.6))
    plt.show()

    return 0

k = 2

print("\nТаблица узлов интерполирования")

x = np.around(list(np.linspace(-2, 2, 13)), 2)
print("x(k):[", end="")
for i in range(len(x)):
    print('{:.2f}'.format(x[i]), end="")
    if i != len(x)-1:
        print(', ', end="")
print(']')

y = []
for i in range(len(x)):
    y.append(np.around((pow(x[i]**2, 1/3)-1), 2))

print("y(k):[", end="")
for i in range(len(y)):
    print('{:.2f}'.format(y[i]), end="")
    if i != len(y)-1:
        print(', ', end="")
print(']')

renumber()
Aitkens_scheme()
