import numpy as np
import scipy.stats

print("Вариант 302 Бойчеко Дмитрий IB-83 условия")

def get_sum(*args):
    summa = 0
    try:
        if args[0] == "y":
            if len(args) == 1:  # только у
                summa = sum(my_list)
            else:
                for j in range(N):
                    sum_i_temp = 1
                    for i in range(len(args) - 1):  # цикл для элементов, кроме первой "у"
                        sum_i_temp *= x_matrix[j][args[i + 1] - 1]
                    sum_i_temp *= my_list[j]  # умножение на "у"
                    summa += sum_i_temp

        elif len(args) == 1:
            args = args[0] - 1
            for obj in x_matrix:
                summa += obj[args]
        else:
            for obj in x_matrix:
                sum_i_temp = 1
                for i in range(len(args)):
                    sum_i_temp *= obj[
                        args[i] - 1]  # перемножаем все Х из кортежа, для квадрата дважды добавляем в кортеж Х
                summa += sum_i_temp

    except:
        print("def error")
    return summa
#Вариант
x1min, x1max = -10, 50
x2min, x2max = 20, 45
x3min, x3max = -10, 15
m = 3
N = 8

mx_max = (x1max + x2max + x3max) / 3
mx_min = (x1min + x2min + x3min) / 3
y_max = mx_max + 200
y_min = mx_min + 200

y_list = np.random.randint(y_min, y_max, (N, m))  # create tab 8*3 random 'y' in [y_min; y_max]

x_matrix = [
    [x1min, x2min, x3min],
    [x1min, x2min, x3max],
    [x1min, x2max, x3min],
    [x1min, x2max, x3max],
    [x1max, x2min, x3min],
    [x1max, x2min, x3max],
    [x1max, x2max, x3min],
    [x1max, x2max, x3max]
]

while 1:
    def y_add_el():  # функция увеличения m
        for obj in y_list:
            obj.append(np.random.randint(y_min, y_max))


    my_list = []
    mx1 = 0
    mx2 = 0
    mx3 = 0

    for obj in y_list:
        my_list.append(sum(obj) / len(obj))

    for obj in x_matrix:
        mx1 += obj[0]
        mx2 += obj[1]
        mx3 += obj[2]

    mx1 /= 8
    mx2 /= 8
    mx3 /= 8
    my = sum(my_list) / 8

    """Coefficients"""
    mi0 = [N, get_sum(1), get_sum(2), get_sum(3), get_sum(1, 2), get_sum(1, 3), get_sum(2, 3), get_sum(1, 2, 3)]
    mi1 = [get_sum(1), get_sum(1, 1), get_sum(1, 2), get_sum(1, 3), get_sum(1, 1, 2), get_sum(1, 1, 3),
           get_sum(1, 2, 3), get_sum(1, 1, 2, 3)]
    mi2 = [get_sum(2), get_sum(1, 2), get_sum(2, 2), get_sum(2, 3), get_sum(1, 2, 2), get_sum(1, 2, 3),
           get_sum(2, 2, 3), get_sum(1, 2, 2, 3)]
    mi3 = [get_sum(3), get_sum(1, 3), get_sum(2, 3), get_sum(3, 3), get_sum(1, 2, 3), get_sum(1, 3, 3),
           get_sum(2, 3, 3), get_sum(1, 2, 3, 3)]
    mi4 = [get_sum(1, 2), get_sum(1, 1, 2), get_sum(1, 2, 2), get_sum(1, 2, 3), get_sum(1, 1, 2, 2),
           get_sum(1, 1, 2, 3), get_sum(1, 2, 2, 3), get_sum(1, 1, 2, 2, 3)]
    mi5 = [get_sum(1, 3), get_sum(1, 1, 3), get_sum(1, 2, 3), get_sum(1, 3, 3), get_sum(1, 1, 2, 3),
           get_sum(1, 1, 3, 3), get_sum(1, 2, 3, 3), get_sum(1, 1, 2, 3, 3)]
    mi6 = [get_sum(2, 3), get_sum(1, 2, 3), get_sum(2, 2, 3), get_sum(2, 3, 3), get_sum(1, 2, 2, 3),
           get_sum(1, 2, 3, 3), get_sum(2, 2, 3, 3), get_sum(1, 2, 2, 3, 3)]
    mi7 = [get_sum(1, 2, 3), get_sum(1, 1, 2, 3), get_sum(1, 2, 2, 3), get_sum(1, 2, 3, 3), get_sum(1, 1, 2, 2, 3),
           get_sum(1, 1, 2, 3, 3), get_sum(1, 2, 2, 3, 3), get_sum(1, 1, 2, 2, 3, 3)]

    k0, k1, k2, k3, k4, k5, k6, k7 = get_sum("y"), get_sum("y", 1), get_sum("y", 2), get_sum("y", 3), \
                                     get_sum("y", 1, 2), get_sum("y", 1, 3), get_sum("y", 2, 3), get_sum("y", 1, 2, 3)
    denominator = np.linalg.det([
        mi0,
        mi1,
        mi2,
        mi3,
        mi4,
        mi5,
        mi6,
        mi7
    ])

    numerator_b0 = np.linalg.det([
        [k0, mi0[1], mi0[2], mi0[3], mi0[4], mi0[5], mi0[6], mi0[7]],
        [k1, mi1[1], mi1[2], mi1[3], mi1[4], mi1[5], mi1[6], mi1[7]],
        [k2, mi2[1], mi2[2], mi2[3], mi2[4], mi2[5], mi2[6], mi2[7]],
        [k3, mi3[1], mi3[2], mi3[3], mi3[4], mi3[5], mi3[6], mi3[7]],
        [k4, mi4[1], mi4[2], mi4[3], mi4[4], mi4[5], mi4[6], mi4[7]],
        [k5, mi5[1], mi5[2], mi5[3], mi5[4], mi5[5], mi5[6], mi5[7]],
        [k6, mi6[1], mi6[2], mi6[3], mi6[4], mi6[5], mi6[6], mi6[7]],
        [k7, mi7[1], mi7[2], mi7[3], mi7[4], mi7[5], mi7[6], mi7[7]]
    ])

    numerator_b1 = np.linalg.det([
        [mi0[0], k0, mi0[2], mi0[3], mi0[4], mi0[5], mi0[6], mi0[7]],
        [mi1[0], k1, mi1[2], mi1[3], mi1[4], mi1[5], mi1[6], mi1[7]],
        [mi2[0], k2, mi2[2], mi2[3], mi2[4], mi2[5], mi2[6], mi2[7]],
        [mi3[0], k3, mi3[2], mi3[3], mi3[4], mi3[5], mi3[6], mi3[7]],
        [mi4[0], k4, mi4[2], mi4[3], mi4[4], mi4[5], mi4[6], mi4[7]],
        [mi5[0], k5, mi5[2], mi5[3], mi5[4], mi5[5], mi5[6], mi5[7]],
        [mi6[0], k6, mi6[2], mi6[3], mi6[4], mi6[5], mi6[6], mi6[7]],
        [mi7[0], k7, mi7[2], mi7[3], mi7[4], mi7[5], mi7[6], mi7[7]]
    ])

    numerator_b2 = np.linalg.det([
        [mi0[0], mi0[1], k0, mi0[3], mi0[4], mi0[5], mi0[6], mi0[7]],
        [mi1[0], mi1[1], k1, mi1[3], mi1[4], mi1[5], mi1[6], mi1[7]],
        [mi2[0], mi2[1], k2, mi2[3], mi2[4], mi2[5], mi2[6], mi2[7]],
        [mi3[0], mi3[1], k3, mi3[3], mi3[4], mi3[5], mi3[6], mi3[7]],
        [mi4[0], mi4[1], k4, mi4[3], mi4[4], mi4[5], mi4[6], mi4[7]],
        [mi5[0], mi5[1], k5, mi5[3], mi5[4], mi5[5], mi5[6], mi5[7]],
        [mi6[0], mi6[1], k6, mi6[3], mi6[4], mi6[5], mi6[6], mi6[7]],
        [mi7[0], mi7[1], k7, mi7[3], mi7[4], mi7[5], mi7[6], mi7[7]]
    ])

    numerator_b3 = np.linalg.det([
        [mi0[0], mi0[1], mi0[2], k0, mi0[4], mi0[5], mi0[6], mi0[7]],
        [mi1[0], mi1[1], mi1[2], k1, mi1[4], mi1[5], mi1[6], mi1[7]],
        [mi2[0], mi2[1], mi2[2], k2, mi2[4], mi2[5], mi2[6], mi2[7]],
        [mi3[0], mi3[1], mi3[2], k3, mi3[4], mi3[5], mi3[6], mi3[7]],
        [mi4[0], mi4[1], mi4[2], k4, mi4[4], mi4[5], mi4[6], mi4[7]],
        [mi5[0], mi5[1], mi5[2], k5, mi5[4], mi5[5], mi5[6], mi5[7]],
        [mi6[0], mi6[1], mi6[2], k6, mi6[4], mi6[5], mi6[6], mi6[7]],
        [mi7[0], mi7[1], mi7[2], k7, mi7[4], mi7[5], mi7[6], mi7[7]]
    ])

    numerator_b12 = np.linalg.det([
        [mi0[0], mi0[1], mi0[2], mi0[3], k0, mi0[5], mi0[6], mi0[7]],
        [mi1[0], mi1[1], mi1[2], mi1[3], k1, mi1[5], mi1[6], mi1[7]],
        [mi2[0], mi2[1], mi2[2], mi2[3], k2, mi2[5], mi2[6], mi2[7]],
        [mi3[0], mi3[1], mi3[2], mi3[3], k3, mi3[5], mi3[6], mi3[7]],
        [mi4[0], mi4[1], mi4[2], mi4[3], k4, mi4[5], mi4[6], mi4[7]],
        [mi5[0], mi5[1], mi5[2], mi5[3], k5, mi5[5], mi5[6], mi5[7]],
        [mi6[0], mi6[1], mi6[2], mi6[3], k6, mi6[5], mi6[6], mi6[7]],
        [mi7[0], mi7[1], mi7[2], mi7[3], k7, mi7[5], mi7[6], mi7[7]]
    ])

    numerator_b13 = np.linalg.det([
        [mi0[0], mi0[1], mi0[2], mi0[3], mi0[4], k0, mi0[6], mi0[7]],
        [mi1[0], mi1[1], mi1[2], mi1[3], mi1[4], k1, mi1[6], mi1[7]],
        [mi2[0], mi2[1], mi2[2], mi2[3], mi2[4], k2, mi2[6], mi2[7]],
        [mi3[0], mi3[1], mi3[2], mi3[3], mi3[4], k3, mi3[6], mi3[7]],
        [mi4[0], mi4[1], mi4[2], mi4[3], mi4[4], k4, mi4[6], mi4[7]],
        [mi5[0], mi5[1], mi5[2], mi5[3], mi5[4], k5, mi5[6], mi5[7]],
        [mi6[0], mi6[1], mi6[2], mi6[3], mi6[4], k6, mi6[6], mi6[7]],
        [mi7[0], mi7[1], mi7[2], mi7[3], mi7[4], k7, mi7[6], mi7[7]]
    ])

    numerator_b23 = np.linalg.det([
        [mi0[0], mi0[1], mi0[2], mi0[3], mi0[4], mi0[5], k0, mi0[7]],
        [mi1[0], mi1[1], mi1[2], mi1[3], mi1[4], mi1[5], k1, mi1[7]],
        [mi2[0], mi2[1], mi2[2], mi2[3], mi2[4], mi2[5], k2, mi2[7]],
        [mi3[0], mi3[1], mi3[2], mi3[3], mi3[4], mi3[5], k3, mi3[7]],
        [mi4[0], mi4[1], mi4[2], mi4[3], mi4[4], mi4[5], k4, mi4[7]],
        [mi5[0], mi5[1], mi5[2], mi5[3], mi5[4], mi5[5], k5, mi5[7]],
        [mi6[0], mi6[1], mi6[2], mi6[3], mi6[4], mi6[5], k6, mi6[7]],
        [mi7[0], mi7[1], mi7[2], mi7[3], mi7[4], mi7[5], k7, mi7[7]]
    ])

    numerator_b123 = np.linalg.det([
        [mi0[0], mi0[1], mi0[2], mi0[3], mi0[4], mi0[5], mi0[6], k0],
        [mi1[0], mi1[1], mi1[2], mi1[3], mi1[4], mi1[5], mi1[6], k1],
        [mi2[0], mi2[1], mi2[2], mi2[3], mi2[4], mi2[5], mi2[6], k2],
        [mi3[0], mi3[1], mi3[2], mi3[3], mi3[4], mi3[5], mi3[6], k3],
        [mi4[0], mi4[1], mi4[2], mi4[3], mi4[4], mi4[5], mi4[6], k4],
        [mi5[0], mi5[1], mi5[2], mi5[3], mi5[4], mi5[5], mi5[6], k5],
        [mi6[0], mi6[1], mi6[2], mi6[3], mi6[4], mi6[5], mi6[6], k6],
        [mi7[0], mi7[1], mi7[2], mi7[3], mi7[4], mi7[5], mi7[6], k7]
    ])

    b0 = numerator_b0 / denominator
    b1 = numerator_b1 / denominator
    b2 = numerator_b2 / denominator
    b3 = numerator_b3 / denominator
    b12 = numerator_b12 / denominator
    b13 = numerator_b13 / denominator
    b123 = numerator_b123 / denominator

    print("b0:", "%.2f" % b0, " b1:", "%.2f" % b1, " b2:", "%.2f" % b2, " b3:", "%.2f" % b3, " b12:", "%.2f" % b12,
          " b13:", "%.2f" % b13, " b123:", "%.2f" % b123)

    print(
        f"Рівняння регресії: y = {b0:.2f}{b1:+.2f}*x1{b2:+.2f}*x2{b3:+.2f}*x3{b12:+.2f}*x12{b13:+.2f}*x13{b123:+.2f}*x123")

    # find dispersion
    S2 = []
    for i in range(len(y_list)):
        S2.append(((y_list[i][0] - my_list[i]) ** 2 + (y_list[i][1] - my_list[i]) ** 2 + (
                    y_list[i][2] - my_list[i]) ** 2) / 3)

    Gp = max(S2) / sum(S2)

    m = len(y_list[0])
    f1 = m - 1
    f2 = N  # N=8
    q = 0.05

    Gt = [None, 0.68, 0.516, 0.438, 0.391, 0.3595, 0.3365, 0.3185, 0.3043, 0.2926, 0.2829, 0.2462, 0.2022, 0.1616,
          0.1250]

    print("Gt:", Gt[f1])

    if Gp < Gt[f1]:
        print("Дисперсія однорідна")
        break
    else:
        print("Дисперсія не однорідна")
        m += 1
        y_add_el()

x_matrix_normal = [
    [1, -1, -1, -1],
    [1, -1, -1, 1],
    [1, -1, 1, -1],
    [1, -1, 1, 1],
    [1, 1, -1, -1],
    [1, 1, -1, 1],
    [1, 1, 1, -1],
    [1, 1, 1, 1],
]

def get_beta(i):
    summa = 0
    for j in range(N):
        summa += my_list[j] * x_matrix_normal[j][i]
    summa /= N
    return summa


S2B = sum(S2) / N
S2beta = S2B / (N * m)
Sbeta = np.sqrt(S2beta)

beta0 = get_beta(0)
beta1 = get_beta(1)
beta2 = get_beta(2)
beta3 = get_beta(3)

t_criterion = []
t_criterion.append(abs(beta0) / Sbeta, )
t_criterion.append(abs(beta1) / Sbeta)
t_criterion.append(abs(beta2) / Sbeta)
t_criterion.append(abs(beta3) / Sbeta)

t0 = abs(beta0) / Sbeta
t1 = abs(beta1) / Sbeta
t2 = abs(beta2) / Sbeta
t3 = abs(beta3) / Sbeta

f3 = f1 * f2

t_tab = scipy.stats.t.ppf((1 + (1 - q)) / 2, f3)
print("t табличне:", t_tab)
if t0 < t_tab:
    b0 = 0
    print("t0:", t0, " t0<t_tab; b0=0")
if t1 < t_tab:
    b1 = 0
    print("t1:", t1, " t1<t_tab; b1=0")
if t2 < t_tab:
    b2 = 0
    print("t2:", t2, " t2<t_tab; b2=0")
if t3 < t_tab:
    b3 = 0
    print("t3:", t3, " t3<t_tab; b3=0")

y_hat = []
for i in range(N):
    y_hat.append(
        b0 + b1 * x_matrix[i][0] + b2 * x_matrix[i][1] + b3 * x_matrix[i][2] + b12 * x_matrix[i][0] * x_matrix[i][1] +
        b13 * x_matrix[i][0] * x_matrix[i][2] + b123 * x_matrix[i][0] * x_matrix[i][1] * x_matrix[i][2])

    print(f"y{i + 1}_hat = {b0:.2f}{b1:+.2f}*x{i + 1}1{b2:+.2f}*x{i + 1}2{b3:+.2f}*x{i + 1}3{b12:+.2f}*x{i + 1}1"
          f"*x{i + 1}2{b13:+.2f}*x{i + 1}1*x{i + 1}3{b123:+.2f}*x{i + 1}1*x{i + 1}2*x{i + 1}3 "
          f"= {y_hat[i]:.2f}")

d = 2
f4 = N - d
S2_ad = 0
for i in range(N):
    S2_ad += (m / (N - d) * ((y_hat[i] - my_list[i]) ** 2))

Fp = S2_ad / S2B
Ft = scipy.stats.f.ppf(1 - q, f4, f3)
print("Fp:", Fp)
print("Ft:", Ft)
if Fp > Ft:
    print("Рівняння регресії не адекватно оригіналу при рівні значимості 0,05")
else:
    print("Рівняння регресії адекватно оригіналу при рівні значимості 0,05")