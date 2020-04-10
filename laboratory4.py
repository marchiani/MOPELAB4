import numpy as np
import scipy.stats
import random
from scipy.stats import f
from random import random
from functools import reduce
import math

np.set_printoptions(precision=3)


class Experiment1:
    print("Done by Dmytro Boychenko")
    def __init__(self):
        self.n = 8
        self.m = 3
        self.p = 0.95
        self.x_min = np.array([-10, -20, -20])
        self.x_max = np.array([50, 40, -15])

        self.x_mean_min = sum(self.x_min) / 3
        self.x_mean_max = sum(self.x_max) / 3

        self.y_min = 200 + self.x_mean_min
        self.y_max = 200 + self.x_mean_max

        self.x0 = (self.x_max + self.x_min) / 2
        self.dx = np.array(map(math.fabs, (self.x0 - self.x_min) / 2))

    def run(self):
        print("new experiment")
        print(f"n = {self.n}")
        print(f"m = {self.m}")
        print(f"p = {self.p}")
        self.create_norm_matrix()
        self.create_plan_matrix()
        self.run_experiment()
        self.b_coefficients = self.find_coefficients(self.x_norm_matrix)
        self.a_coefficients = self.find_coefficients(self.x_plan_matrix)
        self.base_regression_equation(self.b_coefficients, "b")
        self.base_regression_equation(self.a_coefficients, "a")
        self.regression_equation(self.b_coefficients, "b")
        self.regression_equation(self.a_coefficients, "a")
        self.find_yh_natur()
        self.find_yh_norm()
        if self.cohren_criterion() == -1:
            self.m += 1
            print(f"m += 1, m = {self.m}")
            self.run()
        self.student_criterion()
        self.fisher_criterion()

    def generate_y(self):
        return random() * (self.y_max - self.y_min) + self.y_min

    @staticmethod
    def average(arg):
        return reduce(lambda x, y: x + y, arg) / len(arg)

    @staticmethod
    def mult_sum(arr1, arr2):
        temp = 0
        for i in range(len(arr1)):
            temp += arr1[i] * arr2[i]
        return temp

    def create_norm_matrix(self):
        x_norm_matrix = []
        # x_norm_matrix.append(["x1", "x2", "x3", "x1x2", "x1x3", "x2x3", "x1x2x3"])
        x_norm_matrix.append(np.array([-1, -1, -1, +1, +1, +1, -1]))
        x_norm_matrix.append(np.array([-1, -1, +1, +1, -1, -1, +1]))
        x_norm_matrix.append(np.array([-1, +1, -1, -1, +1, -1, +1]))
        x_norm_matrix.append(np.array([-1, +1, +1, -1, -1, +1, -1]))
        x_norm_matrix.append(np.array([+1, -1, -1, -1, -1, +1, +1]))
        x_norm_matrix.append(np.array([+1, -1, +1, -1, +1, -1, -1]))
        x_norm_matrix.append(np.array([+1, +1, -1, +1, -1, -1, -1]))
        x_norm_matrix.append(np.array([+1, +1, +1, +1, +1, +1, +1]))

        self.x_norm_matrix = np.array(x_norm_matrix)

        print("\nx norm matrix")
        print(self.x_norm_matrix)

    def create_plan_matrix(self):
        x_plan_matrix = []
        for i in self.x_norm_matrix:
            temp = []
            temp.append(self.x_min[0] if i[0] == -1 else self.x_max[0])
            temp.append(self.x_min[1] if i[1] == -1 else self.x_max[1])
            temp.append(self.x_min[2] if i[2] == -1 else self.x_max[2])
            temp.append(temp[0] * temp[1])
            temp.append(temp[0] * temp[2])
            temp.append(temp[1] * temp[2])
            temp.append(temp[1] * temp[2] * temp[0])
            x_plan_matrix.append(np.array(temp))
        self.x_plan_matrix = np.array(x_plan_matrix)

        print("\nX plan matrix")
        print(self.x_plan_matrix)

    def run_experiment(self):
        y_plan_matrix = []
        for i in range(self.n):
            y_plan_matrix.append(np.array([self.generate_y() for i in range(self.m)]))
        self.y_plan_matrix = np.array(y_plan_matrix)

        print("\nY plan matrix")
        print(self.y_plan_matrix)

        self.y_mean = np.array([self.average(i) for i in self.y_plan_matrix])
        print()
        print("="*60)

        print("\ny mean values: ")
        print(self.y_mean)


    def find_coefficients(self, matrix):
        def find_mxy(x):
            x = [1 for _ in range(self.n)] if type(x) == int else x
            mxy = [self.mult_sum(x, [1 for _ in range(self.n)])]
            for i in range(7):
                mxy.append(self.mult_sum(x, matrix[:, i]))
            return np.array(mxy)

        m = [find_mxy(1)]
        for i in range(7):
            m.append(find_mxy(matrix[:, i]))
        m = np.array(m)
        k = find_mxy(self.y_mean)
        return np.linalg.solve(m, k)

    def regression_equation(self, coefs, name):
        print(f"\nRegression equation with interaction effect with {name} coefs:")
        print(f"{coefs[0]:.3f} + x1 * {coefs[1]:.3f} + x2 * {coefs[2]:.3f} + x3 * {coefs[3]:.3f}"
              f"+ x1x2 * {coefs[4]:.3f} + x1x3 * {coefs[5]:.3f} + x2x3 * {coefs[6]:.3f}"
              f"+ x1x2x3 * {coefs[7]:.4f}")

    def base_regression_equation(self, coefs, name):
        print(f"\nRegression equation with {name} coefs:")
        print(f"{coefs[0]:.3f} + x1 * {coefs[1]:.3f} + x2 * {coefs[2]:.3f} + x3 * {coefs[3]:.3f}")

    def find_yh_norm(self):
        yh = np.array([self.b_coefficients[0]
                       + self.b_coefficients[1] * self.x_norm_matrix[i, 0]
                       + self.b_coefficients[2] * self.x_norm_matrix[i, 1]
                       + self.b_coefficients[3] * self.x_norm_matrix[i, 2]
                       for i in range(self.n)])

        yh_ext = np.array([self.b_coefficients[0]
                           + self.b_coefficients[1] * self.x_norm_matrix[i, 0]
                           + self.b_coefficients[2] * self.x_norm_matrix[i, 1]
                           + self.b_coefficients[3] * self.x_norm_matrix[i, 2]
                           + self.b_coefficients[4] * self.x_norm_matrix[i, 3]
                           + self.b_coefficients[5] * self.x_norm_matrix[i, 4]
                           + self.b_coefficients[6] * self.x_norm_matrix[i, 5]
                           + self.b_coefficients[7] * self.x_norm_matrix[i, 6]
                           for i in range(self.n)])

        print("\nyh from equation of b coefficients")
        print(yh)
        print("\nyh from equation with interactions effect with b coefficients")
        print(yh_ext)

    def find_yh_natur(self):

        yh = np.array([self.a_coefficients[0]
                       + self.a_coefficients[1] * self.x_plan_matrix[i, 0]
                       + self.a_coefficients[2] * self.x_plan_matrix[i, 1]
                       + self.a_coefficients[3] * self.x_plan_matrix[i, 2]
                       for i in range(self.n)])

        yh_ext = np.array([self.a_coefficients[0]
                           + self.a_coefficients[1] * self.x_plan_matrix[i, 0]
                           + self.a_coefficients[2] * self.x_plan_matrix[i, 1]
                           + self.a_coefficients[3] * self.x_plan_matrix[i, 2]
                           + self.a_coefficients[4] * self.x_plan_matrix[i, 3]
                           + self.a_coefficients[5] * self.x_plan_matrix[i, 4]
                           + self.a_coefficients[6] * self.x_plan_matrix[i, 5]
                           + self.a_coefficients[7] * self.x_plan_matrix[i, 6]
                           for i in range(self.n)])
        print("\nyh from equation of a coefficients")
        print(yh)
        print("\nyh from equation with interactions effect with a coefficients")
        print(yh_ext)

    def cohren_criterion(self):
        print()
        print("="*60)
        print("Criterions")
        print("\n\nCohren criterion")
        self.variances = [sum((self.y_plan_matrix[i] - self.y_mean[i]) ** 2) for i in range(self.n)]
        Gp = max(self.variances) / sum(self.variances)

        self.f1 = self.m - 1
        self.f2 = self.n
        Gt = self.get_cohren_critical()
        print(f"Gt = {Gt:.3f}  Gp = {Gp:.3f}")
        if Gp < Gt:
            print("Cohren criterion is ok, variances are uniform")

        else:
            print("Cohren criterion is not ok, variances are not uniform")
            return -1

    def get_cohren_critical(self):
        f_crit = f.isf((1 - self.p) / self.f2, self.f1, (self.f2 - 1) * self.f1)
        return f_crit / (f_crit + self.f2 - 1)

    def student_criterion(self):
        print("\n\nStudent criterion")
        mean_var = sum(self.variances) / self.n
        self.s2_b = mean_var / (self.m * self.n)
        s_b = self.s2_b ** 0.5
        bs = np.array([sum([self.y_mean[j] * self.x_norm_matrix[j][i]
                            for i in range(7)]) / (self.n * self.n)
                       for j in range(self.n)])

        t = np.array(list(map(math.fabs, bs / s_b)))
        print(t)

        self.f3 = self.f1 * self.f2
        t_tabl = round(scipy.stats.t.ppf((1 + self.p) / 2, self.f3), 3)
        t = np.flip(t)
        print("\nStudent criterion")
        print(f"Values t for factors: {t} \nFt: {t_tabl}")
        valuable = t > t_tabl
        self.d = sum(valuable)
        for i in range(8):
            print(f"b{i} is valuable: {valuable[i]}")
        self.b_coefficients *= valuable
        print("\nnew coefficients")
        print(self.b_coefficients)
        self.yh = np.array([self.b_coefficients[0]
                            + self.b_coefficients[1] * self.x_norm_matrix[i][0]
                            + self.b_coefficients[2] * self.x_norm_matrix[i][1]
                            + self.b_coefficients[3] * self.x_norm_matrix[i][2]
                            for i in range(self.x_norm_matrix.shape[0])])

        print("\nValues for y with significant factors:")
        for i in range(self.n):
            print(f"y{i + 1} = {self.yh[i]}")

    def fisher_criterion(self):
        print("\n\nFisher criterion")
        f4 = self.n - self.d if self.n != self.d else 1

        s2_ad = sum([(self.yh[i] - self.y_mean[i]) ** 2 for i in range(self.n)]) * self.m / (self.n - self.d)
        Fp = s2_ad / self.s2_b
        Ft = scipy.stats.f.ppf(self.p, f4, self.f3)

        print(f"Fp = {Fp} Ft = {Ft}")

        if Fp > Ft:
            print(f"Regression equation is inadequate to original with p = {self.p:.2f}")
        else:
            print(f"Regression equation is adequate to original with p = {self.p:.2f}")


exp = Experiment1()
exp.run()
