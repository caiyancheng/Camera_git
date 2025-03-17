import matplotlib.pyplot as plt
import numpy as np
import json

class Color2Luminance_LG_G1:
    def __int__(self, degree_C2L, degree_L2C):
        self.degree_C2L = degree_C2L
        self.degree_L2C = degree_L2C
        with open(f'E:\Py_codes\Camera_git\Color_correction/Color_Luminance_Fit_result_poly_{degree_C2L}.json', 'r') as fp:
            C2L_fit_result = json.load(fp)
        with open(f'E:\Py_codes\VRR_Real\G1_Calibration/KONICA_Luminance_Color_Fit_result_poly_{degree_L2C}.json', 'r') as fp:
            L2C_fit_result = json.load(fp)

        self.C2L_coefficients = C2L_fit_result['size_nofull_all']['coefficients']
        self.C2L_coefficients_full = C2L_fit_result['size_full']['coefficients']
        self.color_value_min = C2L_fit_result['Color_min']
        self.color_value_min_Luminance = 10 ** np.polyval(self.C2L_coefficients, self.color_value_min)
        self.color_value_min_Luminance_full = 10 ** np.polyval(self.C2L_coefficients_full, self.color_value_min)
        self.color_value_max = C2L_fit_result['Color_max']
        self.color_value_max_Luminance = 10 ** np.polyval(self.C2L_coefficients, self.color_value_max)
        self.color_value_max_Luminance_full = 10 ** np.polyval(self.C2L_coefficients_full, self.color_value_max)

    def C2L(self, color_value, full_screen=False):
        if full_screen:
            if color_value < self.color_value_min:
                return self.color_value_min_Luminance_full
            elif color_value > self.color_value_max:
                return self.color_value_max_Luminance_full
            Luminance_value = 10 ** np.polyval(self.C2L_coefficients_full, color_value)
        else:
            if color_value < self.color_value_min:
                return self.color_value_min_Luminance
            elif color_value > self.color_value_max:
                return self.color_value_max_Luminance
            Luminance_value = 10 ** np.polyval(self.C2L_coefficients, color_value)
        return Luminance_value

    def L2C(self, Luminance_value, full_screen=False):
        if full_screen:
            if Luminance_value < self.Luminance_value_min:
                return self.Luminance_value_min_color_full
            elif Luminance_value > self.Luminance_value_max:
                return self.Luminance_value_max_color_full
            color_value = np.polyval(self.L2C_coefficients_full, np.log10(Luminance_value))
        else:
            if Luminance_value < self.Luminance_value_min:
                return self.Luminance_value_min_color
            elif Luminance_value > self.Luminance_value_max:
                return self.Luminance_value_max_color
            color_value = np.polyval(self.L2C_coefficients, np.log10(Luminance_value))
        return color_value


if __name__ == "__main__":
    CL_transform = Color2Luminance_LG_G1()
    CL_transform.__int__(degree_C2L=7, degree_L2C=7)
    # C2L_begin_time = time.time()
    # n_cycle = 10000
    # for i in range(n_cycle):
    #     L = CL_transform.C2L(color_value=0.5)
    # C2L_end_time = time.time()
    # print('Average C2L time:', (C2L_end_time - C2L_begin_time) / n_cycle)
    # L2C_begin_time = time.time()
    # for i in range(n_cycle):
    #     C = CL_transform.L2C(Luminance_value=100)
    # L2C_end_time = time.time()
    # print('Average C2L time:', (L2C_end_time - L2C_begin_time) / n_cycle)
    color_value = np.linspace(0,1, 100)
    L = np.zeros(len(color_value))
    L_full = np.zeros(len(color_value))
    for cv in range(len(color_value)):
        L[cv] = CL_transform.C2L(color_value=color_value[cv])
        L_full[cv] = CL_transform.C2L(color_value=color_value[cv], full_screen=True)
    plt.figure()
    plt.plot(color_value, L)
    # plt.plot(color_value, L_full)
    plt.show()

