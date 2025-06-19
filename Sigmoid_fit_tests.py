# -*- coding: utf-8 -*-
"""
Created on Thu Jun 19 12:59:36 2025

@author: dl923 / leadbot
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

###tests to find sigmoid params that fit

# Data
xraw = np.array([24.8, 25.6, 27.0, 27.8, 28.8, 29.8, 30.8, 31.8, 32.7, 33.8, 34.7, 35.8, 36.7, 37.8, 38.8, 39.7, 40.7, 41.7, 42.7, 43.7, 44.7, 45.7, 46.7, 47.7, 48.7, 49.7, 50.8, 51.7, 52.7, 53.7, 54.7, 55.7, 56.7, 57.7, 58.7, 59.7, 60.8, 61.8, 62.8, 63.8, 64.9, 65.7, 66.9, 67.7, 68.9, 69.9, 71.0, 71.9, 72.9, 74.0, 75.0, 75.9, 76.9, 77.9, 79.0, 80.0, 80.9, 82.0, 83.0, 84.0, 85.0, 85.9, 86.7, 87.9, 88.9, 89.7, 90.9, 91.7, 92.9, 93.7, 94.7])
yraw = np.array([1.28544423e-02, 1.13421550e-02, 1.05860113e-02, 7.93950851e-03, 5.67107750e-03, 1.51228733e-03, 3.40264650e-03, 1.89035917e-03, 1.13421550e-03, 0.00000000e+00, 3.78071834e-04, 7.56143667e-04, 2.64650284e-03, 1.51228733e-03, 1.51228733e-03, 7.56143667e-04, 3.78071834e-04, 2.26843100e-03, 4.53686200e-03, 5.67107750e-03, 7.56143667e-03, 9.07372401e-03, 1.58790170e-02, 2.45746692e-02, 3.47826087e-02, 5.33081285e-02, 8.31758034e-02, 1.36862004e-01, 2.20793951e-01, 3.46313800e-01, 5.18714556e-01, 7.15311909e-01, 8.75614367e-01, 9.66729679e-01, 1.00000000e+00, 9.93572779e-01, 9.69754253e-01, 9.34593573e-01, 8.85444234e-01, 8.44990548e-01, 7.93194707e-01, 7.55387524e-01, 7.01323251e-01, 6.68052930e-01, 6.24196597e-01, 5.85255198e-01, 5.42533081e-01, 5.04725898e-01, 4.62003781e-01, 4.29111531e-01, 3.96219282e-01, 3.59924386e-01, 3.32703214e-01, 3.06238185e-01, 2.76370510e-01, 2.57466919e-01, 2.38185255e-01, 2.20415879e-01, 2.00000000e-01, 1.78071834e-01, 1.57655955e-01, 1.42911153e-01, 1.20982987e-01, 1.05482042e-01, 9.18714556e-02, 7.93950851e-02, 6.35160681e-02, 6.12476371e-02, 4.68809074e-02, 3.96975425e-02, 3.40264650e-02])

# Automatically find rising segment up to the peak
peak_idx = np.argmax(yraw)
x_rise = xraw[:peak_idx+1]
y_rise = yraw[:peak_idx+1]

# Sigmoid for rising part
def sigmoid(x, bottom, top, Tm, slope):
    return bottom + (top - bottom) / (1 + np.exp((Tm - x) / slope))

# Initial guess
p0 = [min(y_rise), max(y_rise), x_rise[len(x_rise)//2], 1]

# Fit only the rising portion
popt, _ = curve_fit(sigmoid, x_rise, y_rise, p0)
bottom, top, Tm, slope = popt

# Plot
x_fit = np.linspace(min(xraw), max(xraw), 500)
y_fit = sigmoid(x_fit, *popt)

plt.figure(figsize=(6, 4))
plt.plot(xraw, yraw, 'o', label='Full Data', alpha=0.4)
plt.plot(x_rise, y_rise, 'o', label='Rising Segment', color='darkgreen')
plt.plot(x_fit, y_fit, '-', label=f'Sigmoid Fit\nTm = {Tm:.2f} °C', color='red')
plt.axvline(Tm, linestyle='--', color='gray', label='Tm')
plt.xlabel('Temperature (°C)')
plt.ylabel('Fluorescence')
plt.title('Thermal Shift – Rising Phase Fit')
plt.legend()
plt.tight_layout()
plt.show()