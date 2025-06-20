# -*- coding: utf-8 -*-
"""
Created on Thu Jun 19 14:13:07 2025

@author: dl923 / leadbot
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from pandas import read_excel
from matplotlib.pyplot import subplots
from scipy.optimize import minimize
import lmfit

#%% BLOCK1 read and basic plot
####block 1
#determination of optimal timepoints taken from times and means variables
#takes range of timepoints give optimal initial rate estimation from chi^2 values from MM kinetics
def mm(x, Vm, Km):
    return Vm * x/ (Km + x)

def residuals(p, xAxis, experimental_data, weights):
    Vm = p["Vm"].value
    Km = p["Km"].value
    return (experimental_data - mm(xAxis, Vm, Km)) * weights

def fit_mm_model(xAxis, vm_data, std_data):
    params = lmfit.Parameters()
    params.add("Vm", value=1, min=0, max=20)
    params.add("Km", value=1, min=0, max=20)
    
    weights = 1 / np.array(std_data)
    
    # Use kws to pass weights as a keyword argument
    result = lmfit.minimize(
        residuals,
        params,
        args=(xAxis, vm_data),
        kws={"weights": weights},
        method="least_squares"
    )
    return result


#%% BLOCK 2
wd = "R:/rsrch/ncb/lab/Dan/Papers & Manuscripts erc/LPMO_discovery_paper/Kinetic_data/" 
file = "AA19_kinetics_h2o2_proc.xlsx"  
filename=wd+file
def process_df(file, wd):
    df = read_excel(filename, sheet_name="Sheet3", header=1)
    times = np.array(df.iloc[:, 0], dtype=float)
    df = df.iloc[:, 1:]
    df = df.T
    return df, times

df, times = process_df(file, wd)
#print(df)
def split_df_reps(df, num_reps):
    total_reps = df.shape[0] // num_reps + (df.shape[0] % num_reps > 0)
    individual_reps = np.array_split(df, total_reps, axis=0)
    return individual_reps
reps = split_df_reps(df, 5)
#print(reps)

def plot_data(averages, times, s_data):
    fig, axes = subplots(figsize=(5,5))
    for i, average_row in enumerate(averages):
        means = average_row.mean(axis=0)
        axes.scatter(times, means, label=s_data[i])
        axes.errorbar(times, means, yerr=average_row.std(axis=0))
        axes.legend()
        plt.show()
    return fig

plot_data(reps, times, [0,1,2.5,5,10,25,50,75,100,125,150,200,250,500])



#%% BLOCK 3
####block 3
# List to store experimental data for fitting
h2o2 = np.array([0,1,2.5,5,10,25,50,75,100,125,150,200,250,500], dtype=float)
km_data_h2o2 = []
vm_data_h2o2 = []
std_data_h2o2 = []
reps = np.array(reps)
print(len(reps[:10]))
for i, rep in enumerate(reps[:9]):
    means = rep.mean(axis=0)
    v0_rate = []
    for row in rep:
        row = row / 53200 # determining molarity of hydrocoerulignone
        row = row*(200*10**-6) #determination of moles
        row = row*60 #moles per min
        row = row * 10**6 #umol/min
        slope, intercept = np.polyfit(times, row, 1)  # Initial rate calculation
        v0_rate.append(slope)
    v0 = np.mean(v0_rate)
    v0_ste = np.std(v0_rate)/ np.sqrt(len(v0_rate))
    km_data_h2o2.append([v0, h2o2, v0_ste])
    vm_data_h2o2.append(v0)
    std_data_h2o2.append(v0_ste)

def mm (x, Vm, Km):
    return Vm*x/(Km + x)

xAxis_h2o2 = h2o2[:9]  # Use the first 10 time points => above 100 uM H2O2 we see reduction in activity might be associated with hole hopping
xAxis_h2o2 = np.array(xAxis_h2o2, dtype=float)  # Convert to NumPy array
vm_data_h2o2 = np.array(vm_data_h2o2, dtype=float)

def residuals (p, xAxis, experimental_data):
    Vm = p["Vm"].value
    Km = p["Km"].value
    fi =  mm(xAxis_h2o2, Vm, Km)
    return experimental_data - fi

params = lmfit.Parameters()
params.add("Vm", value=1,min=0, max=20)
params.add("Km", value=1,min=0, max=20)
print(vm_data_h2o2, xAxis_h2o2)

result = lmfit.minimize(residuals, params, args=(xAxis_h2o2, vm_data_h2o2))
print(f"xAxis length: {len(xAxis)}")
print(f"vm_data length: {len(vm_data)}")
print(f"std_data_h2o2 length: {len(std_data_h2o2)}")

#%% BLOCK 4

from matplotlib import pyplot as plt
x_fit_h2o2 = np.linspace(min(xAxis), max(xAxis), 500)  # 500 points for smoothness

# Use fitted parameters to evaluate model
Vm_fit_h2o2 = result.params["Vm"].value
Km_fit_h2o2 = result.params["Km"].value
y_fit_h2o2 = mm(x_fit_h2o2, Vm_fit_h2o2, Km_fit_h2o2)

# Plot
plt.errorbar(xAxis_h2o2, vm_data_h2o2, yerr=std_data_h2o2, fmt='o', label='Experimental Data')
plt.plot(x_fit_h2o2, y_fit_h2o2, '-', label='Michaelis-Menten Fit', color='tab:red')
plt.xlabel('Hydrogen peroxide concentration (μM)')
plt.ylabel('Rate of coerulignone formation (μmol/min)')
plt.legend()
plt.grid(True)
plt.show()


#%% BLOCK 5
####block 5
#processing data for DMP kinetics
file = "AA19_DMP_kinetics_processed.xlsx"
df = read_excel(wd+file,sheet_name="Sheet3", header=1)
times = df.iloc[:,0]
df = df.iloc[:,1:]
df = df.T
reps = split_df_reps(df, 5)
averages = []
s_data = [0,1,2.5,5,10,25,50,100]
fig, axes = subplots(figsize=(5,5))
for i, rep in enumerate(reps):
    #print(chunks[i])
    average_row = rep.to_numpy()
    means = average_row.mean(axis=0)
    print(times[:4],means[:4])
    #means = means - blank_mean
    #print(means.shape, times.shape)
    #print(average_row)
    averages.append(average_row)
    axes.scatter(times, means, label=s_data[i])
    axes.errorbar(times[:3], means[:3], yerr=average_row[:,:3].std(axis=0))
    axes.legend()
#%%
####block 6
# List to store experimental data for fitting
km_data_DMP = []
vm_data_DMP = []
std_data_DMP = []
reps = np.array(reps)
print(len(reps[:10]))
for i, rep in enumerate(reps):
    means = rep.mean(axis=0)
    v0_rate = []
    for row in rep:
        row = row / 53200 # determining molarity of hydrocoerulignone
        row = row*(200*10**-6) #determination of moles
        row = row*60 #moles per min
        row = row * 10**6 #umol/min
        slope, intercept = np.polyfit(times[:3], row[:3], 1)  # Initial rate calculation
        v0_rate.append(slope)
    v0 = np.mean(v0_rate)
    v0_ste = np.std(v0_rate)/ np.sqrt(len(v0_rate))
    km_data_DMP.append([v0, h2o2, v0_ste])
    vm_data_DMP.append(v0)
    std_data_DMP.append(v0_ste)

def mm (x, Vm, Km):
    return Vm*x/(Km + x)

xAxis_DMP = s_data  # Use the first 10 time points => above 100 uM H2O2 we see reduction in activity might be associated with hole hopping
xAxis_DMP = np.array(xAxis_DMP, dtype=float)  # Convert to NumPy array
vm_data_DMP = np.array(vm_data_DMP, dtype=float)

def residuals (p, xAxis, experimental_data):
    Vm = p["Vm"].value
    Km = p["Km"].value
    fi =  mm(xAxis, Vm, Km)
    return experimental_data - fi

params = lmfit.Parameters()
params.add("Vm", value=1,min=0, max=20)
params.add("Km", value=1,min=0, max=20)
print(vm_data_DMP, xAxis_DMP)

result = lmfit.minimize(residuals, params, args=(xAxis_DMP, vm_data_DMP))
print(f"xAxis length: {len(xAxis)}")
print(f"vm_data length: {len(vm_data)}")
print(f"std_data length: {len(std_data)}")
#%%
####block 7
from matplotlib import pyplot as plt
x_fit_DMP = np.linspace(min(xAxis_DMP), max(xAxis_DMP), 500)  # 500 points for smoothness

# Use fitted parameters to evaluate model
Vm_fit_DMP = result.params["Vm"].value
Km_fit_DMP = result.params["Km"].value
y_fit_DMP = mm(x_fit_DMP, Vm_fit_DMP, Km_fit_DMP)

# Plot
plt.errorbar(xAxis_DMP, vm_data_DMP, yerr=std_data_DMP, fmt='o', label='Experimental Data')
plt.plot(x_fit_DMP, y_fit_DMP, '-', label='Michaelis-Menten Fit', color='tab:red')
plt.xlabel('DMP concentration (mM)')
plt.ylabel('Rate of coerulignone formation (μmol/min)')
plt.legend()
plt.grid(True)
plt.show()
#%% block 8
####block 8
# #determination of optimal timepoints taken from times and means variables
#takes range of timepoints give optimal initial rate estimation from chi^2 values from MM kinetics
def mm(x, Vm, Km):
    return Vm * x/ (Km + x)

def residuals(p, xAxis, experimental_data, weights):
    Vm = p["Vm"].value
    Km = p["Km"].value
    return (experimental_data - mm(xAxis, Vm, Km)) * weights

def fit_mm_model(xAxis_DMP, vm_data_DMP, std_data_DMP):
    params = lmfit.Parameters()
    params.add("Vm", value=1, min=0, max=20)
    params.add("Km", value=1, min=0, max=20)
    result = lmfit.minimize(residuals, params, args=(xAxis_DMP, vm_data_DMP, weights), method="least_squares")
    return result
#df, times = process_df("AA19_kinetics_h2o2_proc.xlsx", wd)
df, times = process_df("AA19_DMP_kinetics_processed.xlsx", wd)
reps = split_df_reps(df, 5)
timepoints = len(times)
fit_results = []
#xAxis = np.array([0,1,2.5,5,10,25,50,75,100,125,150,200,250,500], dtype=float)
xAxis_DMP = np.array(s_data, dtype=float)
averages = np.array(reps,dtype=float)[:len(xAxis_DMP)]
for n in range(2, timepoints + 1):
    vm_data_DMP = []
    std_data_DMP = []
    for i, rows in enumerate(averages):
        v0_rate = []
        for row in rows:
            slope, intercept = np.polyfit(times[:n], row[:n], 1)
            v0_rate.append(slope)
        v0 = np.mean(v0_rate)
        v0_std = np.std(v0_rate) / np.sqrt(len(v0_rate))
        vm_data_DMP.append(v0)
        std_data_DMP.append(v0_std)
    
    vm_data_DMP = np.array(vm_data_DMP, dtype=float)
    weights = 1/np.array(std_data)
    assert len(vm_data_DMP) == len(xAxis_DMP), "x and y data length mismatch"
    result = fit_mm_model(xAxis_DMP,vm_data_DMP,std_data_DMP)
    red_chi = result.redchi
    fit_results.append((n, red_chi, result.params["Vm"].value, result.params["Km"].value))

print(f"{'TimePts':>8} | {'RedChi^2':>10} | {'Vm':>6} | {'Km':>6}")
print("-"*40)
for res in fit_results:
    print(f"{res[0]:>8} | {res[1]:>10.4f} | {res[2]:>6.5f} | {res[3]:>6.2f}")

#%%
# Dummy data for demo
from matplotlib.ticker import ScalarFormatter
cs = 2
fs = 12

# Create figure and gridspec layout
fig = plt.figure(figsize=(10, 8))
gs = gridspec.GridSpec(2, 2, height_ratios=[1, 2])

# ------------------- Top-left plot (DMP)
ax1 = fig.add_subplot(gs[0, 0])
ax1.errorbar(xAxis_DMP, vm_data_DMP, yerr=std_data_DMP, fmt='o',
             markeredgecolor='black',
             markerfacecolor=sns.xkcd_palette(['clear blue'])[0], capsize=cs)
ax1.plot(x_fit_DMP, y_fit_DMP, '--', color=sns.xkcd_palette(['coral'])[0])
ax1.set_xlabel('DMP concentration (mM)', size=fs)
ax1.set_ylabel('Coerulignone (μmol.min$^{-1}$)', size=fs)

# Force scientific tick formatting without offset label
yfmt1 = ScalarFormatter(useMathText=True)
yfmt1.set_powerlimits((0, 0))
ax1.yaxis.set_major_formatter(yfmt1)
ax1.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
ax1.get_yaxis().get_offset_text().set_visible(False)

# Annotate manual multiplier
ax1_max = max(ax1.get_yticks())
exp1 = int(np.floor(np.log10(ax1_max)))
ax1.annotate(r'$\times$10$^{%i}$' % exp1,
             xy=(0.02, 0.95), xycoords='axes fraction',
             fontsize=fs, verticalalignment='top')

# ------------------- Top-right plot (H2O2)
ax2 = fig.add_subplot(gs[0, 1])
ax2.errorbar(xAxis_h2o2, vm_data_h2o2, yerr=std_data_h2o2, fmt='o',
             markeredgecolor='black',
             markerfacecolor=sns.xkcd_palette(['clear blue'])[0], capsize=cs)
ax2.plot(x_fit_h2o2, y_fit_h2o2, '--', color=sns.xkcd_palette(['coral'])[0])
ax2.set_xlabel('Hydrogen peroxide concentration (μM)', size=fs)
ax2.set_ylabel('Coerulignone (μmol.min$^{-1}$)', size=fs)

# Scientific formatting for y-axis with manual multiplier
yfmt2 = ScalarFormatter(useMathText=True)
yfmt2.set_powerlimits((0, 0))
ax2.yaxis.set_major_formatter(yfmt2)
ax2.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
ax2.get_yaxis().get_offset_text().set_visible(False)

ax2_max = max(ax2.get_yticks())
exp2 = int(np.floor(np.log10(ax2_max)))
ax2.annotate(r'$\times$10$^{%i}$' % exp2,
             xy=(0.02, 0.95), xycoords='axes fraction',
             fontsize=fs, verticalalignment='top')

# Bottom plot spans both columns
ax3 = fig.add_subplot(gs[1, :])
#ax3.plot(x, y2, label='Plot 2: cos(x)', color='orange')
plot_thermofluor(normalized_data, grouped_data, plot_type='group error', show_sigmoid=True, ax=ax3, samples=['Cu', '10mM EDTA'], title=False)
ax3.set_axisbelow(False)  # sometimes helps prevent grid redraws
for spine in ax3.spines.values():
    spine.set_visible(True)  # optional — keeps axes spines if hidden
for line in ax3.get_xgridlines() + ax3.get_ygridlines():
    line.set_visible(False)
# Layout adjustment
plt.tight_layout()
fig.savefig("Biochemical_characterisation.svg")
plt.show()
