# -*- coding: utf-8 -*-
"""
Created on Wed Jun 18 20:19:53 2025

@author: Dan / leadbot
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from scipy.optimize import curve_fit

def read_plate_map(plate_csv):
    """Reads a 96-well plate map CSV into a dict mapping wells to sample names, skipping NaNs and blanks"""
    df = pd.read_csv(plate_csv, index_col=0)
    df.columns = df.columns.astype(str)  
    plate_map = {}
    for row in df.index:
        for col in df.columns:
            well = f"{col}{row}"
            sample = df.loc[row, col]
            # Skip NaN and empty
            if sample is None:
                continue
            if isinstance(sample, float) and math.isnan(sample):
                continue
            if str(sample).strip() == '':
                continue
            plate_map[well] = str(sample).strip()
    return plate_map


def read_thermofluor_data(input_file, plate_map):
    """Reads thermofluor data into a dictionary of labeled wells only"""
    alphabet = ['A','B','C','D','E','F','G','H']
    well_dict = {i: f"{alphabet[(i-1)//12]}{(i-1)%12 + 1}" for i in range(1, 97)}
    data_dict = {}
    temp_flour_dict = {}
    cyclecount = 0
    with open(input_file) as f:
        while True:
            line = f.readline().strip()
            if not line: break
            if line.endswith('.mxp') or line.endswith('Temperature'): continue

            parts = line.split('\t')
            wellnumber = int(parts[3])
            well = well_dict[wellnumber]
            # Only process labeled wells
            if well not in plate_map:
                continue
            fluorescence = float(parts[6])
            temperature = float(parts[7])
            if well not in data_dict:
                data_dict[well] = {'temperature': [], 'fluorescence': [], 'stdev': []}
                cyclecount = 0
                temp_flour_dict = {}
            cyclecount += 1
            temp_flour_dict[cyclecount] = fluorescence
            if cyclecount == 3:
                avg = np.mean([temp_flour_dict[1], temp_flour_dict[2], temp_flour_dict[3]])
                stdev = np.std([temp_flour_dict[1], temp_flour_dict[2], temp_flour_dict[3]])
                data_dict[well]['fluorescence'].append(avg)
                data_dict[well]['temperature'].append(temperature)
                data_dict[well]['stdev'].append(stdev)
                cyclecount = 0

    return data_dict


def normalize_data(data_dict):
    """Normalise data"""
    for well, data in data_dict.items():
        f = np.array(data['fluorescence'])
        f_min, f_max = np.min(f), np.max(f)
        data['normalized'] = (f - f_min) / (f_max - f_min) if f_max > f_min else f * 0
    return data_dict

def group_and_average(data_dict, plate_map, raw=False):
    """Groupby occurances of the same label and average / stdev"""
    grouped = {}
    for well, sample in plate_map.items():
        # skip wells not in data_dict
        if well not in data_dict:
            continue
        # skip wells with sample name that is NaN or empty string
        if sample is None:
            continue
        if isinstance(sample, float) and math.isnan(sample):
            continue
        if str(sample).strip() == '':
            continue

        if sample not in grouped:
            grouped[sample] = {'curves': []}
            grouped[sample]['temperature']=data_dict[well]['temperature']
        key='normalized'
        if raw==True:
            key='fluorescence'
            print("Raw is set to true!", key)
        grouped[sample]['curves'].append(data_dict[well][key])
    for sample, group in grouped.items():
        stacked = np.vstack(group['curves'])
        grouped[sample]['mean'] = np.mean(stacked, axis=0)
        grouped[sample]['stdev'] = np.std(stacked, axis=0)
        grouped[sample]['normalized_mean'] = (grouped[sample]['mean'] - grouped[sample]['mean'].min()) / \
                                             (grouped[sample]['mean'].max() - grouped[sample]['mean'].min())
    print(data_dict[well]['temperature'])
    return grouped

def sigmoid_4pl(x, bottom, top, Tm, slope):
    """4-param logistic sigmoid for rising phase"""
    return bottom + (top - bottom) / (1 + np.exp((Tm - x) / slope))

def get_tm_midpoint(inlist):
    """Get index and midpoint value of half-max fluorescence"""
    midval = (max(inlist) + min(inlist)) / 2
    for x in range(len(inlist) - 1):
        if inlist[x] <= midval <= inlist[x+1]:
            return x, midval
    return None, midval  # fallback

def plot_with_sigmoid(data_dict, data_column):
    ##data column can be flouresnce or normalised
    for well in data_dict:
        xraw = np.array(data_dict[well]['temperature'])
        yraw = np.array(data_dict[well][data_column])
        # Find peak and crop to rising phase only
        maxindex = np.argmax(yraw)
        x_rise = xraw[:maxindex + 1]
        y_rise = yraw[:maxindex + 1]

        # Fit rising phase using 4PL sigmoid
        try:
            p0 = [min(y_rise), max(y_rise), x_rise[len(x_rise)//2], 1]
            popt, _ = curve_fit(sigmoid_4pl, x_rise, y_rise, p0)
        except Exception as e:
            print(f"Fit failed for well {well}: {e}")
            continue

        bottom, top, Tm, slope = popt

        # Create fitted curve over full xraw
        xsig = np.array(xraw)
        ysig = sigmoid_4pl(xsig, *popt)

        # Store extended curve and Tm
        data_dict[well]['SigX_'+data_column] = xsig.tolist()
        data_dict[well]['SigY_'+data_column] = ysig.tolist()
        data_dict[well]['Tm'] = float(Tm)

        # For compatibility with your previous structure:
        Tmindex, midval = get_tm_midpoint(ysig)
        data_dict[well]['Tmindex'] = Tmindex
    return data_dict

def plot_thermofluor(data_dict, grouped_data, plot_type='raw', selected_samples=None, show_sigmoid=False, ax=None, 
                     title=True, fs=12, save=False):
    handles=[]
    legend_labels=[]
    ylabel='Fluorescence intensity (Ex 470nm/Em 570nm)'
    if ax==None:
        fig, ax = plt.subplots(figsize=(12, 6))
    if plot_type in ['raw', 'normalized']:
        labels = [w for w in data_dict if selected_samples is None or w in selected_samples]
    else:
        labels = [g for g in grouped_data if selected_samples is None or g in selected_samples]
    def rotate(l, n):
         return l[n:] + l[:n]
    palette = sns.color_palette("hls", len(labels))
    grey=sns.xkcd_palette(["slate grey"])
    palette=rotate(palette, -1)[:-1]+grey
    color_map = dict(zip(labels, palette))
    if plot_type in ['group error', 'normalized_group', 'raw grouped']:
        print(labels)
        for sample in labels:
            group = grouped_data[sample]
            temp = data_dict[next(iter(data_dict))]['temperature']
            ydata = group['normalized_mean'] if plot_type == 'normalized_group' else group['mean']
            yerr = group['stdev'] if plot_type == 'group error' else None

            if yerr is not None:
                lines=ax.errorbar(temp, ydata, yerr=yerr, fmt='o', capsize=3, label=str(sample),
                             markeredgecolor='black', markerfacecolor=color_map[sample], alpha=0.9)
                # Set capline color (top and bottom caps)
                for i, cap in enumerate(lines[1]):
                    cap.set_color(color_map[sample])
                    cap.set_alpha(1.0)  # optional

    # Set vertical bar color
                for bar in lines[2]:
                    bar.set_color(color_map[sample])
                    bar.set_alpha(1.0)  # optional
            else:
                lines=ax.scatter(temp, ydata, label=str(sample), edgecolor='black',
                                 color=color_map[sample], s=20, alpha=0.9)
            
            handles.append(lines)
            if show_sigmoid:
                sigX=grouped_data[sample]['SigX_normalized_mean']
                sigY=grouped_data[sample]['SigY_normalized_mean']
                ax.plot(sigX, sigY, '--', color=color_map[sample], alpha=0.8)
                if not samples==None:
                    if sample in samples:
                         ax.axvline(grouped_data[sample]['Tm'], color=color_map[sample], linestyle=':', alpha=0.6, lw=3)
                         ax.text(grouped_data[sample]['Tm'] + 0.3, 0.1, f"{grouped_data[sample]['Tm']:.2f}°C", 
                                rotation=90, va='center', color=color_map[sample], size=12)
        ax.set_ylabel(ylabel if plot_type == 'group error' else 'Normalized Mean Fluorescence', size=fs)
        #ax.set_xlim([45, 60])
        legend_labels = [str(x) + '\u00B2\u207A' if len(str(x)) < 3 else str(x) for x in labels]
        ax.legend(handles, legend_labels, loc='lower right', fontsize=10)

    else:
        if plot_type == 'raw':
            for well in labels:
                temp = data_dict[well]['temperature']
                fluo = data_dict[well]['fluorescence']
                ax.scatter(temp, fluo, label=str(well), color=color_map[well], s=20, alpha=0.8)
                if show_sigmoid==True:
                    sigX=data_dict[well]['SigX_fluorescence']
                    sigY=data_dict[well]['SigY_fluorescence']
                    ax.plot(sigX, sigY, '--', color=color_map[well])
            ax.set_ylabel(ylabel)
            
        elif plot_type == 'normalized':
           for well in labels:
                temp = data_dict[well]['temperature']
                fluo = data_dict[well]['normalized']
                ax.scatter(temp, fluo, label=str(well), color=color_map[well], s=20, alpha=0.8)
                if show_sigmoid==True:
                    sigX=data_dict[well]['SigX_normalized']
                    sigY=data_dict[well]['SigY_normalized']
                    ax.plot(sigX, sigY, '--', color=color_map[well])
           ax.set_ylabel(ylabel)
        
    ax.set_xlabel('Temperature (°C)', size=fs)
    if title:
         ax.set_title(f"ThermoFluor Plot: {plot_type}", size=fs)
    ax.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    if save:
        fig.savefig(f"{plot_type}_theral_shift.svg", transparent=True)
    plt.show()


# File paths
plate_csv = 'plate_layout.csv'
#plate_csv = 'plate_layout_Cu_EDTA.csv'
data_file = 'AA19_metal_stability_310125_Admin, SYBR Green, 01-31-2025, 17Hr 10Min - Instrument Data - Text Format 1.txt'

# Process
plate_map = read_plate_map(plate_csv)
raw_data = read_thermofluor_data(data_file, plate_map)
normalized_data = normalize_data(raw_data)
raw_grouped = group_and_average(raw_data, plate_map, raw=True)
grouped_data = group_and_average(normalized_data, plate_map)

# Plot one of: 'raw', 'normalized', 'group', 'normalized_group'
#plot_thermofluor(raw_data, grouped_data, plot_type='raw')
#plot_thermofluor(normalized_data, grouped_data, plot_type='normalized')
#plot_thermofluor(normalized_data, grouped_data, plot_type='normalized_group')
#plot_thermofluor(normalized_data, grouped_data, plot_type='group')

tm_raw=plot_with_sigmoid(raw_data, 'fluorescence')
tm_raw_grouped=plot_with_sigmoid(raw_grouped, 'mean')
tm_norm=plot_with_sigmoid(normalized_data, 'normalized')
tm_norm_grouped=plot_with_sigmoid(grouped_data, 'normalized_mean')

#tm_n = 
# For each plot type:
plot_thermofluor(raw_data, grouped_data, plot_type='raw', show_sigmoid=True, save=True)
plot_thermofluor(raw_grouped, grouped_data, plot_type='raw grouped', show_sigmoid=True)
plot_thermofluor(normalized_data, grouped_data, plot_type='normalized', show_sigmoid=True)
plot_thermofluor(normalized_data, grouped_data, plot_type='normalized_group', show_sigmoid=True, save=True)
plot_thermofluor(normalized_data, grouped_data, plot_type='group error', show_sigmoid=True, save=True)