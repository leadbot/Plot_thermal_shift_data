# -*- coding: utf-8 -*-
"""
Created on Thu Jun 19 14:13:07 2025

@author: dl923 / leadbot
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# Dummy data for demo
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)
y3 = np.exp(-x / 3) * np.sin(x)

# Create figure and gridspec layout
fig = plt.figure(figsize=(10, 8))
gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1])  # 2 rows, 2 cols

# Top-left plot
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(x, y1, label='Plot 1: sin(x)')
ax1.set_title('Top Left')
ax1.legend()

# Top-right plot
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(x, y2, label='Plot 2: cos(x)', color='orange')
ax2.set_title('Top Right')
ax2.legend()

# Bottom plot spans both columns
ax3 = fig.add_subplot(gs[1, :])
plot_thermofluor(normalized_data, grouped_data, plot_type='group error', show_sigmoid=True, ax=ax3, samples=['Cu', '10mM EDTA'])
ax3.set_title('Bottom (Spanning Both Columns)')
ax3.legend()

# Layout adjustment
plt.tight_layout()
plt.show()
