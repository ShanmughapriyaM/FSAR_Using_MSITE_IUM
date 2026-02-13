import matplotlib.pyplot as plt
import numpy as np

# 1. Prepare Data
shots = [1, 3, 5]
data = {
    'STRM [38]': {'UCF101': [80.5, 92.7, 96.9], 'HMDB51': [52.3, 67.4, 77.3]},
    'CPM [39]':  {'UCF101': [71.4, np.nan, np.nan], 'HMDB51': [60.1, np.nan, np.nan]},
    'HCL [40]':  {'UCF101': [82.6, 91.0, 94.5], 'HMDB51': [59.1, 71.2, 76.3]},
    'Molo [41]': {'UCF101': [86.0, 93.5, 95.5], 'HMDB51': [60.8, 72.0, 77.4]},
    'HM2 [42]':  {'UCF101': [86.6, 94.4, 96.2], 'HMDB51': [61.8, 72.6, 77.9]},
    'MISo [43]': {'UCF101': [68.19, np.nan, 87.11], 'HMDB51': [np.nan, 46.69, 50.31]},
    'Proposed Learnable MSITE':  {'UCF101': [78.16, 94.26, 95.90], 'HMDB51': [67.45, 71.86, 77.78]},
    'Proposed Segmented MSITE':  {'UCF101': [69.70, 81.35, 89.56], 'HMDB51': [65.21, 68.85, 73.51]}
}

# 2. Define Styles (Highlight Proposed, Mute Others)
styles = {
    'STRM [38]': {'color': '#999999', 'ls': '--', 'marker': 'o', 'lw': 1},
    'CPM [39]':  {'color': '#999999', 'ls': 'None', 'marker': 's', 'lw': 1},
    'HCL [40]':  {'color': '#999999', 'ls': '--', 'marker': '^', 'lw': 1},
    'Molo [41]': {'color': '#999999', 'ls': '--', 'marker': 'v', 'lw': 1},
    'HM2 [42]':  {'color': '#999999', 'ls': '--', 'marker': 'D', 'lw': 1},
    'MISo [43]': {'color': '#999999', 'ls': '--', 'marker': 'x', 'lw': 1},
    'Proposed Learnable MSITE': {'color': '#d62728', 'ls': '-', 'marker': '*', 'lw': 2.5, 'ms': 10},
    'Proposed Segmented MSITE': {'color': '#1f77b4', 'ls': '-', 'marker': 'P', 'lw': 2.5, 'ms': 8}
}

# 3. Create Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5))

def plot_dataset(ax, dataset_name, title):
    for method, results in data.items():
        style = styles[method]
        ax.plot(shots, results[dataset_name], label=method, 
                color=style['color'], linestyle=style.get('ls', '-'), 
                marker=style['marker'], linewidth=style['lw'], 
                markersize=style.get('ms', 5))
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Number of Shots', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_xticks(shots)
    ax.grid(True, linestyle=':', alpha=0.6)

plot_dataset(ax1, 'UCF101', 'UCF101 Performance')
plot_dataset(ax2, 'HMDB51', 'HMDB51 Performance')

# Add Legend at the bottom
handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.1), fontsize=10)

plt.tight_layout()
plt.savefig('msite_performance.svg', dpi=300, bbox_inches='tight')
plt.show()