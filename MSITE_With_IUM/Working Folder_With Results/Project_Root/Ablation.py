import matplotlib.pyplot as plt
import numpy as np

# Data from the table
datasets = ['UCF101', 'HMDB51', 'CAUCAFall']
baseline = [87.34, 71.23, 82.45]
multi_scale = [91.56, 74.12, 85.67]
incremental = [93.78, 76.45, 87.34]
ours = [95.90, 77.78, 88.23]

x = np.arange(len(datasets))
width = 0.2  # width of the bars

fig, ax = plt.subplots(figsize=(10, 6))

# Plotting bars with a professional color palette
rects1 = ax.bar(x - 1.5*width, baseline, width, label='Baseline', color='#cccccc')
rects2 = ax.bar(x - 0.5*width, multi_scale, width, label='+ Multi-Scale', color='#9ecae1')
rects3 = ax.bar(x + 0.5*width, incremental, width, label='+ Incremental Encoding', color='#4292c6')
rects4 = ax.bar(x + 1.5*width, ours, width, label='Ours (Full Model)', color='#084594')

# Formatting
ax.set_ylabel('Accuracy (%)', fontsize=12)
ax.set_title('Ablation Study: Incremental Feature Impact', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(datasets, fontsize=11)
ax.set_ylim(65, 100)  # Adjust to focus on the top performance
ax.legend(loc='upper left')
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Add value labels on top of the 'Ours' bars to highlight top performance
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold')

autolabel(rects4)

plt.tight_layout()
plt.savefig('msite_ablation.svg', dpi=300, bbox_inches='tight')
