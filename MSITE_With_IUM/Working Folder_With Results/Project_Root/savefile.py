import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# List your file names here (ensure they are in the same folder)
image_files = [
r"C:\Users\Shanm\Documents\Work 4\MSITE\Project_Root\Results\FallBackwardsS9\StreamingPlots\prediction_timeline.png",
r"C:\Users\Shanm\Documents\Work 4\MSITE\Project_Root\Results\FallBackwardsS9\StreamingPlots\similarity_evolution.png",
r"C:\Users\Shanm\Documents\Work 4\MSITE\Project_Root\Results\PickObject\StreamingPlots\prediction_timeline.png",
r"C:\Users\Shanm\Documents\Work 4\MSITE\Project_Root\Results\PickObject\StreamingPlots\similarity_evolution.png",
r"C:\Users\Shanm\Documents\Work 4\MSITE\Project_Root\Results\Walks9\StreamingPlots\prediction_timeline.png",
r"C:\Users\Shanm\Documents\Work 4\MSITE\Project_Root\Results\Walks9\StreamingPlots\similarity_evolution.png"
]

# Set up the figure (3 rows, 2 columns)
# Width 7 inches is standard for a 2-column journal page
fig, axes = plt.subplots(3, 2, figsize=(7, 9), constrained_layout=True)

row_labels = ['(a)', '(b)', '(c)']

for i in range(3): # For each row
    for j in range(2): # For each column
        idx = i * 2 + j
        ax = axes[i, j]
        
        # Load and display image
        img = mpimg.imread(image_files[idx])
        ax.imshow(img)
        ax.axis('off') # Remove box/ticks
        
        # Add the row label only to the first column of each row
        if j == 0:
            ax.text(-0.1, 0.5, row_labels[i], transform=ax.transAxes, 
                    fontsize=14, fontweight='bold', va='center', ha='right')

# Save in high resolution for publication
# TIFF or PDF are preferred by most journals
#plt.savefig('composite_figure_3x2.tiff', dpi=300, bbox_inches='tight', compression='tiff_lzw')
plt.savefig('composite_figure_3x2.tiff', dpi=300, bbox_inches='tight', pil_kwargs={'compression': 'tiff_lzw'})
print("Composite figure created successfully.")


