import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Classes and corresponding colors
CLASSES = ('background', 'building', 'road', 'water', 'barren', 'forest', 'agricultural')
PALETTE = [[0, 0, 0], [255, 0, 0], [255, 255, 0], [0, 0, 255],
           [159, 129, 183], [0, 255, 0], [255, 195, 128]]

DOMAINS = ['rural', 'urban']
DOMAIN_PALETTE = [[0, 150, 0], [255, 200, 150]]
CLASSES = list(CLASSES) + DOMAINS
PALETTE = PALETTE + DOMAIN_PALETTE
print("Classes:", CLASSES)
print("Palette:", PALETTE)
# Figure and axis
fig, ax = plt.subplots(figsize=(len(CLASSES) * 1.2, 0.4))  # horizontal layout

# Turn off axis
ax.axis('off')

# Draw each class label with background color
for i, (label, color) in enumerate(zip(CLASSES, PALETTE)):
    x = i
    color_norm = [c / 255 for c in color]
    rect = patches.Rectangle((x, 0), 1, 1, facecolor=color_norm)
    ax.add_patch(rect)
    ax.text(x + 0.5, 0.5, label, color='white' if sum(color) < 382 else 'black',
            fontsize=10, ha='center', va='center', weight='bold')

# Set limits
ax.set_xlim(0, len(CLASSES))
ax.set_ylim(0, 1)

# Save to file
plt.savefig('class_palette_strip.png', dpi=300, bbox_inches='tight', pad_inches=0.05)
plt.close()

print("✅ Saved to class_palette_strip.png")
