import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse, PathPatch
from matplotlib.path import Path
import matplotlib.transforms as transforms

# Create figure
fig, ax = plt.subplots(figsize=(10, 8))

# Create brain outline
brain = Ellipse((0.5, 0.5), 0.8, 0.6, facecolor='lightgray', alpha=0.3)
ax.add_patch(brain)

# Add hemisphere division
ax.plot([0.3, 0.7], [0.5, 0.5], 'k-', alpha=0.5)

# Add some brain structure details
def draw_curve(start, end, control1, control2):
    verts = [start, control1, control2, end]
    codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]
    path = Path(verts, codes)
    patch = PathPatch(path, facecolor='none', edgecolor='gray', alpha=0.3)
    ax.add_patch(patch)

# Add some curves to represent brain structure
for i in np.linspace(0.3, 0.7, 5):
    draw_curve((i, 0.3), (i, 0.7), (i-0.1, 0.5), (i+0.1, 0.5))

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')

# Save the template
plt.savefig('visualizations/brain_template.png', dpi=300, bbox_inches='tight', transparent=True)
plt.close()