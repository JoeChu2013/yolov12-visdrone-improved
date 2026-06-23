import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import numpy as np

img = cv2.imread('dataset_comparison.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Global font settings
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['SimSun']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10

# Create figure matching A4 width precisely
# Original image is 528x372. Ratio is 1.419
# We set width=7.2, height=7.2/1.419 = 5.07
fig, ax = plt.subplots(figsize=(7.2, 5.07), dpi=300)
ax.imshow(img)
ax.axis('off')

def mask_and_text(x, y, w, h, text, font_size=10, center_align=True, weight='normal'):
    # draw white rectangle to mask English text
    rect = patches.Rectangle((x, y), w, h, linewidth=0, edgecolor='none', facecolor='white', zorder=2)
    ax.add_patch(rect)
    
    # add text
    text_x = x + w/2 if center_align else x + w/2
    text_y = y + h/2 + 1.5 # slight offset down for vertical centering in matplotlib
    ax.text(text_x, text_y, text, fontsize=font_size, ha='center', va='center', zorder=3, weight=weight)

# Mask and rewrite titles
mask_and_text(60, 135, 410, 23, '(a) MS COCO (左) 与 VisDrone (右) 实例尺度对比', font_size=11, weight='bold')
mask_and_text(110, 350, 310, 23, '(b) VisDrone 数据集目标特征', font_size=11, weight='bold')

# Mask and rewrite Left labels
# Original: "People\nMotor" -> "行人\n摩托车"
mask_and_text(5, 233, 65, 36, '行人\n摩托车', font_size=10)
# Original: "Car" -> "汽车"
mask_and_text(10, 350, 60, 22, '汽车', font_size=10)

# Mask and rewrite Right labels
# Original: "Pedestrian" -> "行人"
mask_and_text(450, 238, 75, 25, '行人', font_size=10)
# Original: "Motor" -> "三轮车" (corrected based on previous conversation)
mask_and_text(455, 350, 70, 22, '三轮车', font_size=10)

# Remove all margins
plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
plt.margins(0, 0)
ax.xaxis.set_major_locator(plt.NullLocator())
ax.yaxis.set_major_locator(plt.NullLocator())

plt.savefig('dataset_comparison_academic.png', bbox_inches='tight', pad_inches=0, dpi=300)
print("Saved dataset_comparison_academic.png")
