import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image
image = cv2.imread('a1.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 150, apertureSize=3)

# --- Step 1: Detect Vertical Lines ---
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=200, minLineLength=100, maxLineGap=10)

x_coords = []
for line in lines:
    x1, y1, x2, y2 = line[0]
    if abs(x1 - x2) < 10:  # near-vertical lines
        x_coords.append(x1)

# --- Step 2: Filter and Sort Line Positions ---
x_coords = sorted(set(x_coords))
filtered_x = []
prev = -100
for x in x_coords:
    if abs(x - prev) > 30:  # filter nearby duplicates
        filtered_x.append(x)
        prev = x

# --- Step 3: Compute Column Widths and Keep Wide Columns ---
min_width = 150  # adjust as needed
columns = []
for i in range(len(filtered_x) - 1):
    x_start = filtered_x[i]
    x_end = filtered_x[i + 1]
    width = x_end - x_start
    if width >= min_width:
        col_crop = image[:, x_start:x_end]
        columns.append((x_start, x_end, width, col_crop))

# --- Step 4: Display Wide Columns ---
for idx, (x_start, x_end, width, col_img) in enumerate(columns):
    print(f"Column {idx+1}: x_start={x_start}, x_end={x_end}, width={width}")
    plt.figure(figsize=(4, 10))
    plt.title(f"Column {idx+1} (Width={width})")
    plt.imshow(cv2.cvtColor(col_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()
