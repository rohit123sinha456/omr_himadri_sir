import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image
image = cv2.imread('a1.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 150, apertureSize=3)

# --- Step 1: Detect Straight Lines ---
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=200, minLineLength=100, maxLineGap=10)

# Draw lines on a copy
line_img = image.copy()
x_coords = []

for line in lines:
    x1, y1, x2, y2 = line[0]
    if abs(x1 - x2) < 10:  # vertical lines
        cv2.line(line_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        x_coords.append(x1)

# --- Step 2: Segment into 4 columns ---
# Get unique x coordinates and sort
x_coords = sorted(set(x_coords))
# Optionally filter nearby duplicates
filtered_x = []
prev = -100
for x in x_coords:
    if abs(x - prev) > 50:
        filtered_x.append(x)
        prev = x

# Determine column bounds
columns = []
for i in range(len(filtered_x)-1):
    x_start = filtered_x[i]
    x_end = filtered_x[i+1]
    col_crop = image[:, x_start:x_end]
    columns.append((x_start, x_end, col_crop))

# --- Step 3: Detect Circles in Each Column ---
for idx, (x_start, x_end, col_img) in enumerate(columns):
    gray_col = cv2.cvtColor(col_img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray_col, 5)

    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20,
                               param1=50, param2=30, minRadius=10, maxRadius=20)

    output = col_img.copy()
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            # draw the outer circle
            cv2.circle(output, (i[0], i[1]), i[2], (0, 255, 0), 2)
            # draw the center
            cv2.circle(output, (i[0], i[1]), 2, (0, 0, 255), 3)

    # Show result
    plt.figure(figsize=(5, 10))
    plt.title(f"Detected Circles - Column {idx+1}")
    plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()
