import cv2
import numpy as np
import random

# Load the image
image_path = "a2.jpg"
image = cv2.imread(image_path)
output = image.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.medianBlur(gray, 5)

# === 1. Detect Circles ===
circles = cv2.HoughCircles(
    blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20,
    param1=50, param2=30, minRadius=10, maxRadius=21
)

# === 2. Remove Outlier Circles by Radius ===
if circles is not None:
    circles = np.round(circles[0, :]).astype("int")
    radii = [r for (_, _, r) in circles]
    mean_r = np.mean(radii)
    std_r = np.std(radii)
    filtered_circles = [(x, y, r) for (x, y, r) in circles if abs(r - mean_r) <= 2 * std_r]
else:
    filtered_circles = []


# === 3. Group by Y-axis (with tolerance) ===
def group_by_y(circles, y_margin=10):
    sorted_circles = sorted(circles, key=lambda c: c[1])
    groups = []
    for circle in sorted_circles:
        x, y, r = circle
        placed = False
        for group in groups:
            if abs(group[0][1] - y) <= y_margin:
                group.append(circle)
                placed = True
                break
        if not placed:
            groups.append([circle])
    return groups


# === 4. Group each Y-group by X-axis using mean diff threshold ===
def group_by_x_with_mean_threshold(circles):
    if len(circles) <= 1:
        return [circles]

    sorted_circles = sorted(circles, key=lambda c: c[0])
    xs = [c[0] for c in sorted_circles]
    diffs = [xs[i + 1] - xs[i] for i in range(len(xs) - 1)]
    mean_diff = np.mean(diffs) if diffs else 0

    groups = []
    current_group = [sorted_circles[0]]

    for i in range(1, len(sorted_circles)):
        x_prev = sorted_circles[i - 1][0]
        x_curr = sorted_circles[i][0]
        if x_curr - x_prev <= mean_diff:
            current_group.append(sorted_circles[i])
        else:
            groups.append(current_group)
            current_group = [sorted_circles[i]]
    if current_group:
        groups.append(current_group)
    return groups


# === 5. Count black pixels in a circle ===
def count_black_pixels_in_circles(image, circles):
    black_counts = []
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    for (x, y, r) in circles:
        mask = np.zeros_like(gray)
        cv2.circle(mask, (int(x), int(y)), int(r), 255, -1)
        masked = cv2.bitwise_and(gray, gray, mask=mask)
        count = np.sum((masked <= 100) & (mask == 255))
        black_counts.append(count)
    return black_counts

def black_pixel_presence_by_index(image, final_groups, threshold=10):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    index_black_map = []  # List of sets indicating black pixel presence by index

    for group in final_groups:
        index_has_black = set()

        for idx, (x, y, r) in enumerate(group):
            mask = np.zeros_like(gray)
            cv2.circle(mask, (int(x), int(y)), int(r), 255, -1)
            masked = cv2.bitwise_and(gray, gray, mask=mask)

            has_black = np.any((masked <= threshold) & (mask == 255))
            if has_black:
                index_has_black.add(idx)

        index_black_map.append(index_has_black)

    return index_black_map


# === 6. Process groups and draw ===
final_groups = []
y_groups = group_by_y(filtered_circles, y_margin=10)

for y_group in y_groups:
    x_groups = group_by_x_with_mean_threshold(y_group)
    final_groups.extend(x_groups)

# Assign random colors to groups and annotate
for group in final_groups:
    color = [random.randint(0, 255) for _ in range(3)]
    black_counts = count_black_pixels_in_circles(image, group)
    for (x, y, r), count in zip(group, black_counts):
        cv2.circle(output, (x, y), r, color, 2)
        cv2.putText(output, str(count), (x - r, y - r), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

for i in range(len(final_groups)):
    final_groups[i] = sorted(final_groups[i], key=lambda c: c[0])  # sort by x

zz = black_pixel_presence_by_index(image, final_groups, threshold=50)
flag = 1
for index,i in enumerate(zz):
    if (index+1) % 5 == 0:
        flag+=1
    qno = (index*10) + flag
    print(f"Question {qno} - answer {i}")
# Save the final result
cv2.imwrite("result_grouped.jpg", output)
print(f"Processed {len(filtered_circles)} circles into {len(final_groups)} final groups.")
