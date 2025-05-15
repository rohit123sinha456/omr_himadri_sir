import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image
image = cv2.imread(r'D:\omr_himadri_sir\SAMPLE_true.jpg')         # Read image in BGR
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB

# Extract red channel
red_channel = image[:, :, 0]

# Threshold for binarization
threshold = 128  # You can adjust this value

# Binarize: pixels > threshold become 255, else 0
_, binary_red = cv2.threshold(red_channel, threshold, 200, cv2.THRESH_BINARY_INV)

# Plot the original and binarized images

plt.imshow(binary_red, cmap='gray')
plt.title(f'Binarized w.r.t. Red (Threshold={threshold})')
plt.axis('off')

plt.tight_layout()
plt.show()


# Get connected components
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_red, connectivity=8)

# Set minimum area threshold to remove noise
min_area = 50  # Adjust this depending on your image size and noise level

# Prepare output image
output = cv2.cvtColor(binary_red, cv2.COLOR_GRAY2BGR)

# Print and draw only large components
print("Filtered Component Centers (area ≥ {}):".format(min_area))
for i in range(1, num_labels):  # skip background (label 0)
    area = stats[i, cv2.CC_STAT_AREA]
    if area >= min_area:
        x, y = centroids[i]
        print(f"Component {i}: ({x:.2f}, {y:.2f}), Area: {area}")
        cv2.circle(output, (int(x), int(y)), 5, (0, 0, 255), -1)  # red dot

# Display result
plt.imshow(output)
plt.title('Connected Component Centers (Filtered)')
plt.axis('off')
plt.show()





# Compute horizontal histograms
rows = image.shape[0]
red_hist = np.sum(image[:, :, 0], axis=1)     # Red channel row-wise sum
green_hist = np.sum(image[:, :, 1], axis=1)   # Green channel row-wise sum
blue_hist = np.sum(image[:, :, 2], axis=1)    # Blue channel row-wise sum

# Plotting
plt.figure(figsize=(10, 5))
plt.plot(range(rows), red_hist, color='red', label='Red Channel')
# plt.plot(range(rows), green_hist, color='green', label='Green Channel')
# plt.plot(range(rows), blue_hist, color='blue', label='Blue Channel')
plt.title('Horizontal Histogram (Row-wise Intensity Sum)')
plt.xlabel('Row Index')
plt.ylabel('Sum of Intensities')
plt.legend()
plt.tight_layout()
plt.show()
