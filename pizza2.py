import cv2
import numpy as np

# Load the image
image = cv2.imread('ccc.png')
image = cv2.resize(image, (600, 600))  # Resize for easier processing
cv2.imshow("Original Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Step 1: Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray Image", gray)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Step 2: Apply Gaussian blur to the grayscale image to reduce noise
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
cv2.imshow("Blurred Image", blurred)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Step 3: Use Canny edge detection to find edges
edges = cv2.Canny(blurred, 50, 150)  # Adjust these thresholds if needed
cv2.imshow("Canny Edges", edges)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Step 4: Perform morphological closing to fill gaps in the edges (optional, for better results)
kernel = np.ones((5, 5), np.uint8)  # Define a kernel for morphological operations
closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
cv2.imshow("Closed Edges", closed_edges)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Step 5: Find contours in the edge-detected image
contours, _ = cv2.findContours(closed_edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Step 6: Iterate over contours to find circular shapes
output = image.copy()
for contour in contours:
    # Approximate the contour to a polygon
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    
    # Check if the contour is approximately a circle by checking its circularity
    if len(approx) > 5:  # Circles typically have more than 5 points in the approximation
        # Get the bounding circle for the contour
        (x, y), radius = cv2.minEnclosingCircle(contour)
        
        # Calculate the aspect ratio of the bounding circle (it should be nearly 1 for a circle)
        aspect_ratio = 4 * np.pi * cv2.contourArea(contour) / (cv2.arcLength(contour, True) ** 2)
        
        # If the aspect ratio is close to that of a circle, consider it a detected circle
        if aspect_ratio > 0.7:  # Relax the threshold for circularity, allowing for imperfections
            # Draw the circle on the output image
            cv2.circle(output, (int(x), int(y)), int(radius), (0, 255, 0), 3)
            # Optionally, draw the center
            cv2.circle(output, (int(x), int(y)), 5, (0, 0, 255), -1)

# Step 7: Show the result with the detected circle
cv2.imshow("Detected Pizza Circle", output)
cv2.waitKey(0)
cv2.destroyAllWindows()
