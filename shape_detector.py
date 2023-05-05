import cv2
import math
import random

# Load all images
images = [
    cv2.imread('Assets\\shape_detect\\round.png'), 
    cv2.imread('Assets\\shape_detect\\square.jpg'), 
    cv2.imread('Assets\\shape_detect\\triangle.png'),
    cv2.imread('Assets\\shape_detect\\dodecagon.png')
]

# Select single image randomly from list of image
img = random.choice(images)

# Preprocess the image to grayscale to simplify further processing
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Perform edge detection to identify border line of an object inside image
edges = cv2.Canny(gray, 100, 200)

# cv2.imshow("Shape Detector", cv2.resize(edges, (500, 500)))
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Find contours and approximate the polygonal curves
contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
shape = "Unknown"
for cnt in contours:
    approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt,True),True)
    sides = len(approx)

    # Validate sides and define what shape
    if sides == 3:
        shape = "Triangle"
    elif sides == 4:
        shape = "Square"
    elif sides == 5:
        shape = "Pentagon"
    elif sides == 6:
        shape = "Hexagon"
    elif sides == 7:
        shape = "Heptagon"
    elif sides == 8:
        shape = "Octagon"
    elif sides == 9:
        shape = "Nonagon"
    elif sides == 10:
        shape = "Decagon"
    elif sides == 11:
        shape = "Hendecagon"
    elif sides == 12:
        shape = "Dodecagon"
    else:
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        circularity = 4 * math.pi * area / (perimeter * perimeter)
        if circularity > 0.8:
            shape = "Circle"

# Put text inside window
cv2.putText(img, shape, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

# Show image and text
cv2.imshow("Shape Detector", cv2.resize(img, (500, 500)))
cv2.waitKey(0)
cv2.destroyAllWindows()

"""
Ada : 
Image preprocessing (Ubah jadi grayscale, )
Edge detection (Buat cari keliling dr obj yg ada didalam gambar dulu sebelum di detect dia shape apa)
Shape detection
"""