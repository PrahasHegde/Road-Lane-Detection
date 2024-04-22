#LaneDetection -> Video

import cv2
import numpy as np

def grayscale(img):
    """Convert an image to grayscale."""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def gaussian_blur(img, kernel_size):
    """Apply Gaussian blur to the image."""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def canny(img, low_threshold, high_threshold):
    """Apply Canny edge detection to the image."""
    return cv2.Canny(img, low_threshold, high_threshold)

def region_of_interest(img, vertices):
    """Apply a mask to the image to focus on a region of interest."""
    mask = np.zeros_like(img)
    if len(img.shape) > 2:
        channel_count = img.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """Draw detected lane lines on the image."""
    left_lines = []
    right_lines = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            slope = (y2 - y1) / (x2 - x1)
            if slope < 0:
                left_lines.append((x1, y1))
                left_lines.append((x2, y2))
            else:
                right_lines.append((x1, y1))
                right_lines.append((x2, y2))

    if len(left_lines) > 0:
        left_line = np.polyfit(*zip(*left_lines), 1)
        cv2.line(img, (int((img.shape[0] - left_line[1]) / left_line[0]), img.shape[0]),
                 (int((350 - left_line[1]) / left_line[0]), 350), color, thickness)

    if len(right_lines) > 0:
        right_line = np.polyfit(*zip(*right_lines), 1)
        cv2.line(img, (int((img.shape[0] - right_line[1]) / right_line[0]), img.shape[0]),
                 (int((350 - right_line[1]) / right_line[0]), 350), color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """Apply Hough Transform to detect lines in the image."""
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    if lines is None:
        return img
    line_img = np.zeros((*img.shape, 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """Overlay two images with different weights."""
    return cv2.addWeighted(initial_img, α, img, β, λ)



def process_image(image):
    """Process a single image frame to detect lane lines."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    vertices = np.array([[(100, 540), (460, 320), (500, 320), (960, 540)]], dtype=np.int32)
    masked_edges = region_of_interest(edges, vertices)
    lines = cv2.HoughLinesP(masked_edges, 1, np.pi / 180, 50, np.array([]), minLineLength=100, maxLineGap=160)
    line_img = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return weighted_img(line_img, image)

# Set the output video file name
output = 'car_lane_detection2.mp4'
# Open the input video file
cap = cv2.VideoCapture("vid1.mp4")
# Get the FPS and frame size of the input video
fps = cap.get(cv2.CAP_PROP_FPS)
size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
# Create a VideoWriter object for the output video
out = cv2.VideoWriter(output, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)

# Process each frame of the input video
while True:
    ret, frame = cap.read()
    if not ret:
        break
    processed_frame = process_image(frame)
    out.write(processed_frame)
    cv2.imshow('result', processed_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and video writer objects
cap.release()
out.release()
cv2.destroyAllWindows()
