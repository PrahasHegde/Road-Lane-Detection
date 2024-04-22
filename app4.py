import cv2
import numpy as np

def canny(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = 5
    blur = cv2.GaussianBlur(gray, (kernel, kernel), 0)
    canny = cv2.Canny(blur, 50, 150)
    return canny

def region_of_interest(canny):
    height, width = canny.shape
    mask = np.zeros_like(canny)
    triangle = np.array([[(200, height), (800, 350), (1200, height)]], np.int32)
    cv2.fillPoly(mask, triangle, 255)
    masked_image = cv2.bitwise_and(canny, mask)
    return masked_image

def houghLines(cropped_canny):
    return cv2.HoughLinesP(cropped_canny, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)

def draw_lines(img, lines):
    line_image = np.zeros_like(img)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 10)
    return line_image

def region_of_interest_for_lines(image, lines):
    height, width, _ = image.shape
    mask = np.zeros_like(image)
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(mask, (x1, y1), (x2, y2), (255, 255, 255), 10)
    return mask

def average_slope_intercept(image, lines):
    left_fit, right_fit = [], []
    if lines is None:
        return None
    for line in lines:
        for x1, y1, x2, y2 in line:
            fit = np.polyfit((x1, x2), (y1, y2), 1)
            slope = fit[0]
            intercept = fit[1]
            if slope < 0:
                left_fit.append((slope, intercept))
            else:
                right_fit.append((slope, intercept))
    if len(left_fit) == 0 or len(right_fit) == 0:
        return None
    left_fit_average = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0)
    left_line = make_points(image, left_fit_average)
    right_line = make_points(image, right_fit_average)
    averaged_lines = [left_line, right_line]
    return averaged_lines


def make_points(image, line):
    if line is None:
        return None
    slope, intercept = line
    y1 = image.shape[0]
    y2 = int(y1 * 3 / 5)
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])

cap = cv2.VideoCapture("vid2.mp4")
while(cap.isOpened()):
    _, frame = cap.read()
    canny_image = canny(frame)
    cropped_canny = region_of_interest(canny_image)

    lines = houghLines(cropped_canny)
    averaged_lines = average_slope_intercept(frame, lines)

    line_image = np.zeros_like(frame)
    if averaged_lines is not None:
        for line in averaged_lines:
            if line is not None:
                x1, y1, x2, y2 = line
                cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 10)

    combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    cv2.imshow("result", combo_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
