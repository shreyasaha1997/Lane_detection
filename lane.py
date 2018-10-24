import cv2
import numpy as np
image = cv2.imread('t5.jpg')
cap = cv2.VideoCapture("t6.mp4")
def detect_edges(image, low_threshold=50, high_threshold=150):
    return cv2.Canny(image, low_threshold, high_threshold)


def hough_lines(image):
    return cv2.HoughLinesP(image, rho=1, theta=np.pi / 180, threshold=20, minLineLength=20, maxLineGap=300)

def filter_region(image, vertices):
    mask = np.zeros_like(image)
    if len(mask.shape)==2:
        cv2.fillPoly(mask, vertices, 255)
    else:
        cv2.fillPoly(mask, vertices, (255,)*mask.shape[2])
    return cv2.bitwise_and(image, mask)

def select_region(image):
    rows, cols = image.shape[:2]
    bottom_left  = [cols*0.1, rows*0.95]
    top_left     = [cols*0.4, rows*0.6]
    bottom_right = [cols*0.9, rows*0.95]
    top_right    = [cols*0.6, rows*0.6]
    vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
    return filter_region(image, vertices)

def whitemask(image):
    lower = np.uint8([200, 200, 200])
    upper = np.uint8([255, 255, 255])
    white_mask = cv2.inRange(image, lower, upper)
    return white_mask

def yellow_mask(image):
    lower = np.uint8([190, 190,   0])
    upper = np.uint8([255, 255, 255])
    yellow_mask = cv2.inRange(image, lower, upper)
    return  yellow_mask

def get_lane_lines(image):
    mask = cv2.bitwise_or(whitemask(image), yellow_mask(image))
    masked = cv2.bitwise_and(image,image,mask=mask)
    grayscale =cv2.cvtColor(masked, cv2.COLOR_RGB2GRAY)
    smooth = cv2.GaussianBlur(masked, (11,11), 0)
    edges = detect_edges(smooth)
    roi = select_region(edges)
    lines = hough_lines(roi)
    img = image
    for x in range(0, len(lines)):
        for x1,y1,x2,y2 in lines[x]:
            cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)
    return img
img = get_lane_lines(image)
cv2.imshow('lanes_detected',img)

while(True):
    ret, frame = cap.read()
    frame = get_lane_lines(frame)
    cv2.imshow('frame',frame)
    if cv2.waitKey(2) & 0xFF == ord('q'):
        break

cap.release()


cv2.destroyAllWindows()
