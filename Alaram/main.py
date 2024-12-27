import cv2
import torch
import numpy as np
import pygame

# Path to the alarm sound
path_alarm = r"D:\Project\Alaram\target-detector-yolov5\Alarm\alarm.wav"

# Initialize pygame
pygame.init()
pygame.mixer.music.load(path_alarm)

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Attempt to open local video file or default camera
cap = cv2.VideoCapture(r"D:\Project\Alaram\target-detector-yolov5\Test Videos\thief_video.mp4")  # Replace with your video file path
if not cap.isOpened():
    print("Local video not accessible. Switching to default camera...")
    cap = cv2.VideoCapture(0)  # Default camera
    if not cap.isOpened():
        print("Error: Could not access any video source. Exiting...")
        exit()

# Target classes to detect
target_classes = ['car', 'bus', 'truck', 'person']

# Polygon points (manually set the polygon coordinates for the red zone)
pts = [[100, 100], [400, 100], [400, 300], [100, 300]]  # Example rectangle

# Function to check if a point is inside a polygon
def inside_polygon(point, polygon):
    result = cv2.pointPolygonTest(np.array(polygon), (point[0], point[1]), False)
    return result >= 0

# Function to preprocess the frame for YOLOv5 model
def preprocess(img):
    height, width = img.shape[:2]
    ratio = height / width
    img = cv2.resize(img, (640, int(640 * ratio)))
    return img

# Video processing loop
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to retrieve frame.")
        break

    frame_detected = frame.copy()
    frame = preprocess(frame)
    results = model(frame)

    # Iterate over detected objects
    for index, row in results.pandas().xyxy[0].iterrows():
        center_x, center_y = None, None

        if row['name'] in target_classes:
            name = str(row['name'])
            x1, y1 = int(row['xmin']), int(row['ymin'])
            x2, y2 = int(row['xmax']), int(row['ymax'])

            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            # Draw bounding box and labels
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 3)
            cv2.putText(frame, name, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)

            # Draw the polygon (designated area)
            if len(pts) >= 4:
                frame_copy = frame.copy()
                cv2.fillPoly(frame_copy, np.array([pts]), (0, 255, 0))
                frame = cv2.addWeighted(frame_copy, 0.1, frame, 0.9, 0)

                if center_x and center_y and inside_polygon((center_x, center_y), pts) and name == 'person':
                    # Play the alarm automatically
                    if not pygame.mixer.music.get_busy():
                        pygame.mixer.music.play()

                    # Highlight detection
                    cv2.putText(frame, "Person Detected", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)

    # Display the video frame
    cv2.imshow("Video Stream", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
