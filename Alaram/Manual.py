import cv2
import torch
import numpy as np
import pygame


path_alarm = r"D:\Project\Alaram\target-detector-yolov5\Alarm\alarm.wav"


pygame.init()
pygame.mixer.music.load(path_alarm)


model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)


rtsp_url = "rtsp://admin123:suhail243@192.168.100.107/stream1"  # Example RTSP URL
fallback_video = r"D:\Project\Alaram\target-detector-yolov5\Test Videos\thief_video.mp4"


cap = cv2.VideoCapture(rtsp_url)
if not cap.isOpened():
    print("RTSP stream not accessible. Falling back to local video...")
    cap = cv2.VideoCapture(fallback_video)


target_classes = ['car', 'bus', 'truck', 'person']


count = 0
number_of_photos = 3


pts = []

# Function to draw the polygon (Region of Interest)
def draw_polygon(event, x, y, flags, param):
    global pts
    if event == cv2.EVENT_LBUTTONDOWN: 
        print(x, y)
        pts.append([x, y])
    elif event == cv2.EVENT_RBUTTONDOWN: 
        pts = []


def inside_polygon(point, polygon):
    result = cv2.pointPolygonTest(polygon, (point[0], point[1]), False)
    return result >= 0

cv2.namedWindow('Video')
cv2.setMouseCallback('Video', draw_polygon)


def preprocess(img):
    height, width = img.shape[:2]
    ratio = height / width
    img = cv2.resize(img, (640, int(640 * ratio)))
    return img

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to retrieve frame. Exiting...")
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

           
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 3)
            cv2.putText(frame, name, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)

            # Draw the polygon and check for intrusion
            if len(pts) >= 4:
                frame_copy = frame.copy()
                cv2.fillPoly(frame_copy, np.array([pts]), (0, 255, 0))
                frame = cv2.addWeighted(frame_copy, 0.1, frame, 0.9, 0)

                if center_x and center_y and inside_polygon((center_x, center_y), np.array([pts])) and name == 'person':
                    # Save detected image
                    if count < number_of_photos:
                        cv2.imwrite(f"Detected Photos/detected{count}.jpg", frame_detected)
                        count += 1

                    # Play the alarm
                    if not pygame.mixer.music.get_busy():
                        pygame.mixer.music.play()

                    # Highlight detection
                    cv2.putText(frame, "Person Detected", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)

    # Display the video frame
    cv2.imshow("Video", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
