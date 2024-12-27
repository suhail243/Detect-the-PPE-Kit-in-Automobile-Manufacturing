from flask import Flask, Response, render_template
import cv2
import torch
import numpy as np
import pygame

app = Flask(__name__)

# Path to the alarm sound
path_alarm = r"D:\Project\Alaram\target-detector-yolov5\Alarm\alarm.wav"

# Initialize pygame
pygame.init()
pygame.mixer.music.load(path_alarm)

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Target classes to detect
target_classes = ['car', 'bus', 'truck', 'person']

# Polygon points for the red zone
pts = [[100, 100], [400, 100], [400, 300], [100, 300]]

# Video source
cap = cv2.VideoCapture(r"D:\Project\Alaram\target-detector-yolov5\Test Videos\thief_video.mp4")  # Replace with video file path

# Function to check if a point is inside a polygon
def inside_polygon(point, polygon):
    result = cv2.pointPolygonTest(np.array(polygon), (point[0], point[1]), False)
    return result >= 0

# Video streaming generator
def generate_frames():
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_detected = frame.copy()
        results = model(frame)

        for _, row in results.pandas().xyxy[0].iterrows():
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

                if len(pts) >= 4:
                    frame_copy = frame.copy()
                    cv2.fillPoly(frame_copy, np.array([pts]), (0, 255, 0))
                    frame = cv2.addWeighted(frame_copy, 0.1, frame, 0.9, 0)

                    if inside_polygon((center_x, center_y), pts) and name == 'person':
                        # Play alarm
                        if not pygame.mixer.music.get_busy():
                            pygame.mixer.music.play()

                        # Highlight detection
                        cv2.putText(frame, "Person Detected", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)

        # Encode frame to JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Yield frame as MJPEG
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Flask routes
@app.route('/')
def index():
    return render_template('index.html')  # Renders the HTML template

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)

