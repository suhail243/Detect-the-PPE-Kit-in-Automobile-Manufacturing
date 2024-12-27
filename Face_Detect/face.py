!pip install face_recognition
!pip install dlib
!pip install opencv-python-headless
import cv2
from google.colab.patches import cv2_imshow # Import cv2_imshow
import face_recognition
import numpy as np

# Load the pre-approved worker list
def load_known_faces(known_face_encodings, known_face_names):
    # Example: Add known face encodings and their corresponding names
    known_face_encodings.append(face_recognition.face_encodings(face_recognition.load_image_file("/content/s.jpg"))[0])
    known_face_names.append("Worker 1")
    known_face_encodings.append(face_recognition.face_encodings(face_recognition.load_image_file("/content/s.jpg"))[0])
    known_face_names.append("Worker 1")
    return known_face_encodings, known_face_names

# Initialize known faces
known_face_encodings = []
known_face_names = []
known_face_encodings, known_face_names = load_known_faces(known_face_encodings, known_face_names)

# Initialize video capture
video_capture = cv2.VideoCapture('/content/b.mp4')

# Function to detect PPE compliance (Stub for integration)
def check_ppe_compliance(frame):
    # Placeholder function for PPE compliance detection
    # Replace with the PPE detection module logic
    return True

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    # Resize frame for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Detect face locations and encode faces
    face_locations = face_recognition.face_locations(rgb_small_frame)
    # The face_encodings function should be called with only the image,
    # not the face_locations. It will detect and encode faces automatically.
    face_encodings = face_recognition.face_encodings(rgb_small_frame)

    for face_encoding, face_location in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)

        # Check if the face matches any known worker
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
        else:
            name = "Unknown"

        # Display the results
        top, right, bottom, left = face_location
        top, right, bottom, left = top * 4, right * 4, bottom * 4, left * 4

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

```python
!pip install face_recognition
!pip install dlib
!pip install opencv-python-headless
import cv2
from google.colab.patches import cv2_imshow # Import cv2_imshow
import face_recognition
import numpy as np

# Load the pre-approved worker list
def load_known_faces(known_face_encodings, known_face_names):
    # Example: Add known face encodings and their corresponding names
    known_face_encodings.append(face_recognition.face_encodings(face_recognition.load_image_file("/content/s.jpg"))[0])
    known_face_names.append("Worker 1")
    known_face_encodings.append(face_recognition.face_encodings(face_recognition.load_image_file("/content/s.jpg"))[0])
    known_face_names.append("Worker 1")
    return known_face_encodings, known_face_names

# Initialize known faces
known_face_encodings = []
known_face_names = []
known_face_encodings, known_face_names = load_known_faces(known_face_encodings, known_face_names)

# Initialize video capture
video_capture = cv2.VideoCapture('/content/b.mp4')

# Function to detect PPE compliance (Stub for integration)
def check_ppe_compliance(frame):
    # Placeholder function for PPE compliance detection
    # Replace with the PPE detection module logic
    return True

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    # Resize frame for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Detect face locations and encode faces
    face_locations = face_recognition.face_locations(rgb_small_frame)
    # The face_encodings function should be called with only the image,
    # not the face_locations. It will detect and encode faces automatically.
    face_encodings = face_recognition.face_encodings(rgb_small_frame)

    for face_encoding, face_location in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)

        # Check if the face matches any known worker
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
        else:
            name = "Unknown"

        # Display the results
        top, right, bottom, left = face_location
        top, right, bottom, left = top * 4, right * 4, bottom * 4, left * 4

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        # Trigger PPE detection
        if name != "Unknown" and check_ppe_compliance(frame):
            cv2.putText(frame, "PPE Compliant", (left, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2) #Corrected color code

    # Display the video feed
    cv2_imshow(frame)

    # Break loop with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
video_capture.release()
cv2.destroyAllWindows()
```
    cv2_imshow(frame)

    # Break loop with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
video_capture.release()
cv2.destroyAllWindows()
