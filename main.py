import cv2
import dlib
import winsound
import imutils
from scipy.spatial import distance as dist
from imutils import face_utils

# Constants for beep sound on drowsiness
FREQUENCY = 2500  # Hertz
DURATION = 1000   # Milliseconds

# Eye Aspect Ratio calculation function
def calculate_ear(eye):
    # Compute the euclidean distances between the vertical eye landmarks
    vertical_1 = dist.euclidean(eye[1], eye[5])
    vertical_2 = dist.euclidean(eye[2], eye[4])
    
    # Compute the euclidean distance between the horizontal eye landmarks
    horizontal = dist.euclidean(eye[0], eye[3])
    
    # Compute eye aspect ratio
    ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
    return ear

# Thresholds and frame counters
EAR_THRESHOLD = 0.3  # EAR threshold for detecting drowsiness
CONSEC_FRAMES = 48    # Number of consecutive frames with low EAR to trigger alarm
frame_count = 0       # Counter for consecutive low EAR frames

# Initialize dlib's face detector and facial landmarks predictor
SHAPE_PREDICTOR = "C:/Users/bhara/OneDrive/Desktop/DROWZINESS DETECTION  MINI PROJECT/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(SHAPE_PREDICTOR)

# Get facial landmarks indices for the left and right eye
(l_start, l_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(r_start, r_end) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# Initialize video stream (webcam)
cam = cv2.VideoCapture(0)

# Main loop for processing video frames
while True:
    ret, frame = cam.read()
    
    if not ret:
        break
    
    # Resize the frame and convert it to grayscale
    frame = imutils.resize(frame, width=450)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the grayscale frame
    faces = detector(gray_frame, 0)

    # Loop through each face detected
    for face in faces:
        # Get facial landmarks and convert to NumPy array
        shape = predictor(gray_frame, face)
        shape = face_utils.shape_to_np(shape)
        
        # Extract the left and right eye coordinates
        left_eye = shape[l_start:l_end]
        right_eye = shape[r_start:r_end]
        
        # Calculate EAR for both eyes and average them
        left_ear = calculate_ear(left_eye)
        right_ear = calculate_ear(right_eye)
        average_ear = (left_ear + right_ear) / 2.0
        
        # Visualize the eye contours
        cv2.drawContours(frame, [cv2.convexHull(left_eye)], -1, (0, 0, 255), 1)
        cv2.drawContours(frame, [cv2.convexHull(right_eye)], -1, (0, 0, 255), 1)
        
        # Check if EAR falls below the threshold
        if average_ear < EAR_THRESHOLD:
            frame_count += 1
            
            # If eyes are closed for enough frames, trigger drowsiness alert
            if frame_count >= CONSEC_FRAMES:
                cv2.putText(frame, "DROWSINESS DETECTED", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                winsound.Beep(FREQUENCY, DURATION)  # Beep sound (Windows only)
        else:
            # Reset the frame counter if eyes are open
            frame_count = 0

    # Display the frame
    cv2.imshow("Drowsiness Detection", frame)
    
    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cam.release()
cv2.destroyAllWindows()
