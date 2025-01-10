import cv2
import mediapipe as mp

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Initialize video capture (0 for webcam)
video_sources = 0
cap = cv2.VideoCapture(video_sources)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Start Pose estimation
with mp_pose.Pose(static_image_mode=False, model_complexity=0, enable_segmentation=False) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Video capture ended.")
            break
        
        # Flip the frame horizontally
        frame = cv2.flip(frame, 1)
        
        # Convert the frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame to detect pose landmarks
        results = pose.process(frame_rgb)
        
        # If pose landmarks are found, draw them
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
            )
        
        # Resize the frame for display
        display_frame = cv2.resize(frame, (960, 540))
        
        # Show the frame with pose estimation
        cv2.imshow("Pose Estimation", display_frame)
        
        # Exit loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break 

# Release resources and close windows
cap.release()
cv2.destroyAllWindows()
