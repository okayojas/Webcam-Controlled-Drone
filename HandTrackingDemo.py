import cv2
import mediapipe as mp
import numpy as np

joint_to_idx = {
    'wrist' : 0,
    'thumb_cmc' : 1,
    'thumb_mcp' : 2,
    'thumb_ip' : 3,
    'thumb_tip' : 4,
    'index_finger_mcp' : 5,
    'index_finger_pip' : 6,
    'index_finger_dip' : 7,
    'index_finger_tip' : 8,
    'middle_finger_mcp' : 9,
    'middle_finger_pip' : 10,
    'middle_finger_dip' : 11,
    'middle_finger_tip' : 12,
    'ring_finger_mcp' : 13,
    'ring_finger_pip' : 14,
    'ring_finger_dip' : 15,
    'ring_finger_tip' : 16,
    'pinky_cmc' : 17,
    'pinky_mcp' : 18,
    'pinky_ip' : 19,
    'pinky_tip' : 20
    }

# Initialize MediaPipe Gesture Recognizer components.
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    model_complexity=0, 
    min_detection_confidence=0.5, 
    min_tracking_confidence=0.5)

# Initialize the webcam.
cap = cv2.VideoCapture()

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Flip the image horizontally for a later selfie-view display, and convert the BGR image to RGB.
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    
    # To improve performance, mark the image as not writeable to pass by reference.
    image.flags.writeable = False
    
    # Process the image and detect hands.
    results = hands.process(image)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            wrist = hand_landmarks.landmark[joint_to_idx['wrist']]
            x, y = wrist.x, 1-wrist.y
            x = (x-0.5) * 2
            y = (y-0.5) * 2
            print(f"X: {x}, Y: {y}")

    # Display the processed image.
    cv2.imshow('MediaPipe Hands', image)
    if cv2.waitKey(5) & 0xFF == 27:  # Press 'ESC' to exit.
        break

# Release the webcam and destroy all OpenCV windows.
cap.release()
cv2.destroyAllWindows()

