import cv2
import mediapipe as mp
import numpy as np

class VirtualAnalogStick:
    def __init__(self):
        # Initialize Mediapipe hand detection and drawing utilities
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils
        self.cap = None
        self.frame_width = 640
        self.frame_height = 480
        self.x = -1
        self.y = -1

    def start(self):
        """Start the video capture."""
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)

    def close(self):
        """Release the video capture and close all OpenCV windows."""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()

    def is_opened(self):
        """Check if the video capture is open."""
        return self.cap.isOpened()

    def update(self):
        """Capture frames from the camera, process hand landmarks, and display the results."""
        if self.cap is None or not self.cap.isOpened():
            print("Camera is not opened.")
            return

        success, image = self.cap.read()
        if not success:
            print("Failed to capture image.")
            return

        # Flip the image horizontally for a later selfie-view display
        image = cv2.flip(image, 1)
        # Convert the BGR image to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process the image and detect hands
        results = self.hands.process(image_rgb)

        # Draw hand landmarks and connections
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                
                # Get wrist coordinates
                wrist_landmark = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
                wrist_x = int(wrist_landmark.x * self.frame_width)
                wrist_y = int(wrist_landmark.y * self.frame_height)
                
                # Draw the virtual analog stick
                self.draw_virtual_analog_stick(image, wrist_x, wrist_y)

        # Display the images
        cv2.imshow('Virtual Analog Stick', image)

        # Close the windows if the 'Esc' key is pressed
        if cv2.waitKey(1) & 0xFF == 27:
            self.close()

    def draw_virtual_analog_stick(self, frame, wrist_x, wrist_y):
        """Draw a virtual analog stick on the frame."""
        stick_center = (self.frame_width // 2, self.frame_height // 2)
        stick_radius = 200
        stick_color = (0, 255, 0)
        dot_color = (0, 0, 255)
        dot_radius = 10

        # Draw the circle representing the analog stick
        cv2.circle(frame, stick_center, stick_radius, stick_color, 2)
        # Draw lines within the circle
        for i in range(-stick_radius, stick_radius+1, 10):
            x = int(np.sqrt(stick_radius**2 - i**2))
            y = int(np.sqrt(stick_radius**2 - i**2))
            # Draw horizontal lines
            cv2.line(frame, (stick_center[0]-x, stick_center[1]+i), (stick_center[0]+x, stick_center[1]+i), stick_color, 1)
            # cv2.line(frame, (stick_center[0]-stick_radius, stick_center[1]+i), (stick_center[0]+stick_radius, stick_center[1]+i), stick_color, 1)
            # Draw vertical lines
            cv2.line(frame, (stick_center[0]+i, stick_center[1]-y), (stick_center[0]+i, stick_center[1]+y), stick_color, 1)
            # cv2.line(frame, (stick_center[0]+i, stick_center[1]-stick_radius), (stick_center[0]+i, stick_center[1]+stick_radius), stick_color, 1)
            
        # Draw the horizontal center line bolded
        cv2.line(frame, (stick_center[0]-stick_radius, stick_center[1]), (stick_center[0]+stick_radius, stick_center[1]), stick_color, 5)
        # Draw the vertical center line bolded
        cv2.line(frame, (stick_center[0], stick_center[1]-stick_radius), (stick_center[0], stick_center[1]+stick_radius), stick_color, 5)
        
        # Calculate the position of the dot within the stick's range
        offset_x = wrist_x - stick_center[0]
        offset_y = wrist_y - stick_center[1]
        distance = np.sqrt(offset_x**2 + offset_y**2)
        
        if distance > stick_radius:
            angle = np.arctan2(offset_y, offset_x)
            offset_x = int(stick_radius * np.cos(angle))
            offset_y = int(stick_radius * np.sin(angle))

        dot_position = (stick_center[0] + offset_x, stick_center[1] + offset_y)
        
        
        # Draw the dot
        cv2.circle(frame, dot_position, dot_radius, dot_color, -1)
        self.x = offset_x / stick_radius
        self.y = offset_y / stick_radius

def main():
    analog_stick = VirtualAnalogStick()
    analog_stick.start()

    try:
        while analog_stick.is_opened():
            analog_stick.update()
    finally:
        analog_stick.close()

if __name__ == "__main__":
    main()

