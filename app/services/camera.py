"""
camera.py

Part of the GestureSpeak AI project.

This module opens the default webcam, continuously reads frames, sends
each frame to HandDetector.detect_hands() for hand landmark detection,
and displays the processed frame in a window. Press 'q' to quit.

This file intentionally contains ONLY webcam capture/display logic.
It does NOT contain any Flask routes/app code and does NOT contain
any HTML.
"""
import cv2

from app.services.gesture_classifier import GestureClassifier
from app.services.hand_detector import HandDetector


def main():
    """
    Open the default webcam, run hand detection on each frame, and
    display the result until the user presses 'q'.
    """
    # Create an instance of the hand detector
    detector = HandDetector()
    classifier = GestureClassifier()

    # Open the default webcam (index 0)
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Check if the webcam opened successfully
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    try:
        while True:
            # Read a single frame from the webcam
            success, frame = cap.read()

            # If the frame was not read successfully, stop the loop
            if not success:
                print("Error: Failed to read frame from webcam.")
                break

            # Send the frame to the detector for hand detection and
            # landmark drawing
            processed_frame, landmarks = detector.detect_hands(frame)
            gesture = classifier.classify(landmarks)
            cv2.putText(
                processed_frame,
                gesture,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
                )
            # Display the processed frame in a window
            cv2.imshow("GestureSpeak AI - Hand Detection", processed_frame)

            # Wait 1ms for a key press; quit the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        # Always release the camera and destroy windows, even if an
        # error occurs, to properly free system resources
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()