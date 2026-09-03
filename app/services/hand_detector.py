"""
hand_detector.py

Part of the GestureSpeak AI project.

This module provides a HandDetector class that wraps Google's MediaPipe
Hands solution for detecting hands and drawing landmarks on OpenCV frames.

This file intentionally contains ONLY hand detection logic.
It does NOT contain any Flask routes/app code and does NOT contain
any webcam capture (cv2.VideoCapture) code. It is meant to be imported
and used by other parts of the GestureSpeak AI project.
"""

import cv2
import mediapipe as mp


class HandDetector:
    """
    A wrapper class around MediaPipe Hands that detects hands in a given
    OpenCV (BGR) frame, draws the hand landmarks/connections on it, and
    returns the processed frame.
    """

    def __init__(
        self,
        static_image_mode: bool = False,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        """
        Initialize the MediaPipe Hands model and drawing utilities.
        """

        self.mp_hands = mp.solutions.hands

        self.hands = self.mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

    def detect_hands(self, frame):
        """
        Detect hands in the given frame, draw landmarks,
        and return both the processed frame and landmarks.
        """

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = self.hands.process(rgb_frame)

        detected_landmarks = None

        if results.multi_hand_landmarks:
            detected_landmarks = results.multi_hand_landmarks[0]

            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style(),
                )

        return frame, detected_landmarks