"""
gesture_classifier.py

Gesture classification module for GestureSpeak AI.
"""


class GestureClassifier:
    """
    Class responsible for recognizing gestures from hand landmarks.
    """

    def __init__(self):
        pass

    def is_open_palm(self, landmarks):
        index_up = landmarks[8].y < landmarks[6].y
        middle_up = landmarks[12].y < landmarks[10].y
        ring_up = landmarks[16].y < landmarks[14].y
        pinky_up = landmarks[20].y < landmarks[18].y

        return index_up and middle_up and ring_up and pinky_up

    def is_fist(self, landmarks):
        index_down = landmarks[8].y > landmarks[6].y
        middle_down = landmarks[12].y > landmarks[10].y
        ring_down = landmarks[16].y > landmarks[14].y
        pinky_down = landmarks[20].y > landmarks[18].y

        return index_down and middle_down and ring_down and pinky_down

    def is_thumbs_up(self, landmarks):
        thumb_tip = landmarks[4]
        thumb_ip = landmarks[3]

        thumb_up = thumb_tip.y < thumb_ip.y

        index_down = landmarks[8].y > landmarks[6].y
        middle_down = landmarks[12].y > landmarks[10].y
        ring_down = landmarks[16].y > landmarks[14].y
        pinky_down = landmarks[20].y > landmarks[18].y

        return (
            thumb_up
            and index_down
            and middle_down
            and ring_down
            and pinky_down
        )

    def is_peace(self, landmarks):
        index_up = landmarks[8].y < landmarks[6].y
        middle_up = landmarks[12].y < landmarks[10].y

        ring_down = landmarks[16].y > landmarks[14].y
        pinky_down = landmarks[20].y > landmarks[18].y

        return (
            index_up
            and middle_up
            and ring_down
            and pinky_down
        )

    def is_pointing(self, landmarks):
        index_up = landmarks[8].y < landmarks[6].y

        middle_down = landmarks[12].y > landmarks[10].y
        ring_down = landmarks[16].y > landmarks[14].y
        pinky_down = landmarks[20].y > landmarks[18].y

        return (
            index_up
            and middle_down
            and ring_down
            and pinky_down
        )

    def classify(self, hand_landmarks):
        """
        Classify the detected hand gesture.
        """

        if hand_landmarks is None:
            return "No Hand"

        landmarks = hand_landmarks.landmark

        if self.is_open_palm(landmarks):
            return "OPEN PALM"

        if self.is_fist(landmarks):
            return "FIST"

        if self.is_thumbs_up(landmarks):
            return "THUMBS UP"

        if self.is_peace(landmarks):
            return "PEACE"

        if self.is_pointing(landmarks):
            return "POINTING"

        return "HAND DETECTED"