"""
app.ml

Machine-learning pipeline for GestureSpeak AI:

    21 MediaPipe landmarks -> feature_extraction.py -> model.py -> prediction

This package intentionally contains NO hard-coded gesture rules
(no "if thumb is up then THUMBS_UP" style logic). All predictions
come from a trained statistical model.
"""
