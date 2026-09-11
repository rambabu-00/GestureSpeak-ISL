"""
collect_data.py

Part of the GestureSpeak AI ML pipeline.

Command-line tool for collecting REAL Indian Sign Language landmark
data via the local webcam, for later use with `app/ml/train.py`.

THIS FILE SUPPORTS TWO COLLECTION MODES, CHOSEN EXPLICITLY BY THE
OPERATOR AT COLLECTION TIME (never inferred, never asked of the live
end-user during recognition):

  STATIC
    A single hand pose/shape. Natural variation (distance, position
    in frame, slight orientation/hand-shape variation) is captured by
    recording MANY SEPARATE examples, not by recording motion. Each
    captured frame becomes its OWN independent example
    (frame_count=1) -- exactly like the previous version of this
    tool, just re-labeled with type="static" in the new dataset
    format.

  DYNAMIC
    A hand movement/shape-change performed over time. Each requested
    repetition of the gesture is captured as ONE ordered sequence of
    frames (a fixed sequence length chosen up front), written to disk
    as a single group of rows sharing one `example_id` and increasing
    `frame_index` values so temporal order is preserved. Frames within
    a sequence are captured automatically (no key press per frame);
    the operator presses one key to start each full repetition.

Why the dataset format changed (see also app/ml/train.py docstring):
The previous flat CSV (label, synthetic, lm0_x..lm20_z) has no way to
express "these N rows are one ordered sequence" -- consecutive rows
of a dynamic gesture would be indistinguishable from unrelated,
independently-captured static frames. The new format adds
`example_id`, `type`, `frame_index`, and `frame_count` columns so a
sequence can be reconstructed unambiguously, while a static example
is simply the frame_count=1 special case of the same shape.

New CSV format (written to app/ml/data/isl_real_data_v2.csv by
default -- the OLD app/ml/data/isl_real_data.csv / synthetic file
are never written to or modified by this script):

    example_id, label, type, frame_index, frame_count, synthetic,
    lm0_x, lm0_y, lm0_z, ..., lm20_z

  - example_id  : unique id grouping all rows of one example (one
                  static frame, or all frames of one dynamic
                  repetition).
  - label       : gesture label string.
  - type        : "static" or "dynamic".
  - frame_index : 0-based position of this row within its example.
  - frame_count : total number of frames in this example.
  - synthetic   : always False for data written by this tool.
  - lmI_x/y/z   : RAW (un-normalized) landmark coordinates for
                  landmark index I, exactly as previously.

Normalization is intentionally NOT applied here. Exactly as before,
`app.ml.feature_extraction.extract_features()` is called on every
captured frame purely as a VALIDATION step (confirms the sample is
well-formed and normalizable using the exact same function that will
be used at train time and predict time), while raw coordinates are
what get written to disk -- there is a single source of truth for
normalization logic, living only in feature_extraction.py.

Usage:
    python -m app.ml.collect_data
    python -m app.ml.collect_data --output app/ml/data/isl_real_data_v2.csv
    python -m app.ml.collect_data --camera-index 0

Controls (while the webcam window is focused):
    s   - STATIC mode : start/stop automatic per-frame capture
                         (~10 samples/second), each frame saved as
                         its own independent example.
          DYNAMIC mode: start capturing ONE full sequence (the
                         configured number of frames, captured
                         automatically at ~10 frames/second). The
                         tool automatically stops that sequence once
                         it reaches the configured length.
    n   - finish the current label and move on to a new label
    q   - quit the tool
"""

from __future__ import annotations

import argparse
import csv
import time
import uuid
from pathlib import Path

import cv2

from app.ml.feature_extraction import NUM_LANDMARKS, extract_features
from app.services.hand_detector import HandDetector

DEFAULT_OUTPUT_PATH = Path(__file__).resolve().parent / "data" / "isl_real_data_v2.csv"

# Automatic capture cadence, used for both STATIC per-frame capture
# and DYNAMIC per-frame-within-a-sequence capture: ~10 frames/second.
CAPTURE_INTERVAL_SECONDS = 0.1

DEFAULT_STATIC_EXAMPLES = 30
DEFAULT_DYNAMIC_EXAMPLES = 15
DEFAULT_SEQUENCE_LENGTH = 20  # frames per dynamic example (~2 sec at 10fps)

LANDMARK_COLUMNS = [
    f"lm{i}_{axis}" for i in range(NUM_LANDMARKS) for axis in ("x", "y", "z")
]

CSV_HEADER = [
    "example_id",
    "label",
    "type",
    "frame_index",
    "frame_count",
    "synthetic",
] + LANDMARK_COLUMNS


def _ensure_csv_with_header(csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if not csv_path.exists():
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(CSV_HEADER)


def _new_example_id(label: str, example_type: str) -> str:
    return f"{label}_{example_type}_{int(time.time() * 1000)}_{uuid.uuid4().hex[:6]}"


def _validate_and_flatten(mp_landmark_list) -> list:
    """
    mp_landmark_list: the `.landmark` sequence from a MediaPipe
    NormalizedLandmarkList (21 points with .x/.y/.z).

    Validates the sample via the EXISTING extract_features() function
    (same normalization used at train/predict time) and returns the
    raw flattened [x0, y0, z0, x1, y1, z1, ...] coordinate list.
    Raises ValueError if the sample is malformed.
    """
    extract_features(mp_landmark_list)  # validation only; raises on bad input
    flat_coords = []
    for point in mp_landmark_list:
        flat_coords.extend([point.x, point.y, point.z])
    return flat_coords


def _build_row(example_id, label, example_type, frame_index, frame_count, flat_coords):
    return [
        example_id,
        label,
        example_type,
        frame_index,
        frame_count,
        False,
        *flat_coords,
    ]


def _prompt_label() -> str:
    while True:
        label = input(
            "\nEnter a gesture label to collect (or 'quit' to exit): "
        ).strip()
        if label.lower() == "quit":
            return ""
        if label:
            return label
        print("Label cannot be empty.")


def _prompt_mode() -> str:
    while True:
        raw = input("Collection mode -- STATIC or DYNAMIC? [static/dynamic]: ").strip().lower()
        if raw in ("static", "s"):
            return "static"
        if raw in ("dynamic", "d"):
            return "dynamic"
        print("Please enter 'static' or 'dynamic'.")


def _prompt_positive_int(prompt_text: str, default: int) -> int:
    raw = input(f"{prompt_text} [default {default}]: ").strip()
    if not raw:
        return default
    try:
        value = int(raw)
        if value <= 0:
            raise ValueError
        return value
    except ValueError:
        print(f"Invalid number, using default ({default}).")
        return default


def _draw_overlay(frame, lines):
    y = 40
    for text, color in lines:
        cv2.putText(
            frame, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2,
        )
        y += 32


class _NextLabelSignal(Exception):
    """Raised internally to break out to the next label prompt."""


class _QuitSignal(Exception):
    """Raised internally to quit the whole collector."""


def _collect_static(label, target_count, cap, detector, writer, f):
    collected = 0
    collecting = False
    last_capture_time = 0.0

    while True:
        success, frame = cap.read()
        if not success:
            print("Error: Failed to read frame from webcam.")
            return

        processed_frame, mp_hand_landmarks = detector.detect_hands(frame)
        hand_detected = mp_hand_landmarks is not None
        target_reached = collected >= target_count

        if collecting and not target_reached:
            now = time.perf_counter()
            if now - last_capture_time >= CAPTURE_INTERVAL_SECONDS:
                if hand_detected:
                    try:
                        flat_coords = _validate_and_flatten(mp_hand_landmarks.landmark)
                    except ValueError as exc:
                        print(f"Skipped invalid sample: {exc}")
                    else:
                        example_id = _new_example_id(label, "static")
                        row = _build_row(
                            example_id, label, "static", 0, 1, flat_coords
                        )
                        writer.writerow(row)
                        f.flush()
                        collected += 1
                        last_capture_time = now
                        print(f"Captured STATIC example {collected}/{target_count} for '{label}'")
                        if collected >= target_count:
                            collecting = False
                            print(
                                f"Target reached: {collected} STATIC examples "
                                f"saved for '{label}'. Press 'n' for next "
                                "label or 's' to collect more."
                            )

        state_text = "Target reached" if target_reached else (
            "COLLECTING..." if collecting else "Idle"
        )
        hint_text = "[s] start   [n] next label   [q] quit"
        if collecting and not hand_detected:
            hint_text = "No hand detected -- waiting..."

        _draw_overlay(
            processed_frame,
            [
                (f"Label: {label}  Mode: STATIC  {collected}/{target_count}", (0, 255, 0)),
                (state_text, (0, 0, 255) if (collecting and not hand_detected) else (0, 255, 0)),
                (hint_text, (255, 255, 0)),
            ],
        )

        cv2.imshow("GestureSpeak AI - ISL Data Collector", processed_frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            raise _QuitSignal()
        if key == ord("n"):
            print(f"Moving to next label ({collected} STATIC examples saved).")
            raise _NextLabelSignal()
        if key == ord("s"):
            if target_reached:
                print("Target already reached for this label. Press 'n' to move on.")
            elif not collecting:
                collecting = True
                last_capture_time = 0.0
                print(
                    f"Started STATIC auto-capture for '{label}' "
                    f"(~{1 / CAPTURE_INTERVAL_SECONDS:.0f} samples/sec)."
                )


def _collect_dynamic(label, target_examples, sequence_length, cap, detector, writer, f):
    examples_collected = 0
    recording = False
    sequence_buffer = []  # list of flat_coords for the in-progress sequence
    last_capture_time = 0.0

    while True:
        success, frame = cap.read()
        if not success:
            print("Error: Failed to read frame from webcam.")
            return

        processed_frame, mp_hand_landmarks = detector.detect_hands(frame)
        hand_detected = mp_hand_landmarks is not None
        target_reached = examples_collected >= target_examples

        if recording:
            now = time.perf_counter()
            if now - last_capture_time >= CAPTURE_INTERVAL_SECONDS:
                if not hand_detected:
                    print(
                        "Hand lost mid-sequence -- discarding this attempt. "
                        "Press 's' to try the repetition again."
                    )
                    sequence_buffer = []
                    recording = False
                else:
                    try:
                        flat_coords = _validate_and_flatten(mp_hand_landmarks.landmark)
                    except ValueError as exc:
                        print(f"Skipped invalid frame (sequence discarded): {exc}")
                        sequence_buffer = []
                        recording = False
                    else:
                        sequence_buffer.append(flat_coords)
                        last_capture_time = now
                        print(
                            f"  frame {len(sequence_buffer)}/{sequence_length} "
                            f"for DYNAMIC example {examples_collected + 1}/{target_examples}"
                        )
                        if len(sequence_buffer) >= sequence_length:
                            example_id = _new_example_id(label, "dynamic")
                            for idx, coords in enumerate(sequence_buffer):
                                writer.writerow(
                                    _build_row(
                                        example_id, label, "dynamic",
                                        idx, sequence_length, coords,
                                    )
                                )
                            f.flush()
                            examples_collected += 1
                            recording = False
                            sequence_buffer = []
                            print(
                                f"Saved DYNAMIC example {examples_collected}/"
                                f"{target_examples} for '{label}' "
                                f"({sequence_length} frames)."
                            )
                            if examples_collected >= target_examples:
                                print(
                                    f"Target reached: {examples_collected} DYNAMIC "
                                    f"examples saved for '{label}'. Press 'n' for "
                                    "next label or 's' to collect more."
                                )

        state_text = "Target reached" if target_reached else (
            f"RECORDING sequence... ({len(sequence_buffer)}/{sequence_length})"
            if recording else "Idle"
        )
        hint_text = "[s] start next repetition   [n] next label   [q] quit"

        _draw_overlay(
            processed_frame,
            [
                (
                    f"Label: {label}  Mode: DYNAMIC  "
                    f"{examples_collected}/{target_examples}",
                    (0, 255, 0),
                ),
                (state_text, (0, 255, 0)),
                (hint_text, (255, 255, 0)),
            ],
        )

        cv2.imshow("GestureSpeak AI - ISL Data Collector", processed_frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            raise _QuitSignal()
        if key == ord("n"):
            print(f"Moving to next label ({examples_collected} DYNAMIC examples saved).")
            raise _NextLabelSignal()
        if key == ord("s"):
            if target_reached:
                print("Target already reached for this label. Press 'n' to move on.")
            elif not recording:
                recording = True
                sequence_buffer = []
                last_capture_time = 0.0
                print(
                    f"Recording DYNAMIC repetition {examples_collected + 1}/"
                    f"{target_examples} -- perform the gesture now "
                    f"({sequence_length} frames @ "
                    f"~{1 / CAPTURE_INTERVAL_SECONDS:.0f} fps)."
                )


def run_collector(
    output_path: Path = DEFAULT_OUTPUT_PATH,
    default_static_examples: int = DEFAULT_STATIC_EXAMPLES,
    default_dynamic_examples: int = DEFAULT_DYNAMIC_EXAMPLES,
    default_sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
    camera_index: int = 0,
) -> None:
    output_path = Path(output_path)
    _ensure_csv_with_header(output_path)

    detector = HandDetector(max_num_hands=1)
    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print(f"\nSaving REAL (synthetic=False) samples to: {output_path}")
    print(
        "Window controls: 's' = start auto-capture, 'n' = next label, "
        "'q' = quit\n"
    )

    try:
        while True:
            label = _prompt_label()
            if not label:
                break

            mode = _prompt_mode()

            try:
                with open(output_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    if mode == "static":
                        target = _prompt_positive_int(
                            "How many STATIC examples to collect",
                            default_static_examples,
                        )
                        _collect_static(label, target, cap, detector, writer, f)
                    else:
                        target = _prompt_positive_int(
                            "How many DYNAMIC examples (repetitions) to collect",
                            default_dynamic_examples,
                        )
                        seq_len = _prompt_positive_int(
                            "Sequence length in frames per DYNAMIC example",
                            default_sequence_length,
                        )
                        _collect_dynamic(
                            label, target, seq_len, cap, detector, writer, f
                        )
            except _NextLabelSignal:
                continue
            except _QuitSignal:
                print("Quitting collector.")
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description="Collect REAL ISL hand-landmark samples (static or dynamic) via webcam."
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT_PATH),
        help="CSV file to append collected samples to (new sequence-capable format).",
    )
    parser.add_argument(
        "--static-samples",
        type=int,
        default=DEFAULT_STATIC_EXAMPLES,
        help="Default number of STATIC examples to collect per label.",
    )
    parser.add_argument(
        "--dynamic-examples",
        type=int,
        default=DEFAULT_DYNAMIC_EXAMPLES,
        help="Default number of DYNAMIC examples (repetitions) to collect per label.",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=DEFAULT_SEQUENCE_LENGTH,
        help="Default number of frames per DYNAMIC example.",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="OpenCV camera index to use.",
    )
    args = parser.parse_args()

    run_collector(
        output_path=Path(args.output),
        default_static_examples=args.static_samples,
        default_dynamic_examples=args.dynamic_examples,
        default_sequence_length=args.sequence_length,
        camera_index=args.camera_index,
    )


if __name__ == "__main__":
    main()