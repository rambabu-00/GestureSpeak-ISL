/**
 * handTracking.js
 *
 * Part of the GestureSpeak AI project.
 *
 * This file connects the ALREADY-RUNNING browser webcam (started in
 * main.js via getUserMedia) to MediaPipe's browser-based "Hands"
 * solution. It draws the 21 hand landmark points + connections on a
 * transparent <canvas> positioned on top of the <video>, and updates
 * a simple "Hand detected" / "No hand detected" status message.
 *
 * IMPORTANT (read this before extending the file):
 * MediaPipe here ONLY detects hand landmark POINTS (their x/y/z
 * positions). It does NOT recognize Indian Sign Language on its own.
 * Turning these landmark coordinates into actual ISL letters/words is
 * handled server-side: this file sends each detected hand's 21
 * landmarks to the Flask `/predict` endpoint (app/ml/*), which runs a
 * trained ML classifier and returns a predicted label + confidence.
 * Do not hard-code gesture rules in this file (e.g. "if fingers look
 * like X, show HELLO") -- all classification logic lives in the
 * backend ML pipeline, not here.
 *
 * This file does landmark detection + drawing + status text, and
 * forwards landmarks to the backend for prediction + displays the
 * result. It does not itself do any gesture classification.
 */

(function () {
    const video = document.getElementById("camera");
    const canvas = document.getElementById("landmarkCanvas");
    const handStatus = document.getElementById("handStatus");
    const coordsBox = document.getElementById("landmarkCoords");
    const predictionResult = document.getElementById("predictionResult");
    const confidenceResult = document.getElementById("confidenceResult");
    const predictionNote = document.getElementById("predictionNote");

    // If these elements aren't on the page, we're not on the
    // recognition page -- do nothing.
    if (!video || !canvas) {
        return;
    }

    const ctx = canvas.getContext("2d");

    // --- ML prediction (landmarks -> Flask /predict -> label) ---
    //
    // We throttle predictions instead of sending on every single
    // MediaPipe frame (which can fire 30+ times/sec) to avoid
    // overwhelming the Flask server with requests.
    const PREDICTION_INTERVAL_MS = 400;
    let lastPredictionAt = 0;
    let predictionInFlight = false;

    function updatePredictionUI(result) {
        if (!predictionResult || !confidenceResult) {
            return;
        }

        if (result.status === "ok") {
            predictionResult.textContent = `Predicted sign: ${result.prediction}`;
            confidenceResult.textContent =
                `Confidence: ${(result.confidence * 100).toFixed(1)}%`;
            if (predictionNote) {
                predictionNote.textContent = result.warning || "";
            }
        } else if (result.status === "model_unavailable") {
            predictionResult.textContent = "Predicted sign: --";
            confidenceResult.textContent = "Confidence: --%";
            if (predictionNote) {
                predictionNote.textContent =
                    "No trained model available yet. Train the model " +
                    "(see README) before predictions can be made.";
            }
        } else {
            predictionResult.textContent = "Predicted sign: --";
            confidenceResult.textContent = "Confidence: --%";
            if (predictionNote) {
                predictionNote.textContent =
                    result.message || "Prediction unavailable.";
            }
        }
    }

    async function sendLandmarksForPrediction(landmarks) {
        if (predictionInFlight) {
            return;
        }
        predictionInFlight = true;
        try {
            const response = await fetch("/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    landmarks: landmarks.map((point) => ({
                        x: point.x,
                        y: point.y,
                        z: point.z,
                    })),
                }),
            });
            const result = await response.json();
            updatePredictionUI(result);
        } catch (error) {
            console.error("Prediction request failed:", error);
            if (predictionNote) {
                predictionNote.textContent =
                    "Could not reach the prediction server.";
            }
        } finally {
            predictionInFlight = false;
        }
    }

    function maybePredict(landmarks) {
        const now = performance.now();
        if (now - lastPredictionAt < PREDICTION_INTERVAL_MS) {
            return;
        }
        lastPredictionAt = now;
        sendLandmarksForPrediction(landmarks);
    }

    // Set up the MediaPipe Hands model. `locateFile` tells it where
    // to download its internal model files from (a public CDN).
    const hands = new Hands({
        locateFile: (file) =>
            `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`,
    });

    hands.setOptions({
        maxNumHands: 2,
        modelComplexity: 1,
        minDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5,
    });

    hands.onResults(onResults);

    /**
     * Called by MediaPipe every time it has finished analyzing one
     * video frame. `results.multiHandLandmarks` is an array of hands
     * found, each containing 21 landmark points (x, y normalized
     * 0-1 relative to the frame, z = relative depth).
     */
    function onResults(results) {
        // Keep the overlay canvas the exact same size as the video.
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;

        ctx.save();
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        const handsFound =
            results.multiHandLandmarks &&
            results.multiHandLandmarks.length > 0;

        if (handsFound) {
            for (const landmarks of results.multiHandLandmarks) {
                drawConnectors(ctx, landmarks, HAND_CONNECTIONS, {
                    color: "#6C5CE7",
                    lineWidth: 3,
                });
                drawLandmarks(ctx, landmarks, {
                    color: "#00e6a8",
                    lineWidth: 1,
                    radius: 3,
                });
            }

            if (handStatus) {
                handStatus.textContent = "Hand detected";
            }

            const firstHand = results.multiHandLandmarks[0];

            if (coordsBox) {
                // Debug only: show raw coordinates for the first
                // detected hand's 21 landmarks.
                const lines = firstHand.map((point, index) => {
                    const x = point.x.toFixed(3);
                    const y = point.y.toFixed(3);
                    const z = point.z.toFixed(3);
                    return `${index}: x=${x} y=${y} z=${z}`;
                });
                coordsBox.textContent = lines.join("\n");
            }

            // Forward this hand's 21 landmarks to the backend ML
            // pipeline for a real prediction (throttled).
            maybePredict(firstHand);
        } else {
            if (handStatus) {
                handStatus.textContent = "No hand detected";
            }
            if (coordsBox) {
                coordsBox.textContent = "";
            }
            if (predictionResult) {
                predictionResult.textContent = "Predicted sign: --";
            }
            if (confidenceResult) {
                confidenceResult.textContent = "Confidence: --%";
            }
        }

        ctx.restore();
    }

    // MediaPipe's Camera helper repeatedly grabs the current video
    // frame and feeds it into `hands`. We only start it once the
    // video is actually playing (i.e. after the user has clicked
    // "Start Camera" in main.js and granted permission).
    let mpCamera = null;

    function startHandTracking() {
        if (mpCamera) {
            return; // already running, don't start a second loop
        }

        mpCamera = new Camera(video, {
            onFrame: async () => {
                await hands.send({ image: video });
            },
            width: 1280,
            height: 720,
        });

        mpCamera.start();
    }

    video.addEventListener("playing", startHandTracking);
})();
