/**
 * recognition.js
 *
 * Part of the GestureSpeak AI frontend (Stage 5 UI/UX build).
 *
 * This file is ADDITIVE: it never touches the existing camera-start
 * logic (main.js) or the existing MediaPipe/landmark/prediction logic
 * (handTracking.js). It only:
 *   - toggles the "camera not started" placeholder overlay
 *   - implements a Stop Camera button
 *   - reads the sign text that handTracking.js already writes into
 *     #predictionResult and offers Speak / Add to Sentence / Clear
 *   - implements a simple client-side sentence builder
 *   - logs recognized items to the same localStorage history used by
 *     the History page (history.js)
 */

(function () {
    const video = document.getElementById("camera");
    const placeholder = document.getElementById("cameraPlaceholder");
    const startBtn = document.getElementById("startCameraBtn");
    const stopBtn = document.getElementById("stopCameraBtn");
    const cameraStatusPill = document.getElementById("cameraStatusPill");
    const handStatusPill = document.getElementById("handStatusPill");
    const cameraStatusText = document.getElementById("cameraStatus");
    const handStatusText = document.getElementById("handStatus");

    const predictionResult = document.getElementById("predictionResult");
    const speakSignBtn = document.getElementById("speakSignBtn");
    const addToSentenceBtn = document.getElementById("addToSentenceBtn");
    const clearSignBtn = document.getElementById("clearSignBtn");

    const sentenceTrack = document.getElementById("sentenceTrack");
    const sentenceEmptyMsg = document.getElementById("sentenceEmptyMsg");
    const sentenceOutput = document.getElementById("sentenceOutput");
    const speakSentenceBtn = document.getElementById("speakSentenceBtn");
    const clearSentenceBtn = document.getElementById("clearSentenceBtn");

    if (!video) {
        return; // not on the recognition page
    }

    const HISTORY_KEY = "gesturespeak_history";

    function addHistoryEntry(text) {
        try {
            const raw = localStorage.getItem(HISTORY_KEY);
            const list = raw ? JSON.parse(raw) : [];
            list.unshift({
                id: `${Date.now()}_${Math.random().toString(36).slice(2, 8)}`,
                text,
                time: new Date().toISOString(),
            });
            localStorage.setItem(HISTORY_KEY, JSON.stringify(list.slice(0, 200)));
        } catch (e) {
            console.warn("Could not save to history:", e);
        }
    }

    // --- Extract the plain sign label out of "Predicted sign: HELLO" ---
    function currentSignLabel() {
        if (!predictionResult) return null;
        const text = predictionResult.textContent || "";
        const match = text.split(":");
        if (match.length < 2) return null;
        const label = match.slice(1).join(":").trim();
        if (!label || label === "--") return null;
        return label;
    }

    function speak(text) {
        if (!("speechSynthesis" in window)) {
            alert("Speech output is not supported in this browser.");
            return;
        }
        window.speechSynthesis.cancel();
        const utterance = new SpeechSynthesisUtterance(text);
        window.speechSynthesis.speak(utterance);
    }

    // --- Camera placeholder + Stop Camera ---
    // main.js already calls getUserMedia when #startCameraBtn is
    // clicked and attaches the stream to #camera. We just react to
    // that by hiding the placeholder once the stream is playing.
    if (placeholder) {
        video.addEventListener("playing", () => {
            placeholder.style.display = "none";
            if (stopBtn) stopBtn.disabled = false;
            if (startBtn) startBtn.disabled = true;
        });
    }

    if (stopBtn) {
        stopBtn.addEventListener("click", () => {
            const stream = video.srcObject;
            if (stream) {
                stream.getTracks().forEach((track) => track.stop());
            }
            video.srcObject = null;

            // Also stop the MediaPipe hand-tracking loop running
            // inside handTracking.js. Stopping the raw stream above
            // does not touch its separate `mpCamera` instance, which
            // would otherwise keep calling hands.send() in the
            // background and would then refuse to restart on the
            // next "playing" event.
            video.dispatchEvent(new Event("gesturespeak:stopcamera"));

            if (placeholder) {
                placeholder.style.display = "flex";
                placeholder.querySelector("span:last-child").textContent = "Camera stopped";
            }
            if (cameraStatusText) cameraStatusText.textContent = "Camera stopped";
            if (handStatusText) handStatusText.textContent = "No hand detected";
            if (startBtn) {
                startBtn.disabled = false;
                startBtn.textContent = "▶ Start Camera";
            }
            stopBtn.disabled = true;
        });
    }

    // --- Status pill coloring (never the ONLY signal -- text already
    // communicates state; this is a supplementary visual cue). ---
    function classifyStatus(el, pill, goodWords, warnWords) {
        if (!el || !pill) return;
        const text = (el.textContent || "").toLowerCase();
        pill.classList.remove("status-good", "status-warn", "status-bad");
        if (goodWords.some((w) => text.includes(w))) {
            pill.classList.add("status-good");
        } else if (warnWords.some((w) => text.includes(w))) {
            pill.classList.add("status-warn");
        } else {
            pill.classList.add("status-bad");
        }
    }

    function refreshStatusPills() {
        classifyStatus(cameraStatusText, cameraStatusPill, ["running", "connected"], ["not started", "stopped"]);
        classifyStatus(handStatusText, handStatusPill, ["hand detected"], []);
    }

    const statusObserver = new MutationObserver(refreshStatusPills);
    if (cameraStatusText) statusObserver.observe(cameraStatusText, { childList: true, characterData: true, subtree: true });
    if (handStatusText) statusObserver.observe(handStatusText, { childList: true, characterData: true, subtree: true });
    refreshStatusPills();

    // --- Speak / Clear current sign ---
    if (speakSignBtn) {
        speakSignBtn.addEventListener("click", () => {
            const label = currentSignLabel();
            if (!label) {
                alert("No recognized sign to speak yet.");
                return;
            }
            speak(label);
        });
    }

    if (clearSignBtn) {
        clearSignBtn.addEventListener("click", () => {
            if (predictionResult) predictionResult.textContent = "Predicted sign: --";
            const confidenceResult = document.getElementById("confidenceResult");
            if (confidenceResult) confidenceResult.textContent = "Confidence: --%";
        });
    }

    // --- Sentence builder ---
    let sentenceWords = [];

    function renderSentence() {
        if (sentenceTrack) {
            sentenceTrack.querySelectorAll(".sentence-chip, .sentence-arrow").forEach((n) => n.remove());
        }
        if (sentenceWords.length === 0) {
            if (sentenceEmptyMsg) sentenceEmptyMsg.style.display = "inline";
            if (sentenceOutput) sentenceOutput.textContent = "—";
            return;
        }
        if (sentenceEmptyMsg) sentenceEmptyMsg.style.display = "none";
        sentenceWords.forEach((word, index) => {
            if (index > 0 && sentenceTrack) {
                const arrow = document.createElement("span");
                arrow.className = "sentence-arrow";
                arrow.textContent = "→";
                sentenceTrack.appendChild(arrow);
            }
            if (sentenceTrack) {
                const chip = document.createElement("span");
                chip.className = "sentence-chip";
                chip.textContent = word;
                sentenceTrack.appendChild(chip);
            }
        });
        if (sentenceOutput) {
            sentenceOutput.textContent = `"${sentenceWords.join(" ")}"`;
        }
    }

    if (addToSentenceBtn) {
        addToSentenceBtn.addEventListener("click", () => {
            const label = currentSignLabel();
            if (!label) {
                alert("No recognized sign to add yet.");
                return;
            }
            sentenceWords.push(label);
            renderSentence();
        });
    }

    if (clearSentenceBtn) {
        clearSentenceBtn.addEventListener("click", () => {
            sentenceWords = [];
            renderSentence();
        });
    }

    if (speakSentenceBtn) {
        speakSentenceBtn.addEventListener("click", () => {
            if (sentenceWords.length === 0) {
                alert("Your sentence is empty -- add some signs first.");
                return;
            }
            const sentence = sentenceWords.join(" ");
            speak(sentence);
            addHistoryEntry(sentence);
        });
    }

    renderSentence();
})();
