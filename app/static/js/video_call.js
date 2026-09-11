/**
 * video_call.js
 *
 * UI-only behavior for the Video Call preview page. There is no real
 * WebRTC connection here -- this only shows a local camera preview
 * and lets the mute/camera/speech/end-call buttons visually toggle,
 * so the interface is ready to be wired to a real calling backend
 * later.
 */

(function () {
    const startPreviewBtn = document.getElementById("startPreviewBtn");
    const localVideo = document.getElementById("localVideo");
    const muteBtn = document.getElementById("muteBtn");
    const cameraToggleBtn = document.getElementById("cameraToggleBtn");
    const speechToggleBtn = document.getElementById("speechToggleBtn");
    const endCallBtn = document.getElementById("endCallBtn");

    if (!localVideo) return;

    let localStream = null;

    if (startPreviewBtn) {
        startPreviewBtn.addEventListener("click", async () => {
            try {
                localStream = await navigator.mediaDevices.getUserMedia({
                    video: true,
                    audio: true,
                });
                localVideo.srcObject = localStream;
                startPreviewBtn.textContent = "Preview Running";
                startPreviewBtn.disabled = true;
            } catch (err) {
                console.error("Camera preview error:", err);
                alert("Unable to access camera/microphone for the preview.");
            }
        });
    }

    function toggleButtonState(btn) {
        if (!btn) return;
        btn.classList.toggle("is-off");
    }

    if (muteBtn) {
        muteBtn.addEventListener("click", () => {
            toggleButtonState(muteBtn);
            if (localStream) {
                localStream.getAudioTracks().forEach((track) => {
                    track.enabled = !muteBtn.classList.contains("is-off");
                });
            }
        });
    }

    if (cameraToggleBtn) {
        cameraToggleBtn.addEventListener("click", () => {
            toggleButtonState(cameraToggleBtn);
            if (localStream) {
                localStream.getVideoTracks().forEach((track) => {
                    track.enabled = !cameraToggleBtn.classList.contains("is-off");
                });
            }
        });
    }

    if (speechToggleBtn) {
        speechToggleBtn.addEventListener("click", () => {
            toggleButtonState(speechToggleBtn);
        });
    }

    if (endCallBtn) {
        endCallBtn.addEventListener("click", () => {
            if (localStream) {
                localStream.getTracks().forEach((track) => track.stop());
                localStream = null;
                localVideo.srcObject = null;
            }
            if (startPreviewBtn) {
                startPreviewBtn.textContent = "Start Camera Preview";
                startPreviewBtn.disabled = false;
            }
            alert("Call ended (preview only -- no real call was in progress).");
        });
    }
})();
