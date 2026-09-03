document.addEventListener("DOMContentLoaded", function () {

    // Start Recognition button
    const startButton = document.getElementById("startRecognitionBtn");

    if (startButton) {
        startButton.addEventListener("click", function () {
            window.location.href = "/recognition";
        });
    }


    // Start Camera button
    const cameraButton = document.getElementById("startCameraBtn");
    const video = document.getElementById("camera");
    const status = document.getElementById("cameraStatus");

    if (cameraButton && video) {

        cameraButton.addEventListener("click", async function () {

            try {

                const stream = await navigator.mediaDevices.getUserMedia({
                    video: true,
                    audio: false
                });

                video.srcObject = stream;

                status.textContent = "Camera is running";

                cameraButton.textContent = "Camera Started";
                cameraButton.disabled = true;

            } catch (error) {

                console.error("Camera error:", error);

                status.textContent =
                    "Unable to access camera. Please allow camera permission.";

            }

        });

    }

});