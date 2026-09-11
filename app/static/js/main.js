document.addEventListener("DOMContentLoaded", function () {

    // Mobile navigation toggle (hamburger menu). The Emergency button
    // lives outside this collapsible menu on purpose, so it stays
    // reachable without opening the menu.
    const navToggle = document.getElementById("navToggle");
    const navbarLinks = document.getElementById("navbarLinks");

    if (navToggle && navbarLinks) {
        navToggle.addEventListener("click", function () {
            const isOpen = navbarLinks.classList.toggle("is-open");
            navToggle.setAttribute("aria-expanded", isOpen ? "true" : "false");
        });

        // Close the mobile menu after a link is chosen.
        navbarLinks.querySelectorAll("a").forEach(function (link) {
            link.addEventListener("click", function () {
                navbarLinks.classList.remove("is-open");
                navToggle.setAttribute("aria-expanded", "false");
            });
        });
    }

    // Start Recognition button (legacy hook -- kept for compatibility;
    // the current homepage links directly to /recognition, so this is
    // a harmless no-op unless a #startRecognitionBtn element exists).
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