/**
 * video_call.js
 *
 * GestureSpeak AI group video call.
 *
 * Architecture:
 *   Browser <--Socket.IO (signaling only)--> Flask-SocketIO (app/sockets.py)
 *   Browser <--WebRTC (audio/video, peer-to-peer mesh)--> Browser
 *
 * No video/audio ever passes through the Flask server -- Socket.IO is
 * only used to exchange room membership and WebRTC offer/answer/ICE
 * candidate messages. Once connected, each pair of participants has a
 * direct RTCPeerConnection to every other participant (mesh), which
 * is why this is only recommended for small groups (a handful of
 * participants) rather than large ones.
 *
 * IMPORTANT: this file intentionally identifies "my own id" using the
 * `sid` field the SERVER sends back in the "joined" event, not any
 * client-side socket.id-style property. This was verified during
 * backend testing to be the reliable, unambiguous way to know which
 * id peers will use to address messages to us.
 */

(function () {
    const joinPanel = document.getElementById("joinPanel");
    const callPanel = document.getElementById("callPanel");
    if (!joinPanel || !callPanel) return; // not on the video call page

    const roomIdInput = document.getElementById("roomIdInput");
    const generateRoomBtn = document.getElementById("generateRoomBtn");
    const joinRoomBtn = document.getElementById("joinRoomBtn");
    const joinError = document.getElementById("joinError");

    const activeRoomId = document.getElementById("activeRoomId");
    const connectionStatus = document.getElementById("connectionStatus");
    const copyRoomBtn = document.getElementById("copyRoomBtn");
    const leaveRoomBtn = document.getElementById("leaveRoomBtn");
    const callError = document.getElementById("callError");
    const videoGrid = document.getElementById("videoGrid");

    const muteBtn = document.getElementById("muteBtn");
    const cameraToggleBtn = document.getElementById("cameraToggleBtn");
    const screenShareBtn = document.getElementById("screenShareBtn");
    const endCallBtn = document.getElementById("endCallBtn");

    // Free public STUN server only -- no TURN server is configured.
    // This is sufficient for local/same-network testing but may not
    // work across arbitrary real-world NATs (see the banner on the
    // page for this limitation).
    const ICE_SERVERS = [{ urls: "stun:stun.l.google.com:19302" }];

    let socket = null;
    let mySid = null;
    let currentRoom = null;
    let localStream = null;
    let screenStream = null;
    let isScreenSharing = false;
    let isJoining = false; // guards against getUserMedia() being called twice concurrently

    // sid -> { pc: RTCPeerConnection, tile: HTMLElement, videoEl: HTMLVideoElement }
    const peers = {};

    function setStatus(text) {
        if (connectionStatus) connectionStatus.textContent = text;
    }

    function showJoinError(message) {
        if (joinError) joinError.textContent = message || "";
    }

    function showCallError(message) {
        if (callError) callError.textContent = message || "";
    }

    function randomRoomId() {
        const chars = "abcdefghijklmnopqrstuvwxyz0123456789";
        let id = "";
        for (let i = 0; i < 8; i++) {
            id += chars[Math.floor(Math.random() * chars.length)];
        }
        return id;
    }

    if (generateRoomBtn) {
        generateRoomBtn.addEventListener("click", () => {
            roomIdInput.value = randomRoomId();
        });
    }

    // --- Video tile management -----------------------------------------

    function updateGridLayout() {
        if (!videoGrid) return;
        const count = videoGrid.children.length;
        videoGrid.classList.remove(
            "grid-count-1", "grid-count-2", "grid-count-3-4", "grid-count-5-6"
        );
        if (count <= 1) {
            videoGrid.classList.add("grid-count-1");
        } else if (count === 2) {
            videoGrid.classList.add("grid-count-2");
        } else if (count <= 4) {
            videoGrid.classList.add("grid-count-3-4");
        } else {
            videoGrid.classList.add("grid-count-5-6");
        }
    }

    function createTile(id, label, isLocal) {
        const tile = document.createElement("div");
        tile.className = "video-tile grid-tile" + (isLocal ? " is-local" : "");
        tile.dataset.participant = id;

        const video = document.createElement("video");
        video.autoplay = true;
        video.playsInline = true;
        if (isLocal) video.muted = true; // never play back our own audio

        const labelEl = document.createElement("span");
        labelEl.className = "video-tile-label";
        labelEl.textContent = label;

        const placeholder = document.createElement("div");
        placeholder.className = "video-tile-placeholder";
        placeholder.innerHTML = '<span aria-hidden="true">🤟</span><span>Camera off</span>';
        placeholder.hidden = true;

        tile.appendChild(video);
        tile.appendChild(placeholder);
        tile.appendChild(labelEl);
        videoGrid.appendChild(tile);
        updateGridLayout();

        return { tile, video, placeholder };
    }

    function removeTile(id) {
        const peer = peers[id];
        if (peer && peer.tile && peer.tile.parentNode) {
            peer.tile.parentNode.removeChild(peer.tile);
        }
        updateGridLayout();
    }

    function setTileCameraOff(videoEl, placeholderEl, off) {
        if (!videoEl || !placeholderEl) return;
        placeholderEl.hidden = !off;
        videoEl.style.visibility = off ? "hidden" : "visible";
    }

    // --- WebRTC peer connection management -------------------------------

    function createPeerConnection(remoteSid) {
        const pc = new RTCPeerConnection({ iceServers: ICE_SERVERS });

        if (localStream) {
            localStream.getTracks().forEach((track) => pc.addTrack(track, localStream));
        }

        pc.onicecandidate = (event) => {
            if (event.candidate) {
                socket.emit("ice-candidate", {
                    room: currentRoom,
                    to: remoteSid,
                    candidate: event.candidate,
                });
            }
        };

        pc.onconnectionstatechange = () => {
            if (pc.connectionState === "failed" || pc.connectionState === "disconnected") {
                showCallError(
                    `Lost connection to a participant (${pc.connectionState}). ` +
                    "This can happen without a TURN server across some networks."
                );
            }
        };

        const { tile, video, placeholder } = createTile(remoteSid, `Participant ${remoteSid.slice(0, 5)}`, false);

        pc.ontrack = (event) => {
            video.srcObject = event.streams[0];
            const [remoteVideoTrack] = event.streams[0].getVideoTracks();
            if (remoteVideoTrack) {
                setTileCameraOff(video, placeholder, false);
                // Best-effort remote "camera off" detection via the
                // standard WebRTC track mute/unmute events. Browser
                // support/reliability for this varies -- if a given
                // browser doesn't fire these reliably, the remote
                // tile may show a frozen last frame instead of the
                // placeholder, which is a known limitation.
                remoteVideoTrack.onmute = () => setTileCameraOff(video, placeholder, true);
                remoteVideoTrack.onunmute = () => setTileCameraOff(video, placeholder, false);
            }
        };

        peers[remoteSid] = { pc, tile, video, placeholder };
        return pc;
    }

    async function callPeer(remoteSid) {
        const pc = createPeerConnection(remoteSid);
        const offer = await pc.createOffer();
        await pc.setLocalDescription(offer);
        socket.emit("offer", { room: currentRoom, to: remoteSid, sdp: offer });
    }

    async function handleOffer(fromSid, sdp) {
        const pc = peers[fromSid] ? peers[fromSid].pc : createPeerConnection(fromSid);
        await pc.setRemoteDescription(new RTCSessionDescription(sdp));
        const answer = await pc.createAnswer();
        await pc.setLocalDescription(answer);
        socket.emit("answer", { room: currentRoom, to: fromSid, sdp: answer });
    }

    async function handleAnswer(fromSid, sdp) {
        const peer = peers[fromSid];
        if (!peer) return;
        await peer.pc.setRemoteDescription(new RTCSessionDescription(sdp));
    }

    async function handleIceCandidate(fromSid, candidate) {
        const peer = peers[fromSid];
        if (!peer) return;
        try {
            await peer.pc.addIceCandidate(new RTCIceCandidate(candidate));
        } catch (err) {
            console.warn("Could not add ICE candidate:", err);
        }
    }

    function closePeer(sid) {
        const peer = peers[sid];
        if (!peer) return;

        if (peer.pc) {
            peer.pc.close();
        }

        if (sid !== "local") {
            removeTile(sid);
        }

        delete peers[sid];
    }

    function closeAllPeers() {
        Object.keys(peers).forEach(closePeer);
    }

    // --- Join / Leave flow -------------------------------------------------

    async function joinRoom(room) {
        // Prevent a double-click (or any re-entrant call) from firing a
        // second getUserMedia() request while one is already in flight
        // or while a camera/microphone stream from a previous join is
        // still active -- this is what previously caused
        // "Device in use" / NotReadableError.
        if (isJoining || localStream) {
            return;
        }
        isJoining = true;

        showJoinError("");

        try {
            localStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
        } catch (err) {
            console.error("getUserMedia error:", err);
            if (err.name === "NotAllowedError" || err.name === "PermissionDeniedError") {
                showJoinError("Camera/microphone permission was denied. Please allow access and try again.");
            } else if (err.name === "NotFoundError") {
                showJoinError("No camera or microphone was found on this device.");
            } else if (err.name === "NotReadableError") {
                showJoinError("Your camera or microphone is already in use by another application or tab. Close it and try again.");
            } else {
                showJoinError("Could not access camera/microphone: " + err.message);
            }
            isJoining = false;
            return;
        }

        currentRoom = room;
        joinPanel.hidden = true;
        callPanel.hidden = false;
        activeRoomId.textContent = room;
        setStatus("Connecting…");
        showCallError("");

        // Local tile always shown first.
        const localTile = createTile("local", "You", true);
        localTile.video.srcObject = localStream;

        peers["local"] = {
            pc: null,
            tile: localTile.tile,
            video: localTile.video,
            placeholder: localTile.placeholder
        };

        socket = io();

        socket.on("connect", () => {
            setStatus("Connected to signaling server");
            socket.emit("join", { room: currentRoom });
        });

        socket.on("connect_error", () => {
            setStatus("Connection error");
            showCallError("Could not connect to the signaling server. Check your connection and try again.");
        });

        socket.on("disconnect", () => {
            setStatus("Disconnected");
        });

        socket.on("join-error", (data) => {
            showCallError((data && data.message) || "Could not join that room.");
        });

        socket.on("joined", (data) => {
            mySid = data.sid;
            setStatus(`In room "${data.room}" — waiting for others to join`);
        });

        socket.on("user-joined", (data) => {
            setStatus("A participant joined — connecting…");
            callPeer(data.sid);
        });

        socket.on("offer", (data) => {
            handleOffer(data.from, data.sdp);
        });

        socket.on("answer", (data) => {
            handleAnswer(data.from, data.sdp);
        });

        socket.on("ice-candidate", (data) => {
            handleIceCandidate(data.from, data.candidate);
        });

        socket.on("user-left", (data) => {
            closePeer(data.sid);
            setStatus("A participant left the call");
        });
    }

    function leaveRoom() {
        if (socket) {
            if (currentRoom) socket.emit("leave", { room: currentRoom });
            socket.disconnect();
            socket = null;
        }

        closeAllPeers();

        if (localStream) {
            localStream.getTracks().forEach((track) => track.stop());
            localStream = null;
        }
        if (screenStream) {
            screenStream.getTracks().forEach((track) => track.stop());
            screenStream = null;
        }
        isScreenSharing = false;

        if (videoGrid) videoGrid.innerHTML = "";
        currentRoom = null;
        mySid = null;
        isJoining = false;

        callPanel.hidden = true;
        joinPanel.hidden = false;
        if (joinRoomBtn) joinRoomBtn.disabled = false;
        showCallError("");
    }

    if (joinRoomBtn) {
        joinRoomBtn.addEventListener("click", async () => {
            if (joinRoomBtn.disabled) return; // already joining, ignore extra clicks

            const room = (roomIdInput.value || "").trim() || randomRoomId();
            if (!/^[A-Za-z0-9-]{1,32}$/.test(room)) {
                showJoinError("Room ID can only contain letters, numbers, and hyphens (max 32 characters).");
                return;
            }
            roomIdInput.value = room;

            joinRoomBtn.disabled = true;
            await joinRoom(room);
            // If joinRoom() failed (no active call started), re-enable the
            // button so the user can retry. On success the join panel is
            // hidden anyway, so this is a no-op in that case.
            if (!currentRoom) {
                joinRoomBtn.disabled = false;
            }
        });
    }

    if (leaveRoomBtn) leaveRoomBtn.addEventListener("click", leaveRoom);
    if (endCallBtn) endCallBtn.addEventListener("click", leaveRoom);

    if (copyRoomBtn) {
        copyRoomBtn.addEventListener("click", async () => {
            if (!currentRoom) return;
            try {
                await navigator.clipboard.writeText(currentRoom);
                const original = copyRoomBtn.textContent;
                copyRoomBtn.textContent = "✓ Copied";
                setTimeout(() => { copyRoomBtn.textContent = original; }, 1500);
            } catch (err) {
                showCallError("Could not copy the room ID automatically -- copy it manually: " + currentRoom);
            }
        });
    }

    // --- Controls: mute / camera / screen share ---------------------------

    if (muteBtn) {
        muteBtn.addEventListener("click", () => {
            if (!localStream) return;
            const audioTracks = localStream.getAudioTracks();
            const nowEnabled = !(audioTracks[0] && audioTracks[0].enabled);
            audioTracks.forEach((t) => (t.enabled = nowEnabled));
            muteBtn.classList.toggle("is-off", !nowEnabled);
        });
    }

    if (cameraToggleBtn) {
        cameraToggleBtn.addEventListener("click", () => {
            if (!localStream) return;
            const videoTracks = localStream.getVideoTracks();
            const nowEnabled = !(videoTracks[0] && videoTracks[0].enabled);
            videoTracks.forEach((t) => (t.enabled = nowEnabled));
            cameraToggleBtn.classList.toggle("is-off", !nowEnabled);

            const localPeer = peers["local"];
            if (localPeer) {
                setTileCameraOff(localPeer.video, localPeer.placeholder, !nowEnabled);
            }
        });
    }

    if (screenShareBtn) {
        screenShareBtn.addEventListener("click", async () => {
            if (!localStream) return;

            try {
                if (!isScreenSharing) {
                    if (typeof navigator.mediaDevices?.getDisplayMedia !== "function") {
                        showCallError(
                            "Screen sharing is not supported in this browser or environment. " +
                            "Please use the latest Chrome or Edge."
                        );
                        return;
                    }

                    screenStream = await navigator.mediaDevices.getDisplayMedia({ video: true });
                    const screenTrack = screenStream.getVideoTracks()[0];

                    // Swap the outgoing video track on every existing
                    // peer connection so remote participants see the
                    // shared screen instead of the camera.
                    Object.values(peers).forEach((peer) => {
                        if (peer.pc) {
                            const sender = peer.pc.getSenders().find((s) => s.track && s.track.kind === "video");
                            if (sender) sender.replaceTrack(screenTrack);
                        }
                    });

                    // Also preview it locally.
                    const localPeer = peers["local"];
                    if (localPeer) localPeer.video.srcObject = screenStream;

                    screenTrack.onended = () => stopScreenShare();

                    isScreenSharing = true;
                    screenShareBtn.classList.add("is-off");
                } else {
                    stopScreenShare();
                }
            } catch (err) {
                console.warn("Screen share error:", err);
                showCallError("Could not start screen sharing: " + err.message);
            }
        });
    }

    function stopScreenShare() {
        if (!isScreenSharing) return;

        const cameraTrack = localStream ? localStream.getVideoTracks()[0] : null;
        Object.values(peers).forEach((peer) => {
            if (peer.pc && cameraTrack) {
                const sender = peer.pc.getSenders().find((s) => s.track && s.track.kind === "video");
                if (sender) sender.replaceTrack(cameraTrack);
            }
        });

        const localPeer = peers["local"];
        if (localPeer && localStream) localPeer.video.srcObject = localStream;

        if (screenStream) {
            screenStream.getTracks().forEach((t) => t.stop());
            screenStream = null;
        }
        isScreenSharing = false;
        screenShareBtn.classList.remove("is-off");
    }

    // Best-effort cleanup if the tab is closed without clicking Leave.
    // The server's own disconnect handler (app/sockets.py) also
    // notifies other participants in this case, so this is a
    // convenience for the leaving user's own resources, not a
    // requirement for correctness.
    window.addEventListener("beforeunload", () => {
        if (localStream) localStream.getTracks().forEach((track) => track.stop());
        if (screenStream) screenStream.getTracks().forEach((track) => track.stop());
    });
})();