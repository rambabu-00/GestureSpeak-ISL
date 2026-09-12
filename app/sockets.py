"""
sockets.py

Flask-SocketIO signaling handlers for the GestureSpeak AI group video
call feature.

This module ONLY relays small signaling messages between browsers
(room membership, WebRTC offers/answers/ICE candidates). No audio or
video ever passes through this server or through Flask -- once two
peers have exchanged an offer/answer/candidates, their media flows
directly between their browsers via WebRTC (peer-to-peer mesh).

No authentication, no database, no persistence: room membership is
tracked purely in memory for the lifetime of the process, which is
sufficient for local development/testing as specified for this stage.

Importing this module registers its event handlers on the shared
`socketio` instance (see app/extensions.py); app/__init__.py imports
it for that side effect inside create_app().
"""

import re

from flask import request
from flask_socketio import emit, join_room, leave_room

from app.extensions import socketio

# Room IDs are restricted to a small safe character set and length so
# they can't be used to inject anything odd into server-side room
# bookkeeping. This is basic input validation, not authentication.
ROOM_ID_PATTERN = re.compile(r"^[A-Za-z0-9-]{1,32}$")

# sid -> room name, so we can look up "which room was this socket in"
# on disconnect (the browser doesn't get a chance to send a tidy
# "leave" event first if the tab is just closed).
_socket_rooms = {}


def _is_valid_room_id(room):
    return isinstance(room, str) and bool(ROOM_ID_PATTERN.match(room))


@socketio.on("join")
def handle_join(data):
    """
    A client wants to join (or create, implicitly) a room.

    data: { "room": "<room id>" }
    """
    room = (data or {}).get("room")
    if not _is_valid_room_id(room):
        emit("join-error", {"message": "Invalid room ID."})
        return

    sid = request.sid
    join_room(room)
    _socket_rooms[sid] = room

    # Tell everyone else already in the room that a new peer has
    # arrived, so they can initiate a WebRTC offer to it.
    emit("user-joined", {"sid": sid}, to=room, include_self=False)

    # Confirm to the joining client which room it's now in (and let
    # it know its own socket id, useful for the frontend's own
    # bookkeeping).
    emit("joined", {"room": room, "sid": sid})


@socketio.on("offer")
def handle_offer(data):
    """
    Relay a WebRTC SDP offer from one peer to a specific target peer
    in the same room.

    data: { "room": "<room id>", "to": "<target sid>", "sdp": {...} }
    """
    room = (data or {}).get("room")
    target = (data or {}).get("to")
    sdp = (data or {}).get("sdp")

    if not _is_valid_room_id(room) or not target or sdp is None:
        return

    emit("offer", {"from": request.sid, "sdp": sdp}, to=target)


@socketio.on("answer")
def handle_answer(data):
    """
    Relay a WebRTC SDP answer from one peer to a specific target peer.

    data: { "room": "<room id>", "to": "<target sid>", "sdp": {...} }
    """
    room = (data or {}).get("room")
    target = (data or {}).get("to")
    sdp = (data or {}).get("sdp")

    if not _is_valid_room_id(room) or not target or sdp is None:
        return

    emit("answer", {"from": request.sid, "sdp": sdp}, to=target)


@socketio.on("ice-candidate")
def handle_ice_candidate(data):
    """
    Relay a single ICE candidate from one peer to a specific target
    peer.

    data: { "room": "<room id>", "to": "<target sid>", "candidate": {...} }
    """
    room = (data or {}).get("room")
    target = (data or {}).get("to")
    candidate = (data or {}).get("candidate")

    if not _is_valid_room_id(room) or not target or candidate is None:
        return

    emit("ice-candidate", {"from": request.sid, "candidate": candidate}, to=target)


@socketio.on("leave")
def handle_leave(data):
    """
    A client is deliberately leaving a room (e.g. clicked "Leave
    Call"), as opposed to just disconnecting.

    data: { "room": "<room id>" }
    """
    room = (data or {}).get("room")
    sid = request.sid

    if not _is_valid_room_id(room):
        return

    leave_room(room)
    _socket_rooms.pop(sid, None)

    emit("user-left", {"sid": sid}, to=room, include_self=False)


@socketio.on("disconnect")
def handle_disconnect():
    """
    Fallback cleanup for clients that vanish without sending "leave"
    first (closed tab, network drop, etc.) so other participants
    still get a "user-left" and can tear down that peer connection.
    """
    sid = request.sid
    room = _socket_rooms.pop(sid, None)

    if room:
        emit("user-left", {"sid": sid}, to=room, include_self=False)