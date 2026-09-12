from app import create_app
from app.extensions import socketio

app = create_app()

if __name__ == "__main__":
    # allow_unsafe_werkzeug=True is required here because Flask-SocketIO's
    # plain Werkzeug transport (no eventlet/gevent installed) now refuses
    # to start on this project's pinned Werkzeug version without an
    # explicit opt-in. This is fine for local development, which is all
    # this project currently targets -- it is not a production deployment
    # setting, and should not be used as-is if this app is ever deployed
    # publicly.
    socketio.run(app, debug=True, allow_unsafe_werkzeug=True)