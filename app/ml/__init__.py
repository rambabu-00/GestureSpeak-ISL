from flask import Flask

from app.extensions import socketio


def create_app():
    app = Flask(__name__)

    socketio.init_app(app)

    from app.routes.main import main

    app.register_blueprint(main)

    # Import-for-side-effect: registers the video-call signaling
    # event handlers (join/offer/answer/ice-candidate/leave/disconnect)
    # on the shared `socketio` instance. Must happen after
    # socketio.init_app(app) above.
    from app import sockets  # noqa: F401

    return app