from flask import Flask

from app.extensions import socketio


def create_app():
    app = Flask(__name__)

    socketio.init_app(app)

    from app.routes.main import main

    app.register_blueprint(main)

    return app