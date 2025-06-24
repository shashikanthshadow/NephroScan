 # app/main.py
from flask import Flask
from .routes import bp


from flask import Flask
from .routes import bp as routes_bp
import os

def create_app():
    # Make sure Flask knows where the 'static' folder is
    app = Flask(__name__,
                static_folder=os.path.abspath("static"),
                static_url_path="/static")

    app.secret_key = "super-secret-key"
    app.register_blueprint(routes_bp)

    return app
