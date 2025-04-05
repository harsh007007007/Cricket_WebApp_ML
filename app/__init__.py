from flask import Flask

# Initialize the Flask app
app = Flask(__name__, static_folder='static', template_folder='templates')

# Import routes at the end to avoid circular imports
from app import routes