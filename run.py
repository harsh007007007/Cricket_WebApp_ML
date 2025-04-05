from app import app
import os

# Auto-create folders if needed
base_dir = os.path.abspath(os.path.dirname(__file__))
os.makedirs(os.path.join(base_dir, 'models'), exist_ok=True)
os.makedirs(os.path.join(base_dir, 'app', 'static', 'viz'), exist_ok=True)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
