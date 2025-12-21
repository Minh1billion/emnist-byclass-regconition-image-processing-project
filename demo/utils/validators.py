from functools import wraps
from flask import jsonify

def validate_file(file):
    """Validate uploaded file"""
    if not file or file.filename == '':
        return 'No file provided'
    return None

def validate_image_loaded(service):
    """Decorator to check if image is loaded"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if service.current_image is None:
                return jsonify({'error': 'No image loaded'}), 400
            return f(*args, **kwargs)
        return decorated_function
    return decorator
