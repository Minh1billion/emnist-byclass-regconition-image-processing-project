from flask import Flask, render_template, request, jsonify
from pathlib import Path
from services.image_service import ImageService
from services.model_service import ModelService
from utils.validators import validate_file, validate_image_loaded
from config import Config

app = Flask(__name__)
app.config.from_object(Config)

# Initialize services
image_service = ImageService()
model_service = ModelService(Config.MODELS_DIR)


@app.route("/")
def index():
    """Render main page"""
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    """Upload and load image"""
    file = request.files.get('image')
    error = validate_file(file)
    if error:
        return jsonify({'error': error}), 400
    
    try:
        result = image_service.load_image(file)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route("/get-color", methods=["POST"])
@validate_image_loaded(image_service)
def get_color():
    """Get HSV range from pixel or region"""
    data = request.json
    x, y = int(data['x']), int(data['y'])
    w, h = data.get('w', 1), data.get('h', 1)
    
    try:
        hsv_range = image_service.get_hsv_range(x, y, w, h)
        return jsonify({'success': True, **hsv_range})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route("/process", methods=["POST"])
@validate_image_loaded(image_service)
def process():
    """Apply color filters to image"""
    data = request.json
    
    try:
        result = image_service.apply_filters(
            bg_hsv=data.get('bg_hsv'),
            text_hsv=data.get('text_hsv')
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route("/extract-characters", methods=["POST"])
@validate_image_loaded(image_service)
def extract_characters():
    """Extract characters from image"""
    data = request.json
    
    try:
        result = image_service.extract_characters(
            mode=data['mode'],
            bg_hsv=data.get('bg_hsv'),
            text_hsv=data.get('text_hsv'),
            threshold_value=data.get('threshold_value', 127),
            denoise=data.get('denoise', False),
            morph_size=data.get('morph_size', 0),
            min_area=data.get('min_area', 100),
            max_area=data.get('max_area', 10000)
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route("/load-model", methods=["POST"])
def load_model():
    """Load ML model"""
    file = request.files.get('model')
    if not file:
        return jsonify({'error': 'No model file'}), 400
    
    try:
        result = model_service.load_model(file)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route("/list-models", methods=["GET"])
def list_models():
    """List all loaded models"""
    try:
        result = model_service.list_models()
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route("/predict", methods=["POST"])
def predict():
    """Predict characters using loaded models"""
    data = request.json
    characters = data.get('characters', [])
    model_names = data.get('models', [])
    
    if not model_names or not characters:
        return jsonify({'error': 'Missing models or characters'}), 400
    
    try:
        result = model_service.predict_batch(characters, model_names)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route("/delete-model/<model_name>", methods=["DELETE"])
def delete_model(model_name):
    """Delete a loaded model"""
    try:
        model_service.delete_model(model_name)
        return jsonify({'success': True})
    except KeyError:
        return jsonify({'error': 'Model not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == "__main__":
    app.run(
        debug=Config.DEBUG,
        host=Config.HOST,
        port=Config.PORT
    )