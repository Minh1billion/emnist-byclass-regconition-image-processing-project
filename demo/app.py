from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import os
import time
import base64

app = Flask(__name__)
UPLOAD = "static/uploads"
MODELS = "static/models"
os.makedirs(UPLOAD, exist_ok=True)
os.makedirs(MODELS, exist_ok=True)

current_image = None
loaded_models = {}  # {'model_name': model_object}

# Try to import tensorflow, but continue if not available
try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("⚠️  TensorFlow not available. Model prediction features disabled.")
    print("   Install with: pip install tensorflow")

def auto_load_models():
    """Automatically load all .keras files from static/models folder on startup"""
    global loaded_models
    
    if not TF_AVAILABLE:
        return
    
    model_files = [f for f in os.listdir(MODELS) if f.endswith('.keras') or f.endswith('.h5')]
    
    for model_file in model_files:
        model_path = os.path.join(MODELS, model_file)
        model_name = os.path.splitext(model_file)[0]
        
        try:
            model = keras.models.load_model(model_path)
            loaded_models[model_name] = model
            print(f"✅ Loaded model: {model_name}")
        except Exception as e:
            print(f"⚠️  Failed to load {model_file}: {str(e)}")

def rgb_to_hsv_range(rgb, tolerance=25):
    """Convert RGB to HSV range"""
    r, g, b = rgb
    pixel = np.uint8([[[b, g, r]]])
    hsv_pixel = cv2.cvtColor(pixel, cv2.COLOR_BGR2HSV)[0][0]
    h, s, v = int(hsv_pixel[0]), int(hsv_pixel[1]), int(hsv_pixel[2])
    
    return {
        'h_min': int(max(0, h - tolerance)),
        'h_max': int(min(179, h + tolerance)),
        's_min': int(max(0, s - 40)),
        's_max': int(min(255, s + 40)),
        'v_min': int(max(0, v - 40)),
        'v_max': int(min(255, v + 40))
    }

def region_to_hsv_range(image, x, y, w, h):
    """Calculate HSV range from region"""
    region = image[y:y+h, x:x+w]
    hsv_region = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    
    # Calculate mean and std of the entire region
    h_mean = np.mean(hsv_region[:,:,0])
    s_mean = np.mean(hsv_region[:,:,1])
    v_mean = np.mean(hsv_region[:,:,2])
    h_std = np.std(hsv_region[:,:,0])
    s_std = np.std(hsv_region[:,:,1])
    v_std = np.std(hsv_region[:,:,2])
    
    # Avoid zero std
    h_std = max(1, h_std)
    s_std = max(1, s_std)
    v_std = max(1, v_std)
    
    return {
        'h_min': max(0, int(h_mean - h_std * 1.5)),
        'h_max': min(179, int(h_mean + h_std * 1.5)),
        's_min': max(0, int(s_mean - s_std * 1.5)),
        's_max': min(255, int(s_mean + s_std * 1.5)),
        'v_min': max(0, int(v_mean - v_std * 1.5)),
        'v_max': min(255, int(v_mean + v_std * 1.5))
    }

def image_to_base64(image):
    """Convert OpenCV image to base64"""
    _, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer).decode('utf-8')

def roi_to_emnist(roi):
    """Convert ROI to EMNIST format (28x28)"""
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    coords = cv2.findNonZero(binary)
    if coords is None:
        return np.zeros((28, 28), dtype=np.float32)
    
    x, y, w, h = cv2.boundingRect(coords)
    digit = binary[y:y+h, x:x+w]
    
    scale = 20.0 / max(h, w)
    digit = cv2.resize(digit, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    
    padded = np.zeros((28, 28), dtype=np.uint8)
    y_offset = (28 - digit.shape[0]) // 2
    x_offset = (28 - digit.shape[1]) // 2
    padded[y_offset:y_offset+digit.shape[0], x_offset:x_offset+digit.shape[1]] = digit
    
    return padded.astype(np.float32) / 255.0

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    global current_image
    
    if 'image' not in request.files:
        return jsonify({'error': 'No file'}), 400
    
    file = request.files["image"]
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400
    
    # Read image directly
    file_bytes = np.frombuffer(file.read(), np.uint8)
    current_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    img_base64 = image_to_base64(current_image)
    
    return jsonify({
        'success': True,
        'image': f'data:image/jpeg;base64,{img_base64}',
        'width': current_image.shape[1],
        'height': current_image.shape[0]
    })

@app.route("/get-color", methods=["POST"])
def get_color():
    """Get color from pixel or region"""
    global current_image
    
    if current_image is None:
        return jsonify({'error': 'No image'}), 400
    
    data = request.json
    x, y = int(data['x']), int(data['y'])
    w, h = data.get('w', 1), data.get('h', 1)
    
    if w > 1 and h > 1:
        hsv_range = region_to_hsv_range(current_image, x, y, w, h)
    else:
        b, g, r = current_image[y, x]
        hsv_range = rgb_to_hsv_range([int(r), int(g), int(b)])
    
    return jsonify({'success': True, **hsv_range})

@app.route("/process", methods=["POST"])
def process():
    """Process image: remove background, keep text"""
    global current_image
    
    if current_image is None:
        return jsonify({'error': 'No image'}), 400
    
    data = request.json
    text_hsv = data.get('text_hsv')
    bg_hsv = data.get('bg_hsv')
    
    result = current_image.copy()
    
    # If both text and bg colors are selected, prioritize text_hsv (keep text, remove non-text)
    if text_hsv and bg_hsv:
        # Keep only text color - remove everything except text
        hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
        lower = np.array([text_hsv['h_min'], text_hsv['s_min'], text_hsv['v_min']], dtype=np.uint8)
        upper = np.array([text_hsv['h_max'], text_hsv['s_max'], text_hsv['v_max']], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
        result = cv2.bitwise_and(result, result, mask=mask)
    elif bg_hsv:
        # Remove background only if text is not specified
        hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
        lower = np.array([bg_hsv['h_min'], bg_hsv['s_min'], bg_hsv['v_min']], dtype=np.uint8)
        upper = np.array([bg_hsv['h_max'], bg_hsv['s_max'], bg_hsv['v_max']], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
        mask_inv = cv2.bitwise_not(mask)
        result = cv2.bitwise_and(result, result, mask=mask_inv)
    elif text_hsv:
        # Keep only text color if specified
        hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
        lower = np.array([text_hsv['h_min'], text_hsv['s_min'], text_hsv['v_min']], dtype=np.uint8)
        upper = np.array([text_hsv['h_max'], text_hsv['s_max'], text_hsv['v_max']], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
        result = cv2.bitwise_and(result, result, mask=mask)
    
    img_base64 = image_to_base64(result)
    
    return jsonify({
        'success': True,
        'image': f'data:image/jpeg;base64,{img_base64}'
    })

@app.route("/extract-characters", methods=["POST"])
def extract_characters():
    """Extract and convert characters to EMNIST"""
    global current_image
    
    if current_image is None:
        return jsonify({'error': 'No image'}), 400
    
    data = request.json
    text_hsv = data.get('text_hsv')
    bg_hsv = data.get('bg_hsv')
    mode = data.get('mode', 'color')
    threshold_value = data.get('threshold_value', 127)
    min_area = data.get('min_area', 100)
    max_area = data.get('max_area', 10000)
    denoise = data.get('denoise', False)
    morph_size = data.get('morph_size', 0)
    
    result = current_image.copy()
    
    # Mode-based processing
    if mode == 'color':
        if bg_hsv:
            hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
            lower = np.array([bg_hsv['h_min'], bg_hsv['s_min'], bg_hsv['v_min']], dtype=np.uint8)
            upper = np.array([bg_hsv['h_max'], bg_hsv['s_max'], bg_hsv['v_max']], dtype=np.uint8)
            mask = cv2.inRange(hsv, lower, upper)
            mask_inv = cv2.bitwise_not(mask)
            result = cv2.bitwise_and(result, result, mask=mask_inv)
        
        if text_hsv:
            hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
            lower = np.array([text_hsv['h_min'], text_hsv['s_min'], text_hsv['v_min']], dtype=np.uint8)
            upper = np.array([text_hsv['h_max'], text_hsv['s_max'], text_hsv['v_max']], dtype=np.uint8)
            mask = cv2.inRange(hsv, lower, upper)
            result = cv2.bitwise_and(result, result, mask=mask)
        
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
    elif mode == 'auto':
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        if denoise:
            gray = cv2.fastNlMeansDenoising(gray, None, h=10)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
    elif mode == 'adaptive':
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        if denoise:
            gray = cv2.fastNlMeansDenoising(gray, None, h=10)
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 2)
        
    else:  # manual
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        if denoise:
            gray = cv2.fastNlMeansDenoising(gray, None, h=10)
        _, binary = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY_INV)
    
    # Morphology operations
    if morph_size > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_size, morph_size))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    chars = []
    display_img = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area < area < max_area:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(display_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            roi = result[y:y+h, x:x+w]
            emnist_img = roi_to_emnist(roi)
            emnist_preview = (emnist_img * 255).astype(np.uint8)
            
            _, buffer = cv2.imencode('.png', emnist_preview)
            emnist_b64 = base64.b64encode(buffer).decode()
            
            chars.append({
                'x': int(x),
                'y': int(y),
                'w': int(w),
                'h': int(h),
                'emnist': f'data:image/png;base64,{emnist_b64}',
                'emnist_array': emnist_img.flatten().tolist()  # For prediction
            })
    
    chars.sort(key=lambda c: c['x'])
    display_base64 = image_to_base64(display_img)
    
    return jsonify({
        'success': True,
        'image': f'data:image/jpeg;base64,{display_base64}',
        'characters': chars
    })

@app.route("/load-model-from-folder", methods=["POST"])
def load_model_from_folder():
    """Load a model file from static/models folder"""
    global loaded_models
    
    if not TF_AVAILABLE:
        return jsonify({'error': 'TensorFlow not installed'}), 400
    
    data = request.json
    filename = data.get('filename', '')
    
    if not filename:
        return jsonify({'error': 'No filename provided'}), 400
    
    if not (filename.endswith('.keras') or filename.endswith('.h5')):
        return jsonify({'error': 'Only .keras or .h5 files are supported'}), 400
    
    model_path = os.path.join(MODELS, filename)
    
    # Security check - prevent directory traversal
    if not os.path.abspath(model_path).startswith(os.path.abspath(MODELS)):
        return jsonify({'error': 'Invalid file path'}), 400
    
    if not os.path.exists(model_path):
        return jsonify({'error': 'File not found'}), 404
    
    model_name = os.path.splitext(filename)[0]
    
    try:
        model = keras.models.load_model(model_path)
        loaded_models[model_name] = model
        
        # Get model info
        input_shape = model.input_shape
        output_shape = model.output_shape
        
        return jsonify({
            'success': True,
            'model_name': model_name,
            'input_shape': str(input_shape),
            'output_shape': str(output_shape),
            'num_classes': int(output_shape[-1]) if output_shape else 0
        })
    except Exception as e:
        return jsonify({'error': f'Failed to load model: {str(e)}'}), 400

@app.route("/load-model", methods=["POST"])
def load_model():
    """Load a Keras model"""
    global loaded_models
    
    if not TF_AVAILABLE:
        return jsonify({'error': 'TensorFlow not installed'}), 400
    
    if 'model' not in request.files:
        return jsonify({'error': 'No model file'}), 400
    
    file = request.files['model']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400
    
    if not file.filename.endswith('.keras') and not file.filename.endswith('.h5'):
        return jsonify({'error': 'Only .keras or .h5 files are supported'}), 400
    
    model_name = os.path.splitext(file.filename)[0]
    model_path = os.path.join(MODELS, file.filename)
    
    try:
        # Save file first
        file.save(model_path)
        
        # Load model
        model = keras.models.load_model(model_path)
        loaded_models[model_name] = model
        
        # Get model info
        input_shape = model.input_shape
        output_shape = model.output_shape
        
        return jsonify({
            'success': True,
            'model_name': model_name,
            'input_shape': str(input_shape),
            'output_shape': str(output_shape),
            'num_classes': int(output_shape[-1]) if output_shape else 0
        })
    except Exception as e:
        # Clean up file if load failed
        if os.path.exists(model_path):
            try:
                os.remove(model_path)
            except:
                pass
        return jsonify({'error': f'Failed to load model: {str(e)}'}), 400

@app.route("/list-models", methods=["GET"])
def list_models():
    """List all loaded models and available models in folder"""
    if not TF_AVAILABLE:
        return jsonify({'success': True, 'models': [], 'available_models': [], 'tf_available': False})
    
    # Get loaded models
    models_info = []
    for name, model in loaded_models.items():
        models_info.append({
            'name': name,
            'input_shape': str(model.input_shape),
            'output_shape': str(model.output_shape),
            'loaded': True
        })
    
    # Get available models in folder (not yet loaded)
    available_models = []
    try:
        model_files = [f for f in os.listdir(MODELS) if f.endswith('.keras') or f.endswith('.h5')]
        for model_file in model_files:
            model_name = os.path.splitext(model_file)[0]
            # Only add if not already loaded
            if model_name not in loaded_models:
                available_models.append({
                    'name': model_name,
                    'filename': model_file
                })
    except Exception as e:
        print(f"Error listing available models: {str(e)}")
    
    return jsonify({
        'success': True, 
        'models': models_info, 
        'available_models': available_models,
        'tf_available': True
    })

@app.route("/predict", methods=["POST"])
def predict():
    """Predict characters using loaded models"""
    global loaded_models
    
    if not TF_AVAILABLE:
        return jsonify({'error': 'TensorFlow not installed'}), 400
    
    data = request.json
    characters = data.get('characters', [])
    model_names = data.get('models', [])
    
    if not model_names:
        return jsonify({'error': 'No models specified'}), 400
    
    if not characters:
        return jsonify({'error': 'No characters provided'}), 400
    
    results = []
    
    for char in characters:
        char_result = {
            'emnist': char['emnist'],
            'predictions': {}
        }
        
        # Prepare input
        try:
            emnist_array = np.array(char['emnist_array'], dtype=np.float32).reshape(1, 28, 28, 1)
        except Exception as e:
            char_result['error'] = f'Invalid input: {str(e)}'
            results.append(char_result)
            continue
        
        # Collect all predictions for voting
        all_predictions = {}
        
        # Predict with each model
        for model_name in model_names:
            if model_name not in loaded_models:
                char_result['predictions'][model_name] = {
                    'error': 'Model not found'
                }
                continue
            
            try:
                model = loaded_models[model_name]
                predictions = model.predict(emnist_array, verbose=0)[0]
                
                # Get top prediction
                top_idx = np.argmax(predictions)
                top_conf = float(predictions[top_idx])
                
                # Get top 3 predictions
                top3_indices = np.argsort(predictions)[-3:][::-1]
                top3 = [
                    {
                        'class': int(idx),
                        'confidence': float(predictions[idx])
                    }
                    for idx in top3_indices
                ]
                
                char_result['predictions'][model_name] = {
                    'top_class': int(top_idx),
                    'top_confidence': top_conf,
                    'top3': top3
                }
                
                # Store for voting
                all_predictions[model_name] = int(top_idx)
            except Exception as e:
                char_result['predictions'][model_name] = {
                    'error': str(e)
                }
        
        # Calculate ensemble conclusion (voting)
        if all_predictions:
            from collections import Counter
            votes = Counter(all_predictions.values())
            consensus_class = votes.most_common(1)[0][0]
            consensus_confidence = votes.most_common(1)[0][1] / len(all_predictions)
            char_result['consensus'] = {
                'class': consensus_class,
                'confidence': consensus_confidence,
                'votes': len(all_predictions)
            }
        
        results.append(char_result)
    
    return jsonify({
        'success': True,
        'results': results
    })

@app.route("/delete-model/<model_name>", methods=["DELETE"])
def delete_model(model_name):
    """Delete a loaded model"""
    global loaded_models
    
    if model_name in loaded_models:
        del loaded_models[model_name]
        
        # Try to delete file
        model_path = os.path.join(MODELS, f"{model_name}.keras")
        if os.path.exists(model_path):
            try:
                os.remove(model_path)
            except:
                pass
        
        return jsonify({'success': True})
    
    return jsonify({'error': 'Model not found'}), 404

if __name__ == "__main__":
    # Auto-load models from static/models folder on startup
    print("🚀 Loading models from folder...")
    auto_load_models()
    print(f"✅ Ready with {len(loaded_models)} loaded model(s)")
    
    app.run(debug=True, host='0.0.0.0', port=5000)