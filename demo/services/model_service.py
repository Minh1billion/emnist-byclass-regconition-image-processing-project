import os
import numpy as np
from tensorflow import keras
from pathlib import Path

class ModelService:
    """Service for ML model management"""
    
    def __init__(self, models_dir):
        self.models_dir = Path(models_dir)
        self.loaded_models = {}
    
    def load_model(self, file):
        """Load a Keras model from file"""
        if not file.filename or not (file.filename.endswith('.keras') or file.filename.endswith('.h5')):
            raise ValueError('Only .keras or .h5 files are supported')
        
        model_name = os.path.splitext(file.filename)[0]
        model_path = self.models_dir / file.filename
        
        # Save and load model
        file.save(str(model_path))
        
        try:
            model = keras.models.load_model(str(model_path))
            self.loaded_models[model_name] = model
            
            return {
                'success': True,
                'model_name': model_name,
                'input_shape': str(model.input_shape),
                'output_shape': str(model.output_shape),
                'num_classes': int(model.output_shape[-1]) if model.output_shape else 0
            }
        except Exception as e:
            # Clean up on failure
            if model_path.exists():
                model_path.unlink()
            raise
    
    def list_models(self):
        """List all loaded models"""
        models_info = [
            {
                'name': name,
                'input_shape': str(model.input_shape),
                'output_shape': str(model.output_shape)
            }
            for name, model in self.loaded_models.items()
        ]
        return {
            'success': True, 
            'models': models_info, 
            'tf_available': True
        }
    
    def predict_character(self, emnist_array, model):
        """Predict single character"""
        predictions = model.predict(emnist_array, verbose=0)[0]
        top_idx = np.argmax(predictions)
        top3_indices = np.argsort(predictions)[-3:][::-1]
        
        return {
            'top_class': int(top_idx),
            'top_confidence': float(predictions[top_idx]),
            'top3': [
                {'class': int(idx), 'confidence': float(predictions[idx])} 
                for idx in top3_indices
            ]
        }
    
    def predict_batch(self, characters, model_names):
        """Predict batch of characters with multiple models"""
        results = []
        
        for char in characters:
            char_result = {'emnist': char['emnist'], 'predictions': {}}
            
            try:
                emnist_array = np.array(char['emnist_array'], dtype=np.float32).reshape(1, 28, 28, 1)
            except Exception as e:
                char_result['error'] = f'Invalid input: {str(e)}'
                results.append(char_result)
                continue
            
            for model_name in model_names:
                if model_name not in self.loaded_models:
                    char_result['predictions'][model_name] = {'error': 'Model not found'}
                    continue
                
                try:
                    model = self.loaded_models[model_name]
                    char_result['predictions'][model_name] = self.predict_character(emnist_array, model)
                except Exception as e:
                    char_result['predictions'][model_name] = {'error': str(e)}
            
            results.append(char_result)
        
        return {'success': True, 'results': results}
    
    def delete_model(self, model_name):
        """Delete a loaded model"""
        if model_name not in self.loaded_models:
            raise KeyError(f'Model {model_name} not found')
        
        del self.loaded_models[model_name]
        
        # Try to delete file
        model_path = self.models_dir / f"{model_name}.keras"
        if model_path.exists():
            try:
                model_path.unlink()
            except Exception:
                pass