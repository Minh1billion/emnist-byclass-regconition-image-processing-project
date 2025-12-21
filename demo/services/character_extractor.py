import cv2
import numpy as np
import base64

class CharacterExtractor:
    """Extract and preprocess characters from images"""
    
    @staticmethod
    def preprocess_roi(roi):
        """Preprocess ROI to EMNIST format (28x28)"""
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        coords = cv2.findNonZero(binary)
        if coords is None:
            return np.zeros((28, 28), dtype=np.float32)
        
        x, y, w, h = cv2.boundingRect(coords)
        digit = binary[y:y+h, x:x+w]
        
        # Scale to fit in 20x20 box
        scale = 20.0 / max(h, w)
        new_size = (int(w * scale), int(h * scale))
        digit = cv2.resize(digit, new_size, interpolation=cv2.INTER_AREA)
        
        # Center in 28x28 canvas
        padded = np.zeros((28, 28), dtype=np.uint8)
        y_offset = (28 - digit.shape[0]) // 2
        x_offset = (28 - digit.shape[1]) // 2
        padded[y_offset:y_offset+digit.shape[0], x_offset:x_offset+digit.shape[1]] = digit
        
        return padded.astype(np.float32) / 255.0
    
    @staticmethod
    def extract_from_contour(contour, image):
        """Extract character data from contour"""
        x, y, w, h = cv2.boundingRect(contour)
        roi = image[y:y+h, x:x+w]
        emnist_img = CharacterExtractor.preprocess_roi(roi)
        emnist_preview = (emnist_img * 255).astype(np.uint8)
        
        _, buffer = cv2.imencode('.png', emnist_preview)
        emnist_b64 = base64.b64encode(buffer).decode()
        
        return {
            'x': x, 'y': y, 'w': w, 'h': h,
            'emnist': f'data:image/png;base64,{emnist_b64}',
            'emnist_array': emnist_img.flatten().tolist()
        }
    
    @staticmethod
    def extract_all(binary, original_image, min_area=100, max_area=10000):
        """Extract all characters from binary image"""
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        chars = []
        display_img = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if min_area < area < max_area:
                char_data = CharacterExtractor.extract_from_contour(cnt, original_image)
                cv2.rectangle(
                    display_img, 
                    (char_data['x'], char_data['y']), 
                    (char_data['x'] + char_data['w'], char_data['y'] + char_data['h']), 
                    (0, 255, 0), 2
                )
                chars.append(char_data)
        
        # Sort by x position (left to right)
        chars.sort(key=lambda c: c['x'])
        
        return chars, display_img