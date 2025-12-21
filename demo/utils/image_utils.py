import cv2
import numpy as np
import base64

class ImageUtils:
    """Utility functions for image processing"""
    
    @staticmethod
    def to_base64(image):
        """Convert image to base64 string"""
        _, buffer = cv2.imencode('.jpg', image)
        return base64.b64encode(buffer).decode('utf-8')
    
    @staticmethod
    def from_file(file):
        """Load image from uploaded file"""
        file_bytes = np.frombuffer(file.read(), np.uint8)
        return cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    @staticmethod
    def rgb_to_hsv_range(rgb, tolerance=25):
        """Convert RGB color to HSV range"""
        r, g, b = rgb
        pixel = np.uint8([[[b, g, r]]])
        hsv_pixel = cv2.cvtColor(pixel, cv2.COLOR_BGR2HSV)[0][0]
        h, s, v = map(int, hsv_pixel)
        
        return {
            'h_min': max(0, h - tolerance),
            'h_max': min(179, h + tolerance),
            's_min': max(0, s - 40),
            's_max': min(255, s + 40),
            'v_min': max(0, v - 40),
            'v_max': min(255, v + 40)
        }
    
    @staticmethod
    def region_to_hsv_range(image, x, y, w, h):
        """Calculate HSV range from image region"""
        region = image[y:y+h, x:x+w]
        hsv_region = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        
        means = np.mean(hsv_region, axis=(0, 1))
        stds = np.std(hsv_region, axis=(0, 1))
        
        return {
            'h_min': max(0, int(means[0] - stds[0] * 2)),
            'h_max': min(179, int(means[0] + stds[0] * 2)),
            's_min': max(0, int(means[1] - stds[1] * 2)),
            's_max': min(255, int(means[1] + stds[1] * 2)),
            'v_min': max(0, int(means[2] - stds[2] * 2)),
            'v_max': min(255, int(means[2] + stds[2] * 2))
        }