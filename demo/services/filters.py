import cv2
import numpy as np

class ColorFilter:
    """Color filtering operations"""
    
    @staticmethod
    def apply(image, hsv_range, keep=True):
        """Apply HSV color filter to image"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower = np.array([hsv_range['h_min'], hsv_range['s_min'], hsv_range['v_min']], dtype=np.uint8)
        upper = np.array([hsv_range['h_max'], hsv_range['s_max'], hsv_range['v_max']], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
        
        if not keep:
            mask = cv2.bitwise_not(mask)
        
        return cv2.bitwise_and(image, image, mask=mask)


class BinaryFilter:
    """Binary threshold operations"""
    
    @staticmethod
    def create_mask(image, mode, threshold_value=127, denoise=False):
        """Create binary mask from image"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        if denoise:
            gray = cv2.fastNlMeansDenoising(gray, None, h=10)
        
        if mode in ['auto', 'color']:
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        elif mode == 'adaptive':
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY_INV, 11, 2
            )
        else:
            _, binary = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY_INV)
        
        return binary