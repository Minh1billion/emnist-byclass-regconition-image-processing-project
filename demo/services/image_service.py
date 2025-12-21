from utils.image_utils import ImageUtils
from services.filters import ColorFilter, BinaryFilter
from services.character_extractor import CharacterExtractor
import cv2

class ImageService:
    """Service for image processing operations"""
    
    def __init__(self):
        self.current_image = None
    
    def load_image(self, file):
        """Load image from file"""
        self.current_image = ImageUtils.from_file(file)
        
        return {
            'success': True,
            'image': f'data:image/jpeg;base64,{ImageUtils.to_base64(self.current_image)}',
            'width': self.current_image.shape[1],
            'height': self.current_image.shape[0]
        }
    
    def get_hsv_range(self, x, y, w, h):
        """Get HSV range from pixel or region"""
        if w > 1 and h > 1:
            return ImageUtils.region_to_hsv_range(self.current_image, x, y, w, h)
        else:
            b, g, r = self.current_image[y, x]
            return ImageUtils.rgb_to_hsv_range([int(r), int(g), int(b)])
    
    def apply_filters(self, bg_hsv=None, text_hsv=None):
        """Apply color filters to image"""
        result = self.current_image.copy()
        
        if bg_hsv:
            result = ColorFilter.apply(result, bg_hsv, keep=False)
        
        if text_hsv:
            result = ColorFilter.apply(result, text_hsv, keep=True)
        
        return {
            'success': True,
            'image': f'data:image/jpeg;base64,{ImageUtils.to_base64(result)}'
        }
    
    def extract_characters(self, mode, bg_hsv=None, text_hsv=None, 
                          threshold_value=127, denoise=False, morph_size=0,
                          min_area=100, max_area=10000):
        """Extract characters from image"""
        result = self.current_image.copy()
        
        # Apply color filters if in color mode
        if mode == 'color':
            if bg_hsv:
                result = ColorFilter.apply(result, bg_hsv, keep=False)
            if text_hsv:
                result = ColorFilter.apply(result, text_hsv, keep=True)
        
        # Create binary mask
        binary = BinaryFilter.create_mask(result, mode, threshold_value, denoise)
        
        # Apply morphological operations
        if morph_size > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_size, morph_size))
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # Extract characters
        chars, display_img = CharacterExtractor.extract_all(
            binary, result, min_area, max_area
        )
        
        return {
            'success': True,
            'image': f'data:image/jpeg;base64,{ImageUtils.to_base64(display_img)}',
            'characters': chars
        }