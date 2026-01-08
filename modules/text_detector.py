from PIL import Image
from surya.detection import DetectionPredictor
from typing import List, Dict, Any

class TextDetector:
    """Independent text detection module using Surya."""
    
    def __init__(self):
        self.predictor = DetectionPredictor()
    
    def detect(self, images: List[Image.Image]) -> List[Dict[str, Any]]:
        """
        Detect text bounding boxes in images.
        
        Args:
            images: List of PIL Images
            
        Returns:
            List of dictionaries with bboxes, polygons, and confidence scores
        """
        predictions = self.predictor(images)
        return predictions
    
    def detect_single(self, image: Image.Image) -> Dict[str, Any]:
        """Detect text in a single image."""
        return self.detect([image])[0]
