from PIL import Image
from surya.foundation import FoundationPredictor
from surya.recognition import RecognitionPredictor
from typing import List, Dict, Any, Optional

class TextRecognizer:
    """Independent text recognition (OCR) module using Surya."""
    
    def __init__(self):
        self.foundation_predictor = FoundationPredictor()
        self.predictor = RecognitionPredictor(self.foundation_predictor)
    
    def recognize(
        self, 
        images: List[Image.Image],
        detection_predictor: Optional[Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Recognize text in images.
        
        Args:
            images: List of PIL Images
            detection_predictor: Optional external detection predictor
            
        Returns:
            List of dictionaries with text lines, bboxes, and confidence scores
        """
        if not detection_predictor:
            from surya.detection import DetectionPredictor
            detection_predictor = DetectionPredictor()
        predictions = self.predictor(images, det_predictor=detection_predictor)
        return predictions
    
    def recognize_single(self, image: Image.Image) -> Dict[str, Any]:
        """Recognize text in a single image."""
        return self.recognize([image])[0]
