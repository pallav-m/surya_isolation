from PIL import Image
from surya.foundation import FoundationPredictor
from surya.layout import LayoutPredictor
from surya.settings import settings
from typing import List, Dict, Any

class LayoutAnalyzer:
    """Independent layout analysis module using Surya."""
    
    def __init__(self):
        self.foundation_predictor = FoundationPredictor(
            checkpoint=settings.LAYOUT_MODEL_CHECKPOINT
        )
        self.predictor = LayoutPredictor(self.foundation_predictor)
        self.predictor.batch_size = 16
    
    def analyze(self, images: List[Image.Image]) -> List[Dict[str, Any]]:
        """
        Analyze layout of images (detect tables, figures, headers, etc.).
        
        Args:
            images: List of PIL Images
            
        Returns:
            List of dictionaries with layout bboxes, labels, and reading order
        """
        predictions = self.predictor(images)
        return predictions
    
    def analyze_single(self, image: Image.Image) -> Dict[str, Any]:
        """Analyze layout of a single image."""
        return self.analyze([image])[0]
