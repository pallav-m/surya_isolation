from PIL import Image
from surya.texify import TexifyPredictor
from typing import List, Dict, Any

class LaTeXOCR:
    """Independent LaTeX OCR module using Surya."""
    
    def __init__(self):
        self.predictor = TexifyPredictor()
    
    def recognize_latex(self, images: List[Image.Image]) -> List[Dict[str, Any]]:
        """
        Recognize LaTeX from equation images.
        
        Args:
            images: List of PIL Images (should be cropped to equations)
            
        Returns:
            List of dictionaries with LaTeX strings
        """
        predictions = self.predictor(images)
        return predictions
    
    def recognize_single_equation(self, image: Image.Image) -> Dict[str, Any]:
        """Recognize LaTeX from a single equation image."""
        return self.recognize_latex([image])[0]
