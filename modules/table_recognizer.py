from PIL import Image
from surya.table_rec import TableRecPredictor
from typing import List, Dict, Any

class TableRecognizer:
    """Independent table recognition module using Surya."""
    
    def __init__(self):
        self.predictor = TableRecPredictor()
    
    def recognize_tables(self, images: List[Image.Image]) -> List[Dict[str, Any]]:
        """
        Recognize table structures (rows, columns, cells).
        
        Args:
            images: List of PIL Images (should be cropped to tables)
            
        Returns:
            List of dictionaries with rows, columns, and cells
        """
        predictions = self.predictor(images)
        return predictions
    
    def recognize_single_table(self, image: Image.Image) -> Dict[str, Any]:
        """Recognize a single table structure."""
        return self.recognize_tables([image])[0]
