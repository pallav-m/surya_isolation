from modules.text_detector import TextDetector
from modules.text_recognizer import TextRecognizer
from modules.table_recognizer import TableRecognizer
from modules.layout_analyzer import LayoutAnalyzer
from utils.serialization import serialize_results
from utils.text_extraction import extract_text_from_rec_result

class SuryaInferenceEngine:
    def __init__(self):
        self.text_detector = TextDetector()
        self.text_recognizer = TextRecognizer()
        self.table_recognizer = TableRecognizer()
        self.layout_analyzer = LayoutAnalyzer()

    def detect_text(self, pil_image_list):
        text_detection_result = self.text_detector.detect(pil_image_list)

        # transform result into dict:
        text_det_dict = serialize_results(text_detection_result)
        return text_det_dict
    
    def recognize_text(self, pil_image_list):
        
        text_rec_result = self.text_recognizer.recognize(pil_image_list)

        # transform result into dict:
        text_rec_dict = serialize_results(text_rec_result)
        extract_text_from_rec_result(text_rec_dict)
        return text_rec_dict
    
    def extract_layout(self, pil_image_list):
        
        layout_result = self.layout_analyzer.analyze(pil_image_list)

        # transform result into dict:
        layout_res_dict = serialize_results(layout_result)
        return layout_res_dict
    
    def recognize_tables(self, pil_image_list):
        table_rec_result = self.table_recognizer.recognize_tables(pil_image_list)
        table_rec_dict = serialize_results(table_rec_result)

        return table_rec_dict
    