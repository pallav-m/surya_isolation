import os
from dataclasses import dataclass

@dataclass
class PipelineConfig:
    """Configuration for the inference pipeline."""
    
    # Performance settings
    detection_batch_size: int = int(os.getenv('DETECTOR_BATCH_SIZE', '36'))
    recognition_batch_size: int = int(os.getenv('RECOGNITION_BATCH_SIZE', '512'))
    layout_batch_size: int = int(os.getenv('LAYOUT_BATCH_SIZE', '32'))
    table_rec_batch_size: int = int(os.getenv('TABLE_REC_BATCH_SIZE', '64'))
    
    # Device settings
    torch_device: str = os.getenv('TORCH_DEVICE', 'auto')  # 'cuda', 'cpu', 'mps', 'auto'
    
    # Detection thresholds
    detector_blank_threshold: float = float(os.getenv('DETECTOR_BLANK_THRESHOLD', '0.35'))
    detector_text_threshold: float = float(os.getenv('DETECTOR_TEXT_THRESHOLD', '0.6'))
    
    # Enable/disable math detection
    disable_math: bool = os.getenv('DISABLE_MATH', 'false').lower() == 'true'
    
    # Compilation flags (for speed optimization)
    compile_detector: bool = os.getenv('COMPILE_DETECTOR', 'false').lower() == 'true'
    compile_layout: bool = os.getenv('COMPILE_LAYOUT', 'false').lower() == 'true'
    compile_table_rec: bool = os.getenv('COMPILE_TABLE_REC', 'false').lower() == 'true'
