"""
Inference script for Surya OCR models
Processes images from file paths or directories
"""

import os
import argparse
from pathlib import Path
from typing import List, Union
from PIL import Image
import json

from main import SuryaInferenceEngine


class SuryaInference:
    """Standalone inference class for Surya OCR tasks."""
    
    def __init__(self):
        """Initialize the Surya inference engine."""
        print("Loading Surya models...")
        self.engine = SuryaInferenceEngine()
        print("Models loaded successfully!")
        
    def load_images(self, image_paths: List[str]) -> List[Image.Image]:
        """
        Load images from file paths.
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            List of PIL Image objects
        """
        images = []
        for path in image_paths:
            try:
                img = Image.open(path)
                # Convert to RGB if necessary
                if img.mode not in ('RGB', 'L'):
                    img = img.convert('RGB')
                images.append(img)
                print(f"Loaded: {path}")
            except Exception as e:
                print(f"Error loading {path}: {str(e)}")
        
        return images
    
    def run_inference(self, images: List[Image.Image], task_type: str) -> List[dict]:
        """
        Run inference on images based on task type.
        
        Args:
            images: List of PIL Images
            task_type: One of 'extract_text', 'detect_text', 'detect_layout', 'process_tables'
            
        Returns:
            List of result dictionaries
        """
        task_functions = {
            "extract_text": self.engine.recognize_text,
            "detect_text": self.engine.detect_text,
            "detect_layout": self.engine.extract_layout,
            "process_tables": self.engine.recognize_tables
        }
        
        if task_type not in task_functions:
            raise ValueError(
                f"Invalid task type '{task_type}'. "
                f"Must be one of: {list(task_functions.keys())}"
            )
        
        print(f"\nRunning task: {task_type}")
        print(f"Processing {len(images)} image(s)...")
        
        results = task_functions[task_type](images)
        
        print(f"Processing completed!")
        return results
    
    def save_results(self, results: List[dict], output_path: str, format: str = "json"):
        """
        Save inference results to file.
        
        Args:
            results: List of result dictionaries
            output_path: Output file path
            format: Output format ('json' or 'txt')
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == "json":
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"\nResults saved to: {output_path}")
            
        elif format == "txt":
            with open(output_path, 'w', encoding='utf-8') as f:
                for idx, result in enumerate(results):
                    f.write(f"=== Image {idx + 1} ===\n")
                    f.write(json.dumps(result, indent=2, ensure_ascii=False))
                    f.write("\n\n")
            print(f"\nResults saved to: {output_path}")
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def process_directory(self, 
                         input_dir: str, 
                         task_type: str,
                         output_path: str = None,
                         file_extensions: tuple = ('.jpg', '.jpeg', '.png', '.webp', '.tiff')):
        """
        Process all images in a directory.
        
        Args:
            input_dir: Input directory path
            task_type: Task type to perform
            output_path: Output file path (optional)
            file_extensions: Allowed image file extensions
        """
        input_dir = Path(input_dir)
        
        if not input_dir.is_dir():
            raise ValueError(f"Directory not found: {input_dir}")
        
        # Find all image files
        image_paths = []
        for ext in file_extensions:
            image_paths.extend(input_dir.glob(f"*{ext}"))
            image_paths.extend(input_dir.glob(f"*{ext.upper()}"))
        
        if not image_paths:
            print(f"No images found in {input_dir}")
            return
        
        print(f"Found {len(image_paths)} image(s)")
        
        # Load and process images
        images = self.load_images([str(p) for p in image_paths])
        results = self.run_inference(images, task_type)
        
        # Save results if output path provided
        if output_path:
            self.save_results(results, output_path)
        
        return results
    
    def process_files(self, 
                     image_paths: List[str], 
                     task_type: str,
                     output_path: str = None):
        """
        Process specific image files.
        
        Args:
            image_paths: List of image file paths
            task_type: Task type to perform
            output_path: Output file path (optional)
        """
        images = self.load_images(image_paths)
        
        if not images:
            print("No valid images to process")
            return
        
        results = self.run_inference(images, task_type)
        
        # Save results if output path provided
        if output_path:
            self.save_results(results, output_path)
        
        return results


def main():
    """Main entry point for command-line interface."""
    parser = argparse.ArgumentParser(
        description="Surya OCR Inference Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single image
  python infer.py --images image.jpg --task extract_text --output results.json
  
  # Process multiple images
  python infer.py --images img1.jpg img2.png --task detect_layout --output results.json
  
  # Process directory
  python infer.py --input-dir ./images --task process_tables --output results.json
  
  # Extract text and save as txt
  python infer.py --images document.jpg --task extract_text --output results.txt --format txt
        """
    )
    
    # Input arguments
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--images', 
        nargs='+',
        help='One or more image file paths'
    )
    input_group.add_argument(
        '--input-dir',
        help='Directory containing images to process'
    )
    
    # Task argument
    parser.add_argument(
        '--task',
        required=True,
        choices=['extract_text', 'detect_text', 'detect_layout', 'process_tables'],
        help='Task type to perform'
    )
    
    # Output arguments
    parser.add_argument(
        '--output',
        help='Output file path (optional)'
    )
    parser.add_argument(
        '--format',
        default='json',
        choices=['json', 'txt'],
        help='Output format (default: json)'
    )
    
    args = parser.parse_args()
    
    # Initialize inference engine
    inferencer = SuryaInference()
    
    try:
        # Process images
        if args.images:
            results = inferencer.process_files(
                args.images,
                args.task,
                args.output
            )
        else:
            results = inferencer.process_directory(
                args.input_dir,
                args.task,
                args.output
            )
        
        # Print summary
        if results:
            print(f"\n{'='*50}")
            print(f"Successfully processed {len(results)} image(s)")
            print(f"Task: {args.task}")
            if args.output:
                print(f"Output saved to: {args.output}")
            print(f"{'='*50}")
            
    except Exception as e:
        print(f"\nError: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
