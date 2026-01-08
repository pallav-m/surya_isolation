from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List, Optional
from PIL import Image
import io
from pydantic import BaseModel, Field
# from contextlib import asynccontextmanager

from utils.serialization import serialize_results

from main import SuryaInferenceEngine

surya_model = SuryaInferenceEngine()


# Initialize FastAPI app
app = FastAPI(
    title="Surya OCR API",
    description="API for OCR, layout analysis, table recognition using Surya",
    version="1.0.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Response models
class ProcessingResponse(BaseModel):
    """Response model for OCR processing."""
    success: bool
    images_processed: int
    results: List[dict]
    message: Optional[str] = None

class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    pipeline_loaded: bool

# Helper functions
def validate_images(files: List[UploadFile]) -> None:
    """Validate uploaded images."""
    if len(files) == 0:
        raise HTTPException(status_code=400, detail="No images provided")
    
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 images allowed")
    
    # Validate file types
    allowed_types = ["image/jpeg", "image/jpg", "image/png", "image/webp", "image/tiff"]
    for file in files:
        if file.content_type not in allowed_types:
            raise HTTPException(
                status_code=415, 
                detail=f"Unsupported file type: {file.content_type}. Allowed: {allowed_types}"
            )

async def files_to_pil_images(files: List[UploadFile]) -> List[Image.Image]:
    """Convert uploaded files to PIL Images."""
    images = []
    for file in files:
        try:
            contents = await file.read()
            image = Image.open(io.BytesIO(contents))
            # Convert to RGB if necessary
            if image.mode not in ('RGB', 'L'):
                image = image.convert('RGB')
            images.append(image)
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Error processing image {file.filename}: {str(e)}"
            )
        finally:
            await file.seek(0)  # Reset file pointer
    
    return images

# API Endpoints

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "pipeline_loaded": surya_model is not None
    }


@app.post("/run_surya_inference", response_model=ProcessingResponse)
async def infer(
    task_type: str,
    files: List[UploadFile] = File(..., description="Upload up to 10 images"),\
):
    """
    ### Run inference on list of images using surya
    - **task_type**: Any of the following: 
    
    > - **extract_text**: Extract text from images
    > - **detect_text**: Detect text lines in images
    > - **detect_layout**: Detect layout elements (tables, images, headers, etc.)
    > - **process_tables**: Extract table structures
    
    - **files**: Up to 10 image files (JPEG, PNG, WebP, TIFF)
    """
    validate_images(files)

    task_functions = {
        "extract_text": surya_model.recognize_text,
        "detect_text": surya_model.detect_text,
        "detect_layout": surya_model.extract_layout,
        "process_tables": surya_model.recognize_tables
    }

    if not task_type in task_functions:
        raise HTTPException(status_code=400,
                            detail=f"Invalid task type. Must be one of: extract_text, detect_text, detect_layout, process_tables")
    
    try:
        # Convert to PIL images
        pil_images = await files_to_pil_images(files)
        
        # Process each image
        # all_results = []
        all_results = task_functions[task_type](pil_images)
        
        return {
            "success": True,
            "images_processed": len(pil_images),
            "results": all_results,
            "message": "Processing completed successfully"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")


# ==========================================================================================

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": exc.detail,
            "status_code": exc.status_code
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler."""
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": f"Internal server error: {str(exc)}",
            "status_code": 500
        }
    )
