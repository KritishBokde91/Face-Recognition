import os
import torch
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.metrics.pairwise import cosine_similarity
import io
from PIL import Image
import threading
import time
import queue
from fastapi.middleware.cors import CORSMiddleware
import logging
import base64
import atexit

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def convert_numpy_types(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj

app = FastAPI()

# Enhanced CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# Initialize models with error handling
try:
    device = torch.device('cpu')
    mtcnn = MTCNN(image_size=160, margin=0, keep_all=False, post_process=False, device=device)
    model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    logger.info(f"Models loaded successfully on {device}")
except Exception as e:
    logger.error(f"Failed to load models: {str(e)}")
    raise

# Global variables with thread safety
reference_embedding = None
lock = threading.Lock()

def get_face_embedding(img):
    """Detect face and return embedding with error handling"""
    try:
        # Convert PIL Image to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
            
        # Detect face
        face = mtcnn(img)
        if face is not None:
            # Normalize face tensor
            face = (face - 127.5) / 128.0
            with torch.no_grad():  # Prevent gradient computation
                face_embedding = model(face.unsqueeze(0))
            return face_embedding.detach().cpu().numpy()
        return None
    except Exception as e:
        logger.error(f"Error in face embedding: {str(e)}")
        return None

def match_faces(embedding1, embedding2, threshold=0.6):
    """Compare embeddings with error handling"""
    try:
        if embedding1 is None or embedding2 is None:
            return False, 0.0
        similarity = cosine_similarity(embedding1, embedding2)
        # Convert numpy types to Python native types for JSON serialization
        is_match = bool(similarity[0][0] > threshold)
        score = float(similarity[0][0])
        return is_match, score
    except Exception as e:
        logger.error(f"Error in face matching: {str(e)}")
        return False, 0.0

# Add test connection endpoint
@app.get("/test_connection")
async def test_connection():
    """Test endpoint for connection verification"""
    return {"status": "connected", "timestamp": float(time.time())}

@app.post("/upload_reference_image/")
async def upload_reference_image(file: UploadFile = File(...)):
    """Handle reference image upload with proper validation"""
    global reference_embedding
    
    logger.info(f"Received upload request for reference file: {file.filename}")
    
    try:
        # Validate file type
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read and verify image
        contents = await file.read()
        if len(contents) == 0:
            raise HTTPException(status_code=400, detail="Empty file")
            
        if len(contents) > 5 * 1024 * 1024:  # 5MB limit
            raise HTTPException(status_code=413, detail="File too large (max 5MB)")
            
        try:
            image = Image.open(io.BytesIO(contents))
            if image.mode not in ['RGB', 'L', 'RGBA']:
                image = image.convert('RGB')
            elif image.mode == 'RGBA':
                # Handle transparency by creating white background
                background = Image.new('RGB', image.size, (255, 255, 255))
                background.paste(image, mask=image.split()[-1])
                image = background
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")
            
        # Process face
        embedding = get_face_embedding(image)
        if embedding is None:
            raise HTTPException(status_code=400, detail="No face detected in the reference image")
            
        with lock:
            reference_embedding = embedding
            
        return {
            "status": "success",
            "message": "Reference image processed successfully",
            "image_size": image.size,
            "image_mode": image.mode
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/process_frame/")
async def process_frame(file: UploadFile = File(...)):
    """Process frame from mobile camera and return match result"""
    if reference_embedding is None:
        raise HTTPException(status_code=400, detail="No reference image uploaded")
    
    try:
        # Validate file type
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read image
        contents = await file.read()
        if len(contents) == 0:
            raise HTTPException(status_code=400, detail="Empty file")
            
        try:
            image = Image.open(io.BytesIO(contents))
            if image.mode not in ['RGB', 'L', 'RGBA']:
                image = image.convert('RGB')
            elif image.mode == 'RGBA':
                background = Image.new('RGB', image.size, (255, 255, 255))
                background.paste(image, mask=image.split()[-1])
                image = background
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")
        
        # Get face embedding from current frame
        embedding = get_face_embedding(image)
        
        if embedding is not None:
            is_match, score = match_faces(reference_embedding, embedding)
            result = {
                "match": bool(is_match),
                "score": float(score),
                "timestamp": float(time.time()),
                "face_detected": True
            }
        else:
            result = {
                "match": False,
                "score": 0.0,
                "timestamp": float(time.time()),
                "face_detected": False
            }
        
        # Ensure all numpy types are converted
        result = convert_numpy_types(result)
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Frame processing error: {str(e)}")
        raise HTTPException(status_code=500, detail="Error processing frame")

# Keep the old endpoint for backward compatibility (but it won't be used)
@app.get("/get_match_result/")
async def get_match_result():
    """Legacy endpoint - returns empty result"""
    if reference_embedding is None:
        raise HTTPException(status_code=400, detail="No reference image uploaded")
        
    return {
        "match": False,
        "score": 0.0,
        "message": "Use mobile camera for live processing",
        "timestamp": float(time.time()),
        "face_detected": False
    }

@app.get("/status/")
async def get_status():
    """Check service status"""
    return {
        "status": "running",
        "reference_loaded": reference_embedding is not None,
        "device": str(device),
        "model_loaded": True
    }

@app.delete("/clear_reference/")
async def clear_reference():
    """Clear reference image"""
    global reference_embedding
    with lock:
        reference_embedding = None
    return {"status": "success", "message": "Reference image cleared"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)