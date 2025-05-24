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
from dotenv import load_dotenv

load_dotenv()

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

app = FastAPI(
    title="Face Recognition API",
    description="Face recognition service for deployment",
    version="1.0.0"
)

# Enhanced CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# Global variables for lazy loading
mtcnn = None
model = None
device = None
models_loaded = False
model_lock = threading.Lock()

def load_models():
    """Lazy load models to save memory"""
    global mtcnn, model, device, models_loaded
    
    if models_loaded:
        return True
        
    try:
        with model_lock:
            if models_loaded:  # Double-check pattern
                return True
                
            logger.info("Loading models...")
            
            # Force CPU usage and optimize memory
            device = torch.device('cpu')
            torch.set_num_threads(1)  # Limit CPU threads
            
            # Initialize with minimal memory footprint
            mtcnn = MTCNN(
                image_size=112,  # Smaller size to save memory
                margin=0, 
                keep_all=False, 
                post_process=False, 
                device=device,
                min_face_size=40,  # Larger min face to reduce processing
                thresholds=[0.7, 0.8, 0.9]  # Higher thresholds for efficiency
            )
            
            model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
            
            # Enable memory optimizations
            if hasattr(torch.backends, 'cudnn'):
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = True
            
            models_loaded = True
            logger.info(f"Models loaded successfully on {device}")
            return True
            
    except Exception as e:
        logger.error(f"Failed to load models: {str(e)}")
        return False

# Global variables with thread safety
reference_embedding = None
lock = threading.Lock()

def get_face_embedding(img):
    """Detect face and return embedding with error handling and memory optimization"""
    global mtcnn, model
    
    # Load models if not already loaded
    if not load_models():
        return None
        
    try:
        # Convert PIL Image to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
            
        # Resize image more aggressively to save memory
        max_size = 640  # Reduced from 1024
        if max(img.size) > max_size:
            img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            
        # Detect face with memory management
        with torch.no_grad():
            face = mtcnn(img)
            
        if face is not None:
            # Normalize face tensor
            face = (face - 127.5) / 128.0
            with torch.no_grad():
                face_embedding = model(face.unsqueeze(0))
            
            # Immediately move to CPU and cleanup
            result = face_embedding.detach().cpu().numpy()
            del face_embedding, face
            return result
            
        return None
        
    except Exception as e:
        logger.error(f"Error in face embedding: {str(e)}")
        return None
    finally:
        # Aggressive memory cleanup
        if 'face' in locals():
            del face
        if 'face_embedding' in locals():
            del face_embedding
        # Force garbage collection
        import gc
        gc.collect()

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

# Health check endpoint for Render
@app.get("/")
async def root():
    """Root endpoint for health checks - loads models on first access"""
    # Try to load models on first health check
    models_ready = load_models() if not models_loaded else True
    
    return {
        "status": "healthy" if models_ready else "loading",
        "service": "Face Recognition API",
        "version": "1.0.0",
        "models_loaded": models_loaded,
        "timestamp": float(time.time())
    }

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
            
        # Reduced file size limit for Render's memory constraints
        if len(contents) > 1 * 1024 * 1024:  # 1MB limit - much smaller
            raise HTTPException(status_code=413, detail="File too large (max 1MB)")
            
        try:
            image = Image.open(io.BytesIO(contents))
            # Immediately resize to save memory
            if max(image.size) > 512:
                image.thumbnail((512, 512), Image.Resampling.LANCZOS)
                
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
    finally:
        # Clean up memory
        if 'contents' in locals():
            del contents
        if 'image' in locals():
            del image

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
            
        # File size limit for frames - very small
        if len(contents) > 800 * 1024:  # 800KB limit for frames
            raise HTTPException(status_code=413, detail="Frame too large (max 800KB)")
            
        try:
            image = Image.open(io.BytesIO(contents))
            # Aggressively resize frames
            if max(image.size) > 400:
                image.thumbnail((400, 400), Image.Resampling.LANCZOS)
                
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
    finally:
        # Clean up memory
        if 'contents' in locals():
            del contents
        if 'image' in locals():
            del image

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
        "device": str(device) if device else "not loaded",
        "models_loaded": models_loaded,
        "memory_info": {
            "reference_embedding_loaded": reference_embedding is not None
        }
    }

@app.delete("/clear_reference/")
async def clear_reference():
    """Clear reference image"""
    global reference_embedding
    with lock:
        reference_embedding = None
    return {"status": "success", "message": "Reference image cleared"}

# Health check endpoint specifically for Render
@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring"""
    return {
        "status": "healthy",
        "timestamp": float(time.time()),
        "service": "Face Recognition API"
    }

if __name__ == "__main__":
    import uvicorn
    # Get port from environment variable (Render provides this)
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)