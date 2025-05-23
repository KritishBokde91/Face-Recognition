import os
import torch
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import io
from PIL import Image
import threading
import time
from fastapi.middleware.cors import CORSMiddleware
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Force CPU usage and reduce memory
os.environ["CUDA_VISIBLE_DEVICES"] = ""
torch.set_num_threads(1)

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

app = FastAPI(title="Face Recognition API", version="1.0.0")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# Global variables
reference_embedding = None
lock = threading.Lock()
models_loaded = False
mtcnn = None
model = None

def lazy_load_models():
    """Load models only when needed"""
    global mtcnn, model, models_loaded
    
    if models_loaded:
        return True
        
    try:
        logger.info("Loading models...")
        
        # Import here to delay loading
        from facenet_pytorch import MTCNN, InceptionResnetV1
        from sklearn.metrics.pairwise import cosine_similarity
        
        device = torch.device('cpu')
        
        # Load with minimal settings
        mtcnn = MTCNN(
            image_size=160, 
            margin=0, 
            keep_all=False, 
            post_process=False, 
            device=device,
            select_largest=True,
            min_face_size=20  # Faster detection
        )
        
        model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
        
        # Optimize for inference
        torch.set_grad_enabled(False)
        
        models_loaded = True
        logger.info("Models loaded successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to load models: {str(e)}")
        return False

def get_face_embedding(img):
    """Detect face and return embedding"""
    if not lazy_load_models():
        return None
        
    try:
        if img.mode != 'RGB':
            img = img.convert('RGB')
            
        # Resize image if too large (save memory)
        if img.size[0] > 640 or img.size[1] > 640:
            img.thumbnail((640, 640), Image.Resampling.LANCZOS)
            
        face = mtcnn(img)
        if face is not None:
            face = (face - 127.5) / 128.0
            with torch.no_grad():
                face_embedding = model(face.unsqueeze(0))
            return face_embedding.detach().cpu().numpy()
        return None
    except Exception as e:
        logger.error(f"Error in face embedding: {str(e)}")
        return None

def match_faces(embedding1, embedding2, threshold=0.6):
    """Compare embeddings"""
    if not lazy_load_models():
        return False, 0.0
        
    try:
        from sklearn.metrics.pairwise import cosine_similarity
        
        if embedding1 is None or embedding2 is None:
            return False, 0.0
        similarity = cosine_similarity(embedding1, embedding2)
        is_match = bool(similarity[0][0] > threshold)
        score = float(similarity[0][0])
        return is_match, score
    except Exception as e:
        logger.error(f"Error in face matching: {str(e)}")
        return False, 0.0

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Face Recognition API",
        "status": "running",
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": models_loaded,
        "timestamp": time.time()
    }

@app.get("/test_connection")
async def test_connection():
    """Test endpoint"""
    return {
        "status": "connected", 
        "timestamp": float(time.time()),
        "server": "railway"
    }

@app.post("/upload_reference_image/")
async def upload_reference_image(file: UploadFile = File(...)):
    """Handle reference image upload"""
    global reference_embedding
    
    logger.info(f"Processing reference image: {file.filename}")
    
    try:
        # Validate file
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        contents = await file.read()
        if len(contents) == 0:
            raise HTTPException(status_code=400, detail="Empty file")
            
        if len(contents) > 5 * 1024 * 1024:  # 5MB limit
            raise HTTPException(status_code=413, detail="File too large (max 5MB)")
            
        # Process image
        try:
            image = Image.open(io.BytesIO(contents))
            if image.mode == 'RGBA':
                background = Image.new('RGB', image.size, (255, 255, 255))
                background.paste(image, mask=image.split()[-1])
                image = background
            elif image.mode != 'RGB':
                image = image.convert('RGB')
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image: {str(e)}")
            
        # Get embedding
        embedding = get_face_embedding(image)
        if embedding is None:
            raise HTTPException(status_code=400, detail="No face detected in reference image")
            
        with lock:
            reference_embedding = embedding
            
        return {
            "status": "success",
            "message": "Reference image processed successfully",
            "image_size": image.size
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/process_frame/")
async def process_frame(file: UploadFile = File(...)):
    """Process frame and return match result"""
    if reference_embedding is None:
        raise HTTPException(status_code=400, detail="No reference image uploaded")
    
    try:
        # Validate file
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        contents = await file.read()
        if len(contents) == 0:
            raise HTTPException(status_code=400, detail="Empty file")
            
        # Process image
        try:
            image = Image.open(io.BytesIO(contents))
            if image.mode == 'RGBA':
                background = Image.new('RGB', image.size, (255, 255, 255))
                background.paste(image, mask=image.split()[-1])
                image = background
            elif image.mode != 'RGB':
                image = image.convert('RGB')
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image: {str(e)}")
        
        # Get embedding and compare
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
        
        return convert_numpy_types(result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Frame processing error: {str(e)}")
        raise HTTPException(status_code=500, detail="Error processing frame")

@app.get("/status/")
async def get_status():
    """Check service status"""
    return {
        "status": "running",
        "reference_loaded": reference_embedding is not None,
        "device": "cpu",
        "model_loaded": models_loaded
    }

@app.delete("/clear_reference/")
async def clear_reference():
    """Clear reference image"""
    global reference_embedding
    with lock:
        reference_embedding = None
    return {"status": "success", "message": "Reference image cleared"}

# Railway deployment
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"Starting server on port {port}")
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=port,
        workers=1,  # Single worker to save memory
        timeout_keep_alive=30
    )