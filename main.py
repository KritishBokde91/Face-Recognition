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
from fastapi.middleware.cors import CORSMiddleware
import logging
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
    description="Face recognition service",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Initialize models once at startup
device = torch.device('cpu')
models_loaded = False
mtcnn = None
model = None

@app.on_event("startup")
async def load_models():
    global mtcnn, model, models_loaded
    try:
        logger.info("Loading models...")
        
        # Initialize with minimal settings
        mtcnn = MTCNN(
            image_size=112,
            margin=0,
            keep_all=False,
            post_process=False,
            device=device,
            min_face_size=40,
            thresholds=[0.7, 0.8, 0.9]
        )
        
        model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
        
        # Freeze model and optimize
        for param in model.parameters():
            param.requires_grad = False
            
        torch.set_grad_enabled(False)
        models_loaded = True
        logger.info(f"Models loaded successfully on {device}")
        
    except Exception as e:
        logger.error(f"Failed to load models: {str(e)}")
        raise

# Global variables with thread safety
reference_embedding = None
lock = threading.Lock()

def get_face_embedding(img):
    """Detect face and return embedding"""
    try:
        if img.mode != 'RGB':
            img = img.convert('RGB')
            
        # Resize to save memory
        max_size = 512
        if max(img.size) > max_size:
            img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            
        # Detect face
        face = mtcnn(img)
        if face is not None:
            face = (face - 127.5) / 128.0
            embedding = model(face.unsqueeze(0))
            return embedding.detach().cpu().numpy()
        return None
    except Exception as e:
        logger.error(f"Face embedding error: {str(e)}")
        return None

def match_faces(embedding1, embedding2, threshold=0.6):
    """Compare face embeddings"""
    try:
        if embedding1 is None or embedding2 is None:
            return False, 0.0
        similarity = cosine_similarity(embedding1, embedding2)
        return bool(similarity[0][0] > threshold), float(similarity[0][0])
    except Exception as e:
        logger.error(f"Matching error: {str(e)}")
        return False, 0.0

@app.get("/")
async def health_check():
    return {
        "status": "running",
        "models_loaded": models_loaded,
        "reference_loaded": reference_embedding is not None
    }

@app.get("/test_connection")
async def test_connection():
    return {"status": "connected", "timestamp": time.time()}

@app.post("/upload_reference_image/")
async def upload_reference_image(file: UploadFile = File(...)):
    global reference_embedding
    
    try:
        # Validate input
        if not file.content_type.startswith('image/'):
            raise HTTPException(400, "File must be an image")
            
        # Read image with size limit
        contents = await file.read()
        if len(contents) > 1 * 1024 * 1024:  # 1MB
            raise HTTPException(413, "File too large (max 1MB)")
            
        image = Image.open(io.BytesIO(contents))
        if max(image.size) > 512:
            image.thumbnail((512, 512), Image.Resampling.LANCZOS)
            
        if image.mode == 'RGBA':
            background = Image.new('RGB', image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[-1])
            image = background
        elif image.mode != 'RGB':
            image = image.convert('RGB')
            
        # Process image
        embedding = get_face_embedding(image)
        if embedding is None:
            raise HTTPException(400, "No face detected")
            
        with lock:
            reference_embedding = embedding
            
        return {
            "status": "success",
            "message": "Reference image processed"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        raise HTTPException(500, "Internal server error")

@app.post("/process_frame/")
async def process_frame(file: UploadFile = File(...)):
    if reference_embedding is None:
        raise HTTPException(400, "No reference image uploaded")
    
    try:
        # Validate input
        if not file.content_type.startswith('image/'):
            raise HTTPException(400, "File must be an image")
            
        contents = await file.read()
        if len(contents) > 800 * 1024:  # 800KB
            raise HTTPException(413, "Frame too large (max 800KB)")
            
        image = Image.open(io.BytesIO(contents))
        if max(image.size) > 400:
            image.thumbnail((400, 400), Image.Resampling.LANCZOS)
            
        if image.mode == 'RGBA':
            background = Image.new('RGB', image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[-1])
            image = background
        elif image.mode != 'RGB':
            image = image.convert('RGB')
            
        # Process frame
        embedding = get_face_embedding(image)
        
        if embedding is not None:
            is_match, score = match_faces(reference_embedding, embedding)
            result = {
                "match": is_match,
                "score": score,
                "timestamp": time.time(),
                "face_detected": True
            }
        else:
            result = {
                "match": False,
                "score": 0.0,
                "timestamp": time.time(),
                "face_detected": False
            }
            
        return convert_numpy_types(result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Frame processing error: {str(e)}")
        raise HTTPException(500, "Error processing frame")

@app.delete("/clear_reference/")
async def clear_reference():
    global reference_embedding
    with lock:
        reference_embedding = None
    return {"status": "success"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)