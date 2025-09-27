from fastapi import FastAPI, Depends, HTTPException, Header , Form ,UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from passlib.hash import bcrypt
from jose import jwt, JWTError
from datetime import datetime, timedelta
import shutil
import uuid


import requests

from fastapi.responses import FileResponse
import os


import cv2
import numpy as np
import torch

import subprocess

import uuid
import torch
import shutil
import subprocess
import numpy as np






# Example using MiDaS depth model
import torchvision.transforms as transforms
from PIL import Image

# ========================
# Database Setup
# ========================
DATABASE_URL = "sqlite:///./users.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)  # 👈 added
    email = Column(String, unique=True, index=True, nullable=False)
    password = Column(String, nullable=False)

Base.metadata.create_all(bind=engine)

# ========================
# Auth Config
# ========================
SECRET_KEY = "your-secret-key"  # change this in production
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# ========================
# FastAPI App
# ========================
app = FastAPI(title="2D to VR180 Converter")

# Allow frontend (React) to access API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://2d-video-3d-video.vercel.app"], # React dev servers
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependency: Get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ========================
# Auth Functions
# ========================
def create_access_token(data: dict, expires_delta: timedelta | None = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        return email
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

# ========================
# Routes
# ========================

from pydantic import BaseModel

class UserSignup(BaseModel):
    username: str
    email: str
    password: str

class UserAuth(BaseModel):
    email: str
    password: str

from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

@app.post("/signup")
def signup(user: UserSignup, db: Session = Depends(get_db)):
    existing_email = db.query(User).filter(User.email == user.email).first()
    existing_username = db.query(User).filter(User.username == user.username).first()
    if existing_email:
        raise HTTPException(status_code=400, detail="Email already registered")
    if existing_username:
        raise HTTPException(status_code=400, detail="Username already taken")

    hashed_pw = pwd_context.hash(user.password)
    new_user = User(username=user.username, email=user.email, password=hashed_pw)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return {"msg": "User created successfully"}

@app.post("/login")
def login(user: UserAuth, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.email == user.email).first()
    if not db_user or not pwd_context.verify(user.password, db_user.password):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    access_token = create_access_token(
        data={"sub": db_user.email}, 
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    return {"token": access_token, "username": db_user.username}

@app.get("/profile")
def profile(authorization: str = Header(...), db: Session = Depends(get_db)):
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization header")

    token = authorization.split(" ")[1]
    email = verify_token(token)

    user = db.query(User).filter(User.email == email).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    return {
        "id": user.id,
        "username": user.username,
        "email": user.email,
        "message": f"Welcome back {user.username} 🎉"
    }




feedbacks = []

class Feedback(BaseModel):
    username: str
    text: str

@app.get("/feedback")
def get_feedback():
    return feedbacks

@app.post("/feedback")
def post_feedback(item: Feedback):
    feedbacks.append(item.dict())
    return {"message": "Feedback added"}





import os
import uuid
import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
import subprocess
import logging
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import shutil
import tempfile
import json


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)




# Device configuration
# device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")
    logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
elif torch.backends.mps.is_available():  # For Apple Silicon
    device = torch.device("mps")
    logger.info("Using Apple MPS device")
else:
    device = torch.device("cpu")
    logger.info("Using CPU")
logger.info(f"Using device: {device}")

# Load MiDaS model
try:
    midas = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid")
    midas.to(device)
    midas.eval()
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = midas_transforms.dpt_transform
    logger.info("MiDaS model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load MiDaS model: {e}")
    raise

def check_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

if not check_ffmpeg():
    logger.error("ffmpeg is not installed. Video processing will fail.")
    raise RuntimeError("ffmpeg is required for video processing")

def run_command(cmd, description):
    """Run a command with error handling"""
    try:
        logger.info(f"{description}: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"{description} failed: {e.stderr}")
        return False

def inject_vr180_metadata_ffmpeg(input_video, output_video):
    """
    Inject proper VR180 metadata using FFmpeg with the correct metadata format
    """
    # First method: Use the proper VR180 metadata format
    metadata_cmd = [
        "ffmpeg", "-y", "-i", input_video,
        "-movflags", "use_metadata_tags",
        "-metadata", "spherical=1",
        "-metadata", "stereo=1",
        "-metadata", "stereo-mode=left-right",
        "-metadata", "projection-type=equirectangular",
        "-metadata", "video-full-range-off=1",
        "-metadata", "video-color-primaries-unset=1",
        "-metadata", "video-transfer-characteristics-unset=1",
        "-metadata", "video-matrix-coefficients-unset=1",
        "-c", "copy",
        output_video
    ]
    
    if run_command(metadata_cmd, "Injecting VR180 metadata with FFmpeg (method 1)"):
        return True
    
    # Second method: Alternative metadata format
    metadata_cmd_2 = [
        "ffmpeg", "-y", "-i", input_video,
        "-movflags", "use_metadata_tags",
        "-metadata", "com.google.vr.spherical=1",
        "-metadata", "com.google.vr.stereo=1",
        "-metadata", "com.google.vr.stereo-mode=left-right",
        "-metadata", "com.google.vr.projection-type=equirectangular",
        "-c", "copy",
        output_video
    ]
    
    if run_command(metadata_cmd_2, "Injecting VR180 metadata with FFmpeg (method 2)"):
        return True
    
    # Third method: Use a temporary file to inject metadata
    temp_metadata_file = os.path.join(tempfile.gettempdir(), f"metadata_{uuid.uuid4().hex}.txt")
    with open(temp_metadata_file, 'w') as f:
        f.write(";FFMETADATA1\n")
        f.write("spherical=1\n")
        f.write("stereo=1\n")
        f.write("stereo-mode=left-right\n")
        f.write("projection-type=equirectangular\n")
    
    metadata_cmd_3 = [
        "ffmpeg", "-y", "-i", input_video,
        "-i", temp_metadata_file,
        "-map_metadata", "1",
        "-movflags", "use_metadata_tags",
        "-c", "copy",
        output_video
    ]
    
    try:
        result = run_command(metadata_cmd_3, "Injecting VR180 metadata with FFmpeg (method 3)")
        os.remove(temp_metadata_file)
        return result
    except:
        if os.path.exists(temp_metadata_file):
            os.remove(temp_metadata_file)
        return False

def create_vr180_video_simple(input_path, output_path, depth_intensity=10.0):
    """Create a simple VR180 video using frame processing"""
    try:
        # Create temp directory for frames
        temp_dir = tempfile.mkdtemp()
        frames_dir = os.path.join(temp_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)
        
        # Get video info
        cap = cv2.VideoCapture(input_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        logger.info(f"Video info: {width}x{height} @ {fps}fps")
        
        # Process each frame
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            if frame_count % 10 == 0:
                logger.info(f"Processing frame {frame_count}")
            
            # Convert to RGB for depth estimation
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Apply transform and get depth map
            input_batch = transform(frame_rgb).to(device)
            
            with torch.no_grad():
                depth = midas(input_batch)
                depth = depth.squeeze().cpu().numpy()
            
            # Normalize depth map
            depth_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
            depth_norm = depth_norm.astype(np.uint8)
            
            # Resize depth map
            depth_resized = cv2.resize(depth_norm, (width, height))
            
            # Create right view by shifting pixels based on depth
            shift_map = (depth_resized / 255.0 * depth_intensity).astype(np.int32)
            
            # Create coordinate maps
            y_coords, x_coords = np.indices((height, width))
            
            # Shift x coordinates for right view
            x_shifted = np.clip(x_coords + shift_map, 0, width - 1)
            
            # Create right view by remapping
            right_view = frame[y_coords, x_shifted]
            
            # Combine into side-by-side stereo (left + right)
            stereo_frame = np.hstack((frame, right_view))
            
            # Save frame
            frame_path = os.path.join(frames_dir, f"frame_{frame_count:06d}.png")
            cv2.imwrite(frame_path, stereo_frame)
        
        cap.release()
        
        if frame_count == 0:
            logger.error("No frames processed")
            return False
        
        # Create video from frames using ffmpeg
        temp_video = os.path.join(temp_dir, "temp_video.mp4")
        ffmpeg_cmd = [
            "ffmpeg", "-y", "-framerate", str(fps),
            "-i", os.path.join(frames_dir, "frame_%06d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-crf", "23", temp_video
        ]
        
        if not run_command(ffmpeg_cmd, "Creating video from frames"):
            return False
        
        # Copy audio from original video to temp video
        video_with_audio = os.path.join(temp_dir, "video_with_audio.mp4")
        audio_cmd = [
            "ffmpeg", "-y", "-i", temp_video, "-i", input_path,
            "-c:v", "copy", "-map", "0:v:0", "-map", "1:a:0",
            "-shortest", video_with_audio
        ]
        
        if not run_command(audio_cmd, "Adding audio to video"):
            # If audio copy fails, use video without audio
            video_with_audio = temp_video
        
        # Inject VR180 metadata
        metadata_video = os.path.join(temp_dir, "video_with_metadata.mp4")
        if not inject_vr180_metadata_ffmpeg(video_with_audio, metadata_video):
            logger.warning("Metadata injection failed, using video without metadata")
            metadata_video = video_with_audio
        
        # Move final video to output path
        shutil.move(metadata_video, output_path)
        
        # Cleanup
        shutil.rmtree(temp_dir)
        
        return os.path.exists(output_path)
        
    except Exception as e:
        logger.error(f"Error in create_vr180_video_simple: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

@app.post("/convert-2d-to-vr180")
async def convert_2d_to_vr180(
    file: UploadFile = File(..., description="2D video file to convert"),
    depth_intensity: Optional[float] = 10.0
):
    """
    Convert a 2D video to VR180 format with proper metadata for VR headsets.
    """
    if not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be a video")
    
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    input_path = os.path.join(temp_dir, f"input_{uuid.uuid4()}.mp4")
    output_path = os.path.join(temp_dir, f"output_{uuid.uuid4()}.mp4")
    
    try:
        # Save uploaded file
        logger.info(f"Saving uploaded file to: {input_path}")
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        if not os.path.exists(input_path):
            raise HTTPException(status_code=500, detail="Failed to save uploaded file")
        
        logger.info(f"Starting VR180 conversion for: {input_path}")
        
        # Create VR180 video
        success = create_vr180_video_simple(input_path, output_path, depth_intensity)
        
        if not success or not os.path.exists(output_path):
            logger.error(f"Output file does not exist: {output_path}")
            # List files in temp directory for debugging
            if os.path.exists(temp_dir):
                files = os.listdir(temp_dir)
                logger.error(f"Files in temp directory: {files}")
            raise HTTPException(status_code=500, detail="Failed to create VR180 video")
        
        logger.info(f"Conversion successful. Output file: {output_path}")
        logger.info(f"File size: {os.path.getsize(output_path)} bytes")
        
        # Verify metadata was injected
        verify_cmd = ["ffprobe", "-v", "quiet", "-show_format", "-print_format", "json", output_path]
        result = subprocess.run(verify_cmd, capture_output=True, text=True)
        if result.returncode == 0:
            info = json.loads(result.stdout)
            logger.info(f"Video metadata: {info.get('format', {}).get('tags', {})}")
        
        # Return the converted video
        return FileResponse(
            output_path, 
            media_type="video/mp4", 
            filename=f"vr180_{file.filename}"
        )
        
    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        raise HTTPException(status_code=500, detail=f"Conversion failed: {str(e)}")
    
    finally:
        # Note: We don't clean up the temp directory here because
        # Starlette's FileResponse will handle cleaning up after serving the file
        pass

@app.get("/health")
async def health_check():
    return {"status": "healthy", "device": str(device)}

