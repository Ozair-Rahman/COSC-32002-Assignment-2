from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import cv2
import numpy as np
from utils.feature_detectors import (
    harris_corner_detector,
    shi_tomasi_corner_detector,
    fast_detector,
    orb_detector,
)
from utils.feature_matchers import (
    brute_force_matching,
    flann_matching,
    ransac,
)
from utils.face_detection import detect_faces

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.post("/detect-corners/")
async def detect_corners(file: UploadFile = File(...), method: str = "harris"):
    contents = await file.read()
    image = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)

    if method == "harris":
        corners = harris_corner_detector(image)
        return corners
    elif method == "shi-tomasi":
        corners = shi_tomasi_corner_detector(image)
        return corners
    elif method == "fast":
        corners = fast_detector(image)
        return corners
    elif method == "orb":
        corners = orb_detector(image)
        return corners
    else:
        return {"error": "Invalid method specified."}

@app.post("/match-features/")
async def match_features(
    file1: UploadFile = File(...),
    file2: UploadFile = File(...),
    method: str = "brute_force"
):
    contents1 = await file1.read()
    contents2 = await file2.read()
    
    image1 = cv2.imdecode(np.frombuffer(contents1, np.uint8), cv2.IMREAD_COLOR)
    image2 = cv2.imdecode(np.frombuffer(contents2, np.uint8), cv2.IMREAD_COLOR)

    # Detect ORB features and compute descriptors
    keypoints1, descriptors1 = orb_detector(image1)
    keypoints2, descriptors2 = orb_detector(image2)

    if method == "brute_force":
        matches = brute_force_matching(descriptors1, descriptors2)
    elif method == "flann":
        matches = flann_matching(descriptors1, descriptors2)
    else:
        return {"error": "Invalid matching method specified."}

    # Apply RANSAC for outlier detection
    good_matches = ransac(matches)

    return {
        "matches": len(good_matches),
        "keypoints1": [kp.pt for kp in keypoints1],
        "keypoints2": [kp.pt for kp in keypoints2],
    }

@app.post("/detect-faces/")
async def detect_faces_endpoint(file: UploadFile = File(...)):
    contents = await file.read()
    image = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)

    faces = detect_faces(image)

    return {"faces": faces.tolist()}