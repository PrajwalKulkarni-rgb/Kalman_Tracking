import cv2
from ultralytics import YOLO
import os
from datetime import datetime
from tracker import Tracker
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
import shutil

# --- FastAPI App Initialization ---
app = FastAPI(title="Vehicle Tracking API")

# --- Setup and Global Variables ---
os.makedirs("uploads", exist_ok=True)
os.makedirs("outputs", exist_ok=True)
os.makedirs("weights", exist_ok=True)
model_path = "weights/yolov8n.pt"
model = YOLO(model_path)
tracker = Tracker()

# --- Core Processing Function ---
def process_video(video_path, output_filename):
    """
    Processes a video file to track vehicles and saves the output.
    This is the core logic from your original main script.
    """
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    video_writer = cv2.VideoWriter(output_filename, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    local_tracker = Tracker() # Use a fresh tracker for each video

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        results = model(frame, verbose=False)
        detections = []
        for result in results:
            for box in result.boxes:
                if int(box.cls) in [2, 5, 7]: 
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    detections.append([x1, y1, x2, y2])
        
        tracks = local_tracker.update(detections)
        
        for track in tracks:
            x1, y1, x2, y2 = map(int, track.bbox)
            track_id = track.track_id
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        video_writer.write(frame)

    video_writer.release()
    cap.release()
    print(f"✅ Processing complete. Video saved to: {output_filename}")


# --- API Endpoint ---
@app.post("/track_video/")
async def track_video_endpoint(file: UploadFile = File(...)):
    # Save the uploaded video file temporarily
    input_path = os.path.join("uploads", file.filename)
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Define the output path
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_filename = f"outputs/output_{timestamp}_{file.filename}"

    # Run the processing
    process_video(input_path, output_filename)

    # Return the processed video file to the user
    return FileResponse(path=output_filename, media_type='video/mp4', filename=os.path.basename(output_filename))