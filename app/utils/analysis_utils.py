import os
import shutil
import uuid
import numpy as np
import cv2
import json
import ffmpeg
from fastapi import UploadFile
from app.core.config import settings
import onnxruntime as rt
model_session = rt.InferenceSession(settings.MODEL_PATH)

# In-memory storage for demonstration
storage = {}

# Load labels
with open(settings.LABELS_PATH, "r") as f:
    labels = json.load(f)


async def save_upload_file(file: UploadFile):
    request_id = str(uuid.uuid4())
    file_path = os.path.join(settings.UPLOAD_FOLDER, request_id, file.filename)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    storage[request_id] = {"status": "pending", "file_path": file_path}
    return request_id


def img_stats(a, name={}):
    return {
        "name": name,
        "size": a.shape,
        "mean": "{:.2f}".format(a.mean()),
        "std": "{:.2f}".format(a.std()),
        "max": a.max(),
        "min": a.min(),
        "median": "{:.2f}".format(np.median(a)),
    }


def center_crop(img, out_height, out_width):
    height, width, _ = img.shape
    left = int((width - out_width) / 2)
    right = int((width + out_width) / 2)
    top = int((height - out_height) / 2)
    bottom = int((height + out_height) / 2)
    img = img[top:bottom, left:right]
    return img


def resize_with_aspectratio(img, out_height, out_width, scale=87.5, inter_pol=cv2.INTER_LINEAR):
    height, width, _ = img.shape
    new_height = int(100. * out_height / scale)
    new_width = int(100. * out_width / scale)
    if height > width:
        w = new_width
        h = int(new_height * height / width)
    else:
        h = new_height
        w = int(new_width * width / height)
    img = cv2.resize(img, (w, h), interpolation=inter_pol)
    return img


def pre_process_edgetpu(img, dims):
    output_height, output_width, _ = dims
    img = resize_with_aspectratio(img, output_height, output_width, inter_pol=cv2.INTER_LINEAR)
    img = center_crop(img, output_height, output_width)
    img = np.asarray(img, dtype='float32')
    img -= [127.0, 127.0, 127.0]
    img /= [128.0, 128.0, 128.0]
    return img


def extract_frames(video_path):
    probe = ffmpeg.probe(video_path)
    video_info = next(stream for stream in probe['streams'] if stream['codec_type'] == 'video')
    width = int(video_info['width'])
    height = int(video_info['height'])
    out, _ = (
        ffmpeg
        .input(video_path)
        .filter('scale',224,224)
        .output('pipe:', format='rawvideo', pix_fmt='rgb24', fps_mode='vfr')
        .run(capture_stdout=True)
    )
    video = np.frombuffer(out, np.uint8).reshape([-1, 224, 224, 3])
    # video = np.frombuffer(out, np.uint8).reshape([-1, height, width, 3])
    return video


def analyze_content(request_id: str):

    # Get the file path from storage
    file_path = storage[request_id]["file_path"]

    # Extract frames using ffmpeg
    frames = extract_frames(file_path)

    frame_results = []

    for frame_index, frame in enumerate(frames):
        print(frame_index,end='\r')
        # Preprocess the frame
        img = pre_process_edgetpu(frame, (224, 224, 3))

        # Create a batch of 1
        img_batch = np.expand_dims(img, axis=0)

        # Run inference
        results = model_session.run(None, {"images:0": img_batch})[0]
        threshold = 0.8

        # Find indices where probability is above the threshold
        above_threshold_indices = np.where(results[0] > threshold)[0]

        # Get the corresponding probabilities and sort them in descending order
        sorted_indices = above_threshold_indices[np.argsort(results[0][above_threshold_indices])[::-1]]

        frame_result = []
        for r in sorted_indices:
            frame_result.append({
                "class": labels[str(r)],
                "probability": float(results[0][r])
            })

        if frame_result:  # Only append if there are results above threshold
            frame_results.append({
                "frame_index": frame_index,
                "detections": frame_result
            })

    # Save the results to storage
    storage[request_id].update({
        "status": "completed",
        "results": frame_results
    })


def get_analysis_results(request_id: str):
    return storage.get(request_id)


def delete_analysis(request_id: str):
    if request_id in storage:
        # Check if the status is not pending
        if storage[request_id]["status"] == "pending":
            return False

        # Get the file path and folder path
        file_path = storage[request_id]["file_path"]
        folder_path = os.path.dirname(file_path)

        # Delete the file if it exists
        if os.path.exists(file_path):
            os.remove(file_path)

        # Delete the folder if it exists
        if os.path.exists(folder_path):
            os.rmdir(folder_path)

        # Remove the request from storage
        del storage[request_id]
        return True
    return False
