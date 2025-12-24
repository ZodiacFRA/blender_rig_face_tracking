import time
import json
import socket
import math

import mediapipe as mp
import numpy as np


class LandmarkDetector:
    def __init__(self, model_path="assets/face_landmarker.task"):
        self.result = None

        # Callback function for async results
        def update_result(result, output_image, timestamp_ms):
            self.result = result

        base_options = mp.tasks.BaseOptions(model_asset_path=model_path)
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.LIVE_STREAM,
            num_faces=1,
            result_callback=update_result,
            output_face_blendshapes=True,
        )
        self.detector = mp.tasks.vision.FaceLandmarker.create_from_options(options)

    def process_async(self, frame):
        # Convert OpenCV frame to MediaPipe Image object
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        timestamp_ms = int(time.time() * 1000)
        self.detector.detect_async(mp_image, timestamp_ms)
        return self.result


class NetworkSender:
    def __init__(self, ip="127.0.0.1", port=5005):
        self.ip = ip
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send_pose(self, data):
        if data is None:
            return
        # 2. Serialize and Send
        msg = json.dumps(data).encode("utf-8")
        try:
            self.sock.sendto(msg, (self.ip, self.port))
        except Exception as e:
            # We print instead of raise to keep the camera loop running
            print(f"UDP Error: {e}")


def to_degrees(p, y, r):
    return (float(np.degrees(p)), float(np.degrees(y)), float(np.degrees(r)))
