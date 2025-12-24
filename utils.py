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
        """
        Takes a dictionary with:
          - 'head_rotation': [pitch, yaw, roll] in RADIANS
          - 'landmarks_positions': {idx: [x, y, z]} (optional)
        Converts rotation to DEGREES and sends via UDP.
        """
        if data is None:
            return
        # 2. Serialize and Send
        try:
            msg = json.dumps(data).encode("utf-8")
            self.sock.sendto(msg, (self.ip, self.port))
        except Exception as e:
            # We print instead of raise to keep the camera loop running
            print(f"UDP Send Error: {e}")


def to_degrees(p, y, r):
    return (float(np.degrees(p)), float(np.degrees(y)), float(np.degrees(r)))


def normalize_face(face_landmarks, euler_angles, center_idx=6):
    """
    Translates and rotates landmarks so the head is centered at (0,0,0)
    and facing perfectly forward (0,0,0 rotation).

    :param face_landmarks: MediaPipe face_landmarks.landmark list
    :param euler_angles: list/array of [pitch, yaw, roll] in radians
    :param center_idx: The landmark to use as the local origin (6 is nose bridge)
    :return: Dictionary of normalized {index: [x, y, z]}
    """
    p, y, r = euler_angles

    # 1. Create Rotation Matrices for each axis
    # Note: Using negative angles to "un-rotate"
    Rx = np.array(
        [[1, 0, 0], [0, math.cos(-p), -math.sin(-p)], [0, math.sin(-p), math.cos(-p)]]
    )
    Ry = np.array(
        [[math.cos(-y), 0, math.sin(-y)], [0, 1, 0], [-math.sin(-y), 0, math.cos(-y)]]
    )
    Rz = np.array(
        [[math.cos(-r), -math.sin(-r), 0], [math.sin(-r), math.cos(-r), 0], [0, 0, 1]]
    )

    # Combined Inverse Rotation Matrix (Order depends on your estimation order)
    # Usually YXZ or ZYX for heads
    R_inv = Rz @ Ry @ Rx

    # 2. Get the Origin (Anchor)
    anchor = face_landmarks[center_idx]
    origin = np.array([anchor.x, anchor.y, anchor.z])

    normalized_lms = {}

    # List of landmarks to track (Chin, Mouth corners, Eyebrows, etc.)
    target_indices = [152, 10, 13, 14, 33, 263]

    for idx in target_indices:
        lm = face_landmarks[idx]
        # Current point as vector
        pt = np.array([lm.x, lm.y, lm.z])

        # Translate to origin
        rel_pt = pt - origin

        # Rotate back to neutral
        neutral_pt = R_inv @ rel_pt

        normalized_lms[idx] = neutral_pt.tolist()

    return normalized_lms
