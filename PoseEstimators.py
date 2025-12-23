import math

import cv2
import numpy as np


class CustomPoseEstimator:
    def __init__(self):
        self.prev_euler = np.zeros(3)
        self.smoothing = 0.5

    def estimate_pose(self, face_landmarks, w, h):
        # 1. KEY POINTS
        # Vertical axis
        top = np.array([face_landmarks[10].x * w, face_landmarks[10].y * h])
        bridge = np.array([face_landmarks[6].x * w, face_landmarks[6].y * h])

        # Horizontal points for Roll and Yaw midpoint
        l_ear = np.array([face_landmarks[454].x * w, face_landmarks[454].y * h])
        r_ear = np.array([face_landmarks[234].x * w, face_landmarks[234].y * h])
        l_brow = np.array([face_landmarks[332].x * w, face_landmarks[332].y * h])
        r_brow = np.array([face_landmarks[103].x * w, face_landmarks[103].y * h])

        # 2. CALCULATE ROLL (Euler Z)
        angle_ears = math.atan2(l_ear[1] - r_ear[1], l_ear[0] - r_ear[0])
        angle_brows = math.atan2(l_brow[1] - r_brow[1], l_brow[0] - r_brow[0])
        roll = (angle_ears + angle_brows) / 2

        # 3. CALCULATE YAW (Euler Y)
        h_midpoint = (l_ear + r_ear + l_brow + r_brow) / 4
        h_width = np.linalg.norm(l_ear - r_ear)
        yaw = (bridge[0] - h_midpoint[0]) / (h_width / 2)
        # Apply sensitivity if needed (e.g., yaw * 1.2)

        # 4. CALCULATE PITCH (Euler X)
        v_midpoint = (top + bridge) / 2
        v_length = np.linalg.norm(top - bridge)

        # REMOVED the negative sign to match PnP orientation
        pitch_raw = (h_midpoint[1] - v_midpoint[1]) / (v_length / 2)
        pitch = pitch_raw * 0.7

        # 5. SMOOTHING
        current_euler = np.array([pitch, yaw, roll])
        self.prev_euler = (self.smoothing * self.prev_euler) + (
            (1 - self.smoothing) * current_euler
        )

        return self.prev_euler


class PnPPoseEstimator:
    def __init__(self):
        # 3D model points exactly from your stable script
        self.face_real_world = np.array(
            [
                [285, 528, 200],  # Nose tip (1)
                [285, 371, 152],  # Forehead/Upper Center (9)
                [197, 574, 128],  # Left Mouth (57)
                [173, 425, 108],  # Left Eye (130)
                [360, 574, 128],  # Right Mouth (287)
                [391, 425, 108],  # Right Eye (359)
            ],
            dtype=np.float64,
        )

        # Landmark indices matching the stable script
        self.indices = [1, 9, 57, 130, 287, 359]

        self.prev_euler = np.zeros(3)
        self.smoothing = 0.4  # Slightly lower smoothing since PnP is stable now

    def estimate_pose(self, face_landmarks, img_w, img_h):
        # 1. Map 2D landmarks to pixel coordinates
        face_image_coords = []
        for idx in self.indices:
            lm = face_landmarks[idx]
            face_image_coords.append([lm.x * img_w, lm.y * img_h])

        face_image_coords = np.array(face_image_coords, dtype=np.float64)
        # 2. Camera Matrix (Same as stable script)
        focal_length = img_w
        cam_matrix = np.array(
            [[focal_length, 0, img_w / 2], [0, focal_length, img_h / 2], [0, 0, 1]],
            dtype=np.float64,
        )
        # 3. Solve PnP
        dist_matrix = np.zeros((4, 1), dtype=np.float64)
        success, rotation_vec, _ = cv2.solvePnP(
            self.face_real_world, face_image_coords, cam_matrix, dist_matrix
        )

        if not success:
            return self.prev_euler
        # 4. Rodrigues to Matrix
        rotation_matrix, _ = cv2.Rodrigues(rotation_vec)
        # 5. Stable Euler Extraction (XYZ order from script)
        # Note: We return Radians to keep consistency with your dashboard
        x = math.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        y = math.atan2(
            -rotation_matrix[2, 0],
            math.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2),
        )
        z = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])

        current_euler = np.array([x, y, z])
        # 6. Apply Smoothing
        self.prev_euler = (self.smoothing * self.prev_euler) + (
            (1 - self.smoothing) * current_euler
        )

        return self.prev_euler
