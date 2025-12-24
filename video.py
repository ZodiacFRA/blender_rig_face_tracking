import math

import cv2
import numpy as np


class CameraSource:
    def __init__(self, width=1280 / 4, height=720 / 4):
        self.cap = cv2.VideoCapture(0)
        # self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        # self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def __del__(self):
        self.release()

    def get_frame(self):
        success, frame = self.cap.read()
        if not success:
            return False, None
        return True, cv2.flip(frame, 1)

    def release(self):
        self.cap.release()


class Visualizer:
    def __init__(self):
        # Colors (BGR)
        self.CLR_MESH = (0, 255, 0)  # Green
        self.CLR_VERTICAL = (255, 255, 0)  # Cyan
        self.CLR_HORIZONTAL = (255, 0, 255)  # Magenta
        self.CLR_MIDPOINT = (255, 255, 255)  # White

    def draw_landmarks(self, frame, face_lms, w, h):
        """Draws the full 468+ face mesh as small dots."""
        for lm in face_lms:
            x, y = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (x, y), 1, self.CLR_MESH, -1)

    def draw_reference_lines(self, frame, face_lms, w, h):
        """Draws the refined Skull Box using 103 and 332."""
        # Define Points
        p_top = (int(face_lms[10].x * w), int(face_lms[10].y * h))
        p_bridge = (int(face_lms[6].x * w), int(face_lms[6].y * h))
        p_l_ear = (int(face_lms[454].x * w), int(face_lms[454].y * h))
        p_r_ear = (int(face_lms[234].x * w), int(face_lms[234].y * h))
        p_l_brow = (int(face_lms[332].x * w), int(face_lms[332].y * h))
        p_r_brow = (int(face_lms[103].x * w), int(face_lms[103].y * h))

        # Vertical Line
        cv2.line(frame, p_top, p_bridge, (255, 255, 0), 2)

        # Ear Line (Lower Baseline)
        cv2.line(frame, p_l_ear, p_r_ear, (255, 0, 255), 2)

        # Brow Line (Upper Baseline)
        cv2.line(frame, p_l_brow, p_r_brow, (0, 255, 255), 2)

        # Connect Brows to Ears to show the "Skull Box"
        cv2.line(frame, p_l_ear, p_l_brow, (150, 150, 150), 1)
        cv2.line(frame, p_r_ear, p_r_brow, (150, 150, 150), 1)

    def draw_cross_midpoint(self, frame, face_lms, w, h):
        """Draws the combined centroid of the skull markers."""
        pts = np.array(
            [
                [face_lms[454].x * w, face_lms[454].y * h],
                [face_lms[234].x * w, face_lms[234].y * h],
                [face_lms[332].x * w, face_lms[332].y * h],
                [face_lms[103].x * w, face_lms[103].y * h],
            ]
        )
        centroid = np.mean(pts, axis=0).astype(int)

        cv2.circle(frame, tuple(centroid), 5, (255, 255, 255), -1)

        # Show Yaw/Pitch offset from bridge
        p_bridge = (int(face_lms[6].x * w), int(face_lms[6].y * h))
        cv2.line(frame, p_bridge, tuple(centroid), (0, 165, 255), 2)

    def _get_rotation_matrix(self, pitch, yaw, roll):
        """Helper to ensure consistent Euler integration."""
        Rx = np.array(
            [
                [1, 0, 0],
                [0, np.cos(pitch), -np.sin(pitch)],
                [0, np.sin(pitch), np.cos(pitch)],
            ]
        )
        Ry = np.array(
            [[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]]
        )
        Rz = np.array(
            [
                [np.cos(roll), -np.sin(roll), 0],
                [np.sin(roll), np.cos(roll), 0],
                [0, 0, 1],
            ]
        )
        # Consistency with the 'Skull Box' logic
        return Ry @ Rx @ Rz

    def _map_fixed(self, points, width, height, scale):
        """Maps points using a fixed global scale and centers them."""
        # Calculate current centroid to 'zero' the face position
        centroid = np.mean(points, axis=0)

        # Determine the canvas center
        canvas_center = np.array([width // 2, height // 2])

        # 1. Offset to origin (relative to centroid)
        # 2. Apply the fixed scale (consistent for all axes)
        # 3. Move to canvas center
        mapped = (points - centroid) * (min(width, height) * scale) + canvas_center
        return mapped.astype(int)

    def render_debug_front(self, face_lms, euler_angles, scale, width=400, height=400):
        img = np.zeros((height, width, 3), dtype=np.uint8)
        pts = np.array([[lm.x, lm.y] for lm in face_lms])
        mapped_pts = self._map_fixed(pts, width, height, scale)

        for pt in mapped_pts:
            cv2.circle(img, tuple(pt), 1, (0, 100, 0), -1)

        self._draw_axis_at_center(img, euler_angles, width, height, mode="front")
        return img

    def render_debug_side(self, face_lms, euler_angles, scale, width=400, height=400):
        img = np.zeros((height, width, 3), dtype=np.uint8)
        # We use the same 'scale' for Z as we do for Y
        pts = np.array([[lm.z, lm.y] for lm in face_lms])
        mapped_pts = self._map_fixed(pts, width, height, scale)

        for pt in mapped_pts:
            cv2.circle(img, tuple(pt), 1, (0, 100, 0), -1)

        self._draw_axis_at_center(img, euler_angles, width, height, mode="side")
        return img

    def render_debug_top(self, face_lms, euler_angles, scale, width=400, height=400):
        img = np.zeros((height, width, 3), dtype=np.uint8)
        # We use the same 'scale' for X as we do for Z.
        # Invert Z to get a true top-down view (Y-axis on screen maps to -Z)
        pts = np.array([[lm.x, -lm.z] for lm in face_lms])
        mapped_pts = self._map_fixed(pts, width, height, scale)

        for pt in mapped_pts:
            cv2.circle(img, tuple(pt), 1, (0, 100, 0), -1)

        self._draw_axis_at_center(img, euler_angles, width, height, mode="top")
        return img

    def _draw_axis_at_center(self, img, euler_angles, width, height, mode="front"):
        """Helper to draw the 3D axis gizmo at the center of a debug pane."""
        origin = (width // 2, height // 2)
        rmat = self._get_rotation_matrix(*euler_angles)
        axis_len = min(width, height) * 0.25
        axes_3d = np.array([[axis_len, 0, 0], [0, axis_len, 0], [0, 0, axis_len]])
        rotated = (rmat @ axes_3d.T).T

        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # BGR
        for i, vec in enumerate(rotated):
            if mode == "front":
                target = (int(origin[0] + vec[0]), int(origin[1] + vec[1]))

            elif mode == "side":
                # vec[2] is Depth (Z), vec[1] is Vertical (Y)
                # We subtract vec[1] to flip the screen's 'down' to be 'up'
                # and use vec[2] to handle horizontal depth
                target = (int(origin[0] - vec[2]), int(origin[1] + vec[1]))

            elif mode == "top":
                target = (int(origin[0] + vec[0]), int(origin[1] - vec[2]))

            cv2.line(img, origin, target, colors[i], 2)

    def show(self, main_frame, f_view, s_view, t_view):
        h, w, _ = main_frame.shape

        # Ensure debug views fit the height of the main window exactly
        debug_h = h // 3
        debug_w = 400

        # Stack vertically
        debug_stack = np.vstack((f_view, s_view, t_view))

        # If there's a slight height mismatch due to integer division, resize the stack
        if debug_stack.shape[0] != h:
            debug_stack = cv2.resize(debug_stack, (debug_w, h))

        combined = np.hstack((main_frame, debug_stack))
        cv2.imshow("Head Tracking Rig Dashboard", combined)

    def render(self, frame, face_lms):
        canvas = frame.copy()
        if face_lms:
            h, w = canvas.shape[:2]
            self.draw_landmarks(canvas, face_lms, w, h)
            self.draw_reference_lines(canvas, face_lms, w, h)
        return canvas
