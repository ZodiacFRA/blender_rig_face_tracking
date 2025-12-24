from pprint import pprint
import argparse

import cv2
import numpy as np

from video import CameraSource, Visualizer
from PoseEstimators import CustomPoseEstimator, PnPPoseEstimator
from utils import NetworkSender, LandmarkDetector, to_degrees


class AppController:
    def __init__(self, mode, visualize=True):
        # 1. Hardware & Detection
        self.camera = CameraSource()
        self.detector = LandmarkDetector()
        self.visualizer = Visualizer()
        self.sender = NetworkSender()

        # 2. Math Engine
        self.stable_estimator = CustomPoseEstimator()
        self.pnp_estimator = PnPPoseEstimator()
        self.mode = mode
        self.visualize = visualize

        # 3. State & Calibration
        self.is_running = True
        self.camera_matrix = None
        self.dist_coeffs = np.zeros((4, 1))

        # Fixed scale for debug views (Calculated at first frame or 'c')
        self.initial_face_scale = None

    def __del__(self):
        self.cleanup()

    def _init_camera_params(self, w, h):
        """Generates a generic camera matrix based on frame size."""
        focal_length = w
        center = (w / 2, h / 2)
        self.camera_matrix = np.array(
            [[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]],
            dtype=np.float32,
        )

    def _calibrate_scale(self, face_lms):
        """
        Calculates a fixed scale based on the face height (forehead to chin).
        This scale is used for all axes to maintain 1:1 proportions.
        """
        # Landmark 10: Forehead center, Landmark 152: Chin center
        p10 = np.array([face_lms[10].x, face_lms[10].y])
        p152 = np.array([face_lms[152].x, face_lms[152].y])

        face_height = np.linalg.norm(p10 - p152)

        if face_height > 0:
            # We want the head to occupy ~65% of the debug pane height
            self.initial_face_scale = 0.65 / face_height
            print(f"Scale Calibrated: {self.initial_face_scale:.2f}")

    def _get_euler_angles(self, face_lms, w, h):
        """Encapsulates the pose estimation logic."""
        if self.mode == "stable":
            return self.stable_estimator.estimate_pose(face_lms, w, h)
        elif self.mode == "pnp":
            return self.pnp_estimator.estimate_pose(face_lms, w, h)
        elif self.mode == "mix":
            stable = self.stable_estimator.estimate_pose(face_lms, w, h)
            pnp = self.pnp_estimator.estimate_pose(face_lms, w, h)
            alpha = 0.5
            return (stable * (1 - alpha)) + (pnp * alpha)
        return np.zeros(3)

    def _generate_debug_views(self, frame, face_lms, angles, w, h):
        """Handles the heavy lifting of rendering debug views."""
        dw, dh = 400, h // 3
        main = self.visualizer.render(frame.copy(), face_lms)

        f = self.visualizer.render_debug_front(
            face_lms, angles, self.initial_face_scale, dw, dh
        )
        s = self.visualizer.render_debug_side(
            face_lms, angles, self.initial_face_scale, dw, dh
        )
        t = self.visualizer.render_debug_top(
            face_lms, angles, self.initial_face_scale, dw, dh
        )

        return main, f, s, t

    def _handle_ui(self, main_display, views):
        """Displays frames and listens for key events."""
        if views:
            self.visualizer.show(*views)
        else:
            # Fallback for when no landmarks are detected but viz is on
            h, w, _ = main_display.shape
            black = np.zeros((h // 3, 400, 3), dtype=np.uint8)
            self.visualizer.show(main_display, black, black, black)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            self.is_running = False
        elif key == ord("c"):
            self.initial_face_scale = None

    def run(self):
        while self.is_running:
            success, frame = self.camera.get_frame()
            if not success:
                return

            h, w, _ = frame.shape
            if self.camera_matrix is None:
                self._init_camera_params(w, h)

            # 1. Detection
            landmarks_result = self.detector.process_async(frame)

            # 2. Processing & Networking
            views = None
            if landmarks_result and landmarks_result.face_landmarks:

                face_lms = landmarks_result.face_landmarks[0]

                # Calibration check
                if self.initial_face_scale is None:
                    self._calibrate_scale(face_lms)

                # Estimation & Sync
                euler_angles = self._get_euler_angles(face_lms, w, h)

                blendshape_data = {}
                for blendshape in landmarks_result.face_blendshapes[0]:
                    blendshape_data[blendshape.category_name] = blendshape.score
                # Send data
                self.sender.send_pose(
                    {
                        "head_rotation": euler_angles.tolist(),
                        "blendshapes": blendshape_data,
                    }
                )

                # Visualization Data
                if self.visualize:
                    views = self._generate_debug_views(
                        frame, face_lms, euler_angles, w, h
                    )

            # 3. UI Rendering & Input
            if self.visualize:
                self._handle_ui(frame if views is None else views[0], views)

    def cleanup(self):
        self.camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--viz",
        dest="viz",
        default=False,
        action="store_true",
        help="Run in headless mode without visualizations",
    )
    parser.add_argument(
        "--mode",
        dest="mode",
        default="mix",
        choices=["stable", "pnp", "mix"],
        help="Choose pose estimation mode: 'stable', 'pnp', or 'mix'",
    )
    args = parser.parse_args()

    app = AppController(mode=args.mode, visualize=args.viz)
    print(f"--- Head Rig Dashboard [Mode: {args.mode}] ---")
    msg = (
        "Keys: 'q' to Quit | 'c' to Recalibrate"
        if args.viz
        else "Running Headless. Ctrl+C to stop."
    )
    print(msg)
    app.run()
