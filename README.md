# Head Rig Dashboard (WIP)

A real-time head pose estimation and tracking system designed for character rigs and motion capture applications. The end goal is to control a custom robot head rig in Blender via a UDP network stream.
For now only the head global rotation is done

## Features

* **Hybrid Pose Estimation**: Toggle between geometric landmark-based estimation (Stable), Perspective-n-Point (PnP), or a weighted mix of both.
* **Multi-View Visualizer**: Includes a main camera feed with a mesh overlay and three orthographic debug views (Front, Side, Top) for orientation verification.
* **Network Integration**: Dedicated transmission module for syncing pose data with external engines via UDP (Port 5005).
* **Blender Integration**: Custom modal operator script for Blender to apply real-time rotations to an armature.

## Installation

### Prerequisites
* Python 3.13
* Blender 5.0
* Webcam

### Setup
1. Clone the repository:
```bash
git clone https://github.com/ZodiacFRA/blender_rig_face_tracking.git
cd blender_rig_face_tracking
```

2. Install the required dependencies:

```bash
pip install opencv-python numpy mediapipe
```

## Usage

### 1. Start the Tracking Server

Launch the tracker from your terminal. This captures your webcam feed and broadcasts the data.

```bash
python main.py --mode mix --viz
```

* `--mode`: Estimation logic. Options: `stable`, `pnp`, `mix` (Default: `mix`).
* `--viz`: Enable the OpenCV visualization windows.
* Key `q`: Quit.
* Key `c`: Recalibrate face scale.

### 2. Start the Blender Receiver

1. Open your Blender project.
2. Ensure you have an Armature object named **"mech_head"** with a bone named **"root"**.
3. Go to the **Scripting** tab and open `blender.py`.
4. Click **Run Script**.
5. The operator starts automatically. To stop it, press `Esc` while the Blender viewport is active.

## Technical Architecture

1. **Capture**: The `CameraSource` class manages the hardware interface and frame buffering.
2. **Detection**: `LandmarkDetector` utilizes MediaPipe to extract high-fidelity facial landmarks.
3. **Solver**:
* **Stable**: Optimized for reducing jitter in high-frequency movement.
* **PnP**: Uses a 3D-to-2D projection matrix for mathematical accuracy.


4. **Transmission**: `NetworkSender` serializes Euler angles as JSON and broadcasts via UDP.
5. **Receiver (Blender)**: A Modal Operator (`FaceTrackReceiver`) listens on `127.0.0.1:5005`, drains the socket buffer for the latest packet, and updates the bone rotation at ~60 FPS.

## Project Structure

* `main.py`: The `AppController` and primary execution loop.
* `blender.py`: The Blender-side script to receive and apply data.
* `PoseEstimators.py`: Implements the mathematical models for orientation.
* `video.py`: Handles the OpenCV rendering and camera management.
* `utils.py`: Contains helper classes for landmark processing and networking.