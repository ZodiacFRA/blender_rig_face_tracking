import bpy
import socket
import json
import math

# Run this script inside Blender's Scripting tab.
# 1. Open Blender
# 2. Go to Scripting tab, open this file or paste content
# 3. Run Script
# 4. Ensure scene has "mech_head" armature. Press F3, search "Face Track Receiver"


class FaceTrackReceiver(bpy.types.Operator):
    """Receive Face Tracking Data via UDP"""

    bl_idname = "wm.face_track_receiver"
    bl_label = "Face Track Receiver"
    bl_options = {"REGISTER"}

    _timer = None
    _sock = None
    TARGET_OBJ = "mech_head"

    # --- Data Handling Logic ---

    def _get_latest_packet(self):
        """Drains the socket buffer to ensure we only process the freshest data."""
        latest_data = None
        try:
            while True:
                data, _ = self._sock.recvfrom(2048)
                latest_data = data
        except BlockingIOError:
            pass
        return latest_data

    def _apply_pose(self, armature, msg):
        """Updates bones based on received rotation and landmarks."""
        # 1. Update Root Rotation
        root = armature.pose.bones.get("root")
        if root:
            root.rotation_mode = "XYZ"
            euler_angles = msg.get("head_rotation", [0, 0, 0])
            # Extract pitch, yaw, roll
            pitch = euler_angles[0]
            yaw = euler_angles[1]
            roll = euler_angles[2]
            root.rotation_euler = (pitch, -roll, yaw)

        # 2. Update Landmarks
        landmarks = msg.get("landmarks_positions", {})
        for idx, (lx, ly, lz) in landmarks.items():
            print(idx, (lx, ly, lz))
            lm_bone = armature.pose.bones.get(f"lm_{idx}")
            if lm_bone:
                # Coordinate Mapping: MP(x,y,z) -> Blender(x, -z, -y)
                lm_bone.location = (lx, -lz, -ly)

    # --- Modal Logic ---

    def modal(self, context, event):
        if event.type == "ESC":
            return self.cancel(context)

        if event.type == "TIMER":
            raw_data = self._get_latest_packet()
            if not raw_data:
                return {"PASS_THROUGH"}

            try:
                msg = json.loads(raw_data.decode("utf-8"))
                obj = bpy.data.objects.get(self.TARGET_OBJ)

                if obj and obj.type == "ARMATURE":
                    self._apply_pose(obj, msg)
                else:
                    self.report({"WARNING"}, f"Target '{self.TARGET_OBJ}' not found.")

            except (json.JSONDecodeError, Exception) as e:
                print(f"Update Error: {e}")

        return {"PASS_THROUGH"}

    # --- Lifecycle Methods ---

    def execute(self, context):
        addr = ("127.0.0.1", 5005)
        print(f"Starting Face Track Receiver on {addr[0]}:{addr[1]}...")

        try:
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._sock.bind(addr)
            self._sock.setblocking(False)
        except Exception as e:
            self.report({"ERROR"}, f"Could not bind socket: {e}")
            return {"CANCELLED"}

        wm = context.window_manager
        self._timer = wm.event_timer_add(0.016, window=context.window)
        wm.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def cancel(self, context):
        if self._timer:
            context.window_manager.event_timer_remove(self._timer)
        if self._sock:
            self._sock.close()
        print("Stopped Face Track Receiver.")
        return {"CANCELLED"}


def register():
    bpy.utils.register_class(FaceTrackReceiver)


def unregister():
    bpy.utils.unregister_class(FaceTrackReceiver)


if __name__ == "__main__":
    register()
    bpy.ops.wm.face_track_receiver()
