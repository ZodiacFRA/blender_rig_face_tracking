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

    def modal(self, context, event):
        if event.type == "TIMER":
            try:
                # Drain socket to get the latest packet
                latest_data = None
                while True:
                    data, _ = self._sock.recvfrom(1024)
                    latest_data = data
            except BlockingIOError:
                pass  # No more data in buffer
            except Exception as e:
                print(f"Socket Error: {e}")

            if latest_data:
                print(f"Received: {latest_data}")
                try:
                    msg = json.loads(latest_data.decode("utf-8"))
                    obj = bpy.data.objects.get("mech_head")
                    if obj and obj.type == "ARMATURE":
                        bone = obj.pose.bones.get("root")
                        if bone:
                            # Convert degrees to radians
                            p = math.radians(msg.get("p", 0))
                            y = math.radians(msg.get("y", 0))
                            r = math.radians(msg.get("r", 0))

                            bone.rotation_mode = "XYZ"
                            bone.rotation_euler = (p, -r, y)
                        else:
                            print("Error: Bone 'root' not found in 'mech_head'!")
                    else:
                        print("Error: Object 'mech_head' not found or not an ARMATURE!")
                except json.JSONDecodeError:
                    print("Error: JSON Decode failed.")

        elif event.type == "ESC":
            return self.cancel(context)

        return {"PASS_THROUGH"}

    def execute(self, context):
        print("Starting Face Track Receiver on 127.0.0.1:5005...")
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.bind(("127.0.0.1", 5005))
        self._sock.setblocking(False)

        wm = context.window_manager
        self._timer = wm.event_timer_add(0.016, window=context.window)  # ~60 FPS
        wm.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def cancel(self, context):
        print("Stopping Face Track Receiver...")
        wm = context.window_manager
        wm.event_timer_remove(self._timer)
        if self._sock:
            self._sock.close()
        return {"CANCELLED"}


def register():
    bpy.utils.register_class(FaceTrackReceiver)


def unregister():
    bpy.utils.unregister_class(FaceTrackReceiver)


if __name__ == "__main__":
    register()
    # This line triggers the operator immediately when you click 'Run Script'
    bpy.ops.wm.face_track_receiver()
