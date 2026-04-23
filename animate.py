import bpy
import json
import math
import mathutils
import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument(
    "--capture",
    type=str,
    required=True,
    help="Path to mocap JSON file"
)
argv = sys.argv
if "--" in argv:
    argv = argv[argv.index("--") + 1:]
else:
    argv = []

args = parser.parse_args(argv)

with open(args.capture, "r") as f:
    capture = json.load(f)

with open("armature.json", "r") as f:
    armature = json.load(f)

bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()
bpy.context.scene.render.fps = int(capture["fps"])
bpy.context.scene.frame_start = 1
bpy.context.scene.frame_end = len(capture["landmarks"])
scene = bpy.context.scene
scene.render.resolution_x = int(capture["width"])
scene.render.resolution_y = int(capture["height"])
scene.render.resolution_percentage = 100

bpy.ops.object.camera_add(location=(0, -3, 1.5))
camera = bpy.context.active_object
camera.name = "MocapCamera"
bpy.context.scene.camera = camera
camera.data.lens_unit = 'FOV'
camera.data.angle = math.radians(50)
camera.location = (0, -5, 1.5)
camera.rotation_euler = (math.radians(75), 0, 0)
target = mathutils.Vector((0, 0, 0))
direction = target - camera.location
rot_quat = direction.to_track_quat('-Z', 'Y')
camera.rotation_euler = rot_quat.to_euler()

points = []

for i in range(len(capture["landmarks"][0])):
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.02)
    obj = bpy.context.active_object
    obj.name = f"pt_{i}"
    points.append(obj)

arm_data = bpy.data.armatures.new("Armature")
arm_obj = bpy.data.objects.new("Armature", arm_data)
bpy.context.collection.objects.link(arm_obj)
bpy.context.view_layer.objects.active = arm_obj

bpy.ops.object.mode_set(mode='EDIT')

def make_armature(d, parent=None):
    for name, bone in d.items():
        new_bone = arm_data.edit_bones.new(name)
        new_bone.head = bone["head"]
        new_bone.tail = bone["tail"]
        if parent:
            new_bone.parent = parent
        children = bone.get("children")
        if children:
            make_armature(children, new_bone)

make_armature(armature)

for frame_idx, frame in enumerate(capture["landmarks"]):
    for i, (x, y, z) in enumerate(frame):
        obj = points[i]
        obj.location = (
            x, y, z
        )
        obj.keyframe_insert(
            data_path="location",
            frame=frame_idx + 1
        )

bpy.context.scene.frame_set(1)

for area in bpy.context.screen.areas:
    if area.type == 'VIEW_3D':
        for space in area.spaces:
            if space.type == 'VIEW_3D':
                space.region_3d.view_perspective = 'CAMERA'

bpy.ops.screen.animation_play()
