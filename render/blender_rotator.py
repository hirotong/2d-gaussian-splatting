"""Blender script to render images of 3D models.

This script is used to render images of 3D models. It takes in a list of paths
to .glb files and renders images of each model. The images are from rotating the
object around the origin. The images are saved to the output directory.

Example usage:
    blender -b -P blender_script.py -- \
        --object_path my_object.glb \
        --output_dir ./views \
        --engine CYCLES \
        --scale 0.8 \
        --num_images 12 \
        --camera_dist 1.2

Here, input_model_paths.json is a json file containing a list of paths to .glb.
"""

import argparse
import json
import math
import os
import random
import sys
import time
import urllib.request
import uuid
from typing import Tuple

from mathutils import Vector, Matrix
import numpy as np

import bpy


def parent_obj_to_camera(b_camera):
    origin = (0, 0, 0)
    b_empty = bpy.data.objects.new("Empty", None)
    b_empty.location = origin
    b_camera.parent = b_empty  # setup parenting

    scn = bpy.context.scene
    scn.collection.objects.link(b_empty)
    bpy.context.view_layer.objects.active = b_empty
    # scn.objects.active = b_empty
    return b_empty


parser = argparse.ArgumentParser()
parser.add_argument(
    "--object_path",
    type=str,
    required=True,
    help="Path to the object file",
)
parser.add_argument("--output_dir", type=str, default="~/.objaverse/hf-objaverse-v1/views_whole_sphere")
parser.add_argument("--engine", type=str, default="CYCLES", choices=["CYCLES", "BLENDER_EEVEE"])
parser.add_argument("--scale", type=float, default=0.8)
parser.add_argument("--num_images", type=int, default=8)
parser.add_argument("--camera_dist", type=float, default=1.2)
parser.add_argument("--image_size", type=int, default=800)

argv = sys.argv[sys.argv.index("--") + 1 :]
args = parser.parse_args(argv)

print("===================", args.engine, "===================")

camera_dist = args.camera_dist
camera_azimth = random.randint(-60, 60)  # math.radians(random.randint(0, 360))
elev_range = [-30, 30]
N_elevs = 5
camera_elevs = np.linspace(elev_range[0], elev_range[1], N_elevs)
context = bpy.context
scene = context.scene
render = scene.render

cam = scene.objects["Camera"]
cam.location = (0, -camera_dist, 0)

cam_constraint = cam.constraints.new(type="TRACK_TO")
cam_constraint.track_axis = "TRACK_NEGATIVE_Z"
cam_constraint.up_axis = "UP_Y"

# virtual_cam = bpy.data.objects.new("Empty", None)
# scene.collection.objects.link(virtual_cam)
# virtual_cam_constraint = virtual_cam.constraints.new(type="TRACK_TO")
# virtual_cam_constraint.track_axis = "TRACK_NEGATIVE_Z"
# virtual_cam_constraint.up_axis = "UP_Y"

# setup lighting

bpy.ops.object.light_add(type="POINT")
# light2 = bpy.data.lights["Point.001"]
# light2.shadow_soft_size = 0.0
# light2_object = scene.objects["Point.001"]
# light2_constraint = light2_object.constraints.new(type="TRACK_TO")
# light2_constraint.track_axis = "TRACK_NEGATIVE_Z"
# light2_constraint.up_axis = "UP_Y"

render.engine = args.engine
render.image_settings.file_format = "PNG"
render.image_settings.color_mode = "RGBA"
render.resolution_x = args.image_size
render.resolution_y = args.image_size
render.resolution_percentage = 100

scene.cycles.device = "GPU"
scene.cycles.samples = 128
scene.cycles.diffuse_bounces = 1
scene.cycles.glossy_bounces = 1
scene.cycles.transparent_max_bounces = 3
scene.cycles.transmission_bounces = 3
scene.cycles.filter_width = 0.01
scene.cycles.use_denoising = True
scene.render.film_transparent = True

# enable depth and normal rendering
scene.use_nodes = True
scene.view_layers["ViewLayer"].use_pass_normal = True
scene.view_layers["ViewLayer"].use_pass_object_index = True
scene.view_layers["ViewLayer"].use_pass_diffuse_color = True
scene.view_layers["ViewLayer"].use_pass_z = True
nodes = bpy.context.scene.node_tree.nodes
links = bpy.context.scene.node_tree.links

# clear default nodes
for n in nodes:
    nodes.remove(n)

# Create input render layer node
render_layers = nodes.new("CompositorNodeRLayers")

# Create depth output nodes
depth_format = "OPEN_EXR"
depth_file_output = nodes.new(type="CompositorNodeOutputFile")
depth_file_output.label = "Depth Output"
depth_file_output.base_path = ""
depth_file_output.file_slots[0].use_node_format = True
depth_file_output.format.file_format = depth_format
depth_file_output.format.color_depth = "16"
if depth_format == "OPEN_EXR":
    links.new(render_layers.outputs["Depth"], depth_file_output.inputs[0])
else:
    depth_file_output.format.color_mode = "BW"

    # remap as other types can't represent the full range of depth
    map = nodes.new(type="CompositorNodeMapValue")
    # Size is chosen kind of arbitrarily, try out until you're satisfied with resulting depth map.
    map.offset = [-0.7]
    map.size = [1.0]
    map.use_min = True
    map.min = [0]

    links.new(render_layers.outputs["Depth"], map.inputs[0])
    links.new(map.outputs[0], depth_file_output.inputs[0])

# Create normal output nodes
scale_node = nodes.new(type="CompositorNodeMixRGB")
scale_node.blend_type = "MULTIPLY"
# scale_node.use_alpha = True
scale_node.inputs[2].default_value = (0.5, 0.5, 0.5, 1)
links.new(render_layers.outputs["Normal"], scale_node.inputs[1])

bias_node = nodes.new(type="CompositorNodeMixRGB")
bias_node.blend_type = "ADD"
# bias_node.use_alpha = True
bias_node.inputs[2].default_value = (0.5, 0.5, 0.5, 0)
links.new(scale_node.outputs[0], bias_node.inputs[1])

normal_file_output = nodes.new(type="CompositorNodeOutputFile")
normal_file_output.label = "Normal Output"
normal_file_output.base_path = ""
normal_file_output.file_slots[0].use_node_format = True
normal_file_output.format.file_format = "PNG"
convert_node = nodes.new(type="CompositorNodeConvertColorSpace")
# convert_node.use_transform = True
convert_node.from_color_space = "sRGB"
convert_node.to_color_space = "Linear Rec.709"

links.new(bias_node.outputs[0], convert_node.inputs[0])
links.new(convert_node.outputs[0], normal_file_output.inputs[0])

# object mask
id_mask_node = nodes.new(type="CompositorNodeIDMask")
id_mask_node.index = 1
id_mask_node.use_antialiasing = False
out_node = nodes.new(type="CompositorNodeComposite")
links.new(render_layers.outputs["Image"], out_node.inputs[0])
links.new(render_layers.outputs["IndexOB"], id_mask_node.inputs[0])

mask_file_output = nodes.new(type="CompositorNodeOutputFile")
mask_file_output.label = "Mask Output"
mask_file_output.base_path = ""
mask_file_output.file_slots[0].use_node_format = True
mask_file_output.format.file_format = "PNG"
mask_file_output.format.color_mode = "BW"
links.new(id_mask_node.outputs[0], mask_file_output.inputs[0])
links.new(id_mask_node.outputs[0], out_node.inputs[1])

bpy.context.preferences.addons["cycles"].preferences.get_devices()
# Set the device_type
try:
    bpy.context.preferences.addons["cycles"].preferences.compute_device_type = "CUDA"  # "CUDA"  # or "OPENCL"
except:
    bpy.context.preferences.addons["cycles"].preferences.compute_device_type = "METAL"  # "CUDA"  # or "OPENCL"
out_data = {
    "camera_angle_x": bpy.data.objects["Camera"].data.angle_x,
}


def sample_point_on_sphere(radius: float) -> Tuple[float, float, float]:
    theta = random.random() * 2 * math.pi
    phi = math.acos(2 * random.random() - 1)
    return (
        radius * math.sin(phi) * math.cos(theta),
        radius * math.sin(phi) * math.sin(theta),
        radius * math.cos(phi),
    )


def sample_spherical(radius=3.0, maxz=3.0, minz=0.0):
    correct = False
    while not correct:
        vec = np.random.uniform(-1, 1, 3)
        vec[2] = np.abs(vec[2])
        vec = vec / np.linalg.norm(vec, axis=0) * radius
        if maxz > vec[2] > minz:
            correct = True
    return vec


def sample_spherical(radius_min=1.5, radius_max=2.0, maxz=1.6, minz=-0.75):
    correct = False
    while not correct:
        vec = np.random.uniform(-1, 1, 3)
        #         vec[2] = np.abs(vec[2])
        radius = np.random.uniform(radius_min, radius_max, 1)
        vec = vec / np.linalg.norm(vec, axis=0) * radius[0]
        if maxz > vec[2] > minz:
            correct = True
    return vec


def fixed_camera_pose():
    elevation = np.array([0] * 6, dtype=float)
    azimuth = np.array([-90, -45, 0, 90, 180, -135], dtype=float)
    radii = np.array([1.3] * 6, dtype=float)
    view_types = [
        "front",
        "front_right",
        "right",
        "back",
        "left",
        "front_left",
    ]
    for el, az, ra, vt in zip(elevation, azimuth, radii, view_types):
        yield el, az, ra, vt


def randomize_camera():
    elevation = random.uniform(-40.0, 40.0)
    azimuth = random.uniform(-180, 180)
    distance = random.uniform(1.2, 1.6)
    return set_camera_location(elevation, azimuth, distance)


def spherical_to_cartesian(elevation, azimuth, distance):
    x = distance * math.cos(math.radians(elevation)) * math.cos(math.radians(azimuth))
    y = distance * math.cos(math.radians(elevation)) * math.sin(math.radians(azimuth))
    z = -distance * math.sin(math.radians(elevation))
    return x, y, z


def set_camera_location(elevation, azimuth, distance):
    # from https://blender.stackexchange.com/questions/18530/
    x, y, z = spherical_to_cartesian(elevation, azimuth, distance)

    camera = bpy.data.objects["Camera"]
    camera.location = x, y, z

    direction = -camera.location
    rot_quat = direction.to_track_quat("-Z", "Y")
    camera.rotation_euler = rot_quat.to_euler()
    return camera


def randomize_lighting(energy=500) -> None:
    light2.energy = energy  # random.uniform(200, 500)
    azim = camera_azimth + 5 * 2 * (random.random() - 0.5)
    elev = camera_elevs[0] + 5 * 2 * (random.random() - 0.5)
    radius = camera_dist + 0.5 * 2 * (random.random() - 0.5)

    b_empty = bpy.data.objects.new("Empty", None)
    b_empty.location = (0, 0, 0)
    light2_object.parent = b_empty  # setup parenting

    scn = bpy.context.scene
    scn.collection.objects.link(b_empty)
    bpy.context.view_layer.objects.active = b_empty

    # debug
    print("Azim: {}, Elev: {}, Radius: {}".format(azim, elev, radius))

    # x, y, z = spherical_to_cartesian(elev, azim, radius)
    light2_object.location = (0, -radius, 0)
    b_empty.rotation_euler = (math.radians(elev), 0, math.radians(azim))

    bpy.context.view_layer.update()

    # bpy.data.objects["Area"].location[0] = x    # random.uniform(-1.0, 1.0)
    # bpy.data.objects["Area"].location[1] = y    # random.uniform(-1.0, 1.0)
    # bpy.data.objects["Area"].location[2] = z    # random.uniform(3, 4)
    # also reset the scales after normalizing the scene, otherwise will cause problem with large or small objects.
    # bpy.data.objects["Point"].scale[0] = 2
    # bpy.data.objects["Point"].scale[1] = 2
    # bpy.data.objects["Point"].scale[2] = 2


def reset_lighting() -> None:
    light2.energy = 1000
    bpy.data.objects["Area"].location[0] = 0
    bpy.data.objects["Area"].location[1] = 0
    bpy.data.objects["Area"].location[2] = 0.5


def reset_scene() -> None:
    """Resets the scene to a clean state."""
    # delete everything that isn't part of a camera or a light
    for obj in bpy.data.objects:
        if obj.type not in {"CAMERA", "LIGHT"}:
            bpy.data.objects.remove(obj, do_unlink=True)
    # delete all the materials
    for material in bpy.data.materials:
        bpy.data.materials.remove(material, do_unlink=True)
    # delete all the textures
    for texture in bpy.data.textures:
        bpy.data.textures.remove(texture, do_unlink=True)
    # delete all the images
    for image in bpy.data.images:
        bpy.data.images.remove(image, do_unlink=True)


# load the glb model
def load_object(object_path: str) -> None:
    """Loads a glb model into the scene."""
    if object_path.endswith(".glb"):
        bpy.ops.import_scene.gltf(filepath=object_path, merge_vertices=True)
    elif object_path.endswith(".fbx"):
        bpy.ops.import_scene.fbx(filepath=object_path)
    else:
        raise ValueError(f"Unsupported file type: {object_path}")


def object_bbox(object, ignore_matrix=False):
    bbox_min = (math.inf,) * 3
    bbox_max = (-math.inf,) * 3
    found = False

    for obj in object.children_recursive:
        if isinstance(obj.data, bpy.types.Mesh):
            found = True
            for coord in obj.bound_box:
                coord = Vector(coord)
                if not ignore_matrix:
                    coord = obj.matrix_world @ coord
                bbox_min = tuple(min(x, y) for x, y in zip(bbox_min, coord))
                bbox_max = tuple(max(x, y) for x, y in zip(bbox_max, coord))
    if not found:
        raise RuntimeError("no objects in scene to compute bounding box for")
    return Vector(bbox_min), Vector(bbox_max)


def scene_bbox(single_obj=None, ignore_matrix=False):
    bbox_min = (math.inf,) * 3
    bbox_max = (-math.inf,) * 3
    found = False
    for obj in scene_meshes() if single_obj is None else [single_obj]:
        found = True
        for coord in obj.bound_box:
            coord = Vector(coord)
            if not ignore_matrix:
                coord = obj.matrix_world @ coord
            bbox_min = tuple(min(x, y) for x, y in zip(bbox_min, coord))
            bbox_max = tuple(max(x, y) for x, y in zip(bbox_max, coord))
    if not found:
        raise RuntimeError("no objects in scene to compute bounding box for")
    return Vector(bbox_min), Vector(bbox_max)


def scene_root_objects():
    for obj in bpy.context.scene.objects.values():
        if not obj.parent:
            yield obj


def scene_meshes():
    for obj in bpy.context.scene.objects.values():
        if isinstance(obj.data, (bpy.types.Mesh)):
            yield obj


# function from https://github.com/panmari/stanford-shapenet-renderer/blob/master/render_blender.py
def get_3x4_RT_matrix_from_blender(cam):
    # bcam stands for blender camera
    # R_bcam2cv = Matrix(
    #     ((1, 0,  0),
    #     (0, 1, 0),
    #     (0, 0, 1)))

    # Transpose since the rotation is object rotation,
    # and we want coordinate rotation
    # R_world2bcam = cam.rotation_euler.to_matrix().transposed()
    # T_world2bcam = -1*R_world2bcam @ location
    #
    # Use matrix_world instead to account for all constraints
    location, rotation = cam.matrix_world.decompose()[0:2]
    R_world2bcam = rotation.to_matrix().transposed()

    # Convert camera location to translation vector used in coordinate changes
    # T_world2bcam = -1*R_world2bcam @ cam.location
    # Use location from matrix_world to account for constraints:
    T_world2bcam = -1 * R_world2bcam @ location

    # # Build the coordinate transform matrix from world to computer vision camera
    # R_world2cv = R_bcam2cv@R_world2bcam
    # T_world2cv = R_bcam2cv@T_world2bcam

    # put into 3x4 matrix
    RT = Matrix(
        (
            R_world2bcam[0][:] + (T_world2bcam[0],),
            R_world2bcam[1][:] + (T_world2bcam[1],),
            R_world2bcam[2][:] + (T_world2bcam[2],),
        )
    )
    return RT


def get_matrix_world(cam):
    location = cam.matrix_world.translation
    rotation = cam.matrix_world.to_quaternion()

    rotation_matrix = rotation.to_matrix().to_4x4()

    translation_matrix = Matrix.Translation(location)

    matrix_without_scaling = translation_matrix @ rotation_matrix
    return matrix_without_scaling


def normalize_scene(single_obj=None):
    bbox_min, bbox_max = object_bbox(single_obj)
    print(bbox_min, bbox_max)
    scale = 1.8 / max(bbox_max - bbox_min)
    print("Scale", scale)
    if single_obj is not None:
        single_obj.scale = single_obj.scale * scale
    else:
        for obj in scene_root_objects():
            if not isinstance(obj.data, bpy.types.Light):
                obj.scale = obj.scale * scale
    # Apply scale to matrix_world.
    bpy.context.view_layer.update()
    bbox_min, bbox_max = object_bbox(single_obj)
    offset = -(bbox_min + bbox_max) / 2
    print("Offset", offset)
    # translate the object to the center of the scene
    if single_obj is not None:
        single_obj.matrix_world.translation += offset
    else:
        for obj in scene_root_objects():
            if not isinstance(obj.data, bpy.types.Light):
                obj.matrix_world.translation += offset
    bpy.ops.object.select_all(action="DESELECT")


def listify_matrix(matrix):
    matrix_list = []
    for row in matrix:
        matrix_list.append(list(row))
    return matrix_list


def subdivide_object(obj, levels=1):
    if isinstance(obj.data, bpy.types.Mesh):
        mod = obj.modifiers.new(name="subdivision", type="SUBSURF")
        mod.levels = levels
        mod.render_levels = levels
        mod.subdivision_type = "SIMPLE"

        mod_2 = obj.modifiers.new(name="subdivision", type="SUBSURF")
        mod_2.levels = levels
        mod_2.render_levels = levels
        mod_2.subdivision_type = "CATMULL_CLARK"

    for child in obj.children_recursive:
        if isinstance(child.data, bpy.types.Mesh):
            mod = child.modifiers.new(name="subdivision", type="SUBSURF")
            mod.levels = levels
            mod.render_levels = levels
            mod.subdivision_type = "SIMPLE"

            mod_2 = child.modifiers.new(name="subdivision", type="SUBSURF")
            mod_2.levels = levels
            mod_2.render_levels = levels
            mod_2.subdivision_type = "CATMULL_CLARK"


def render_envmap(path):
    hdr_cam_data = bpy.data.cameras.new("hdr_cam")
    hdr_cam_obj = bpy.data.objects.new("hdr_cam", hdr_cam_data)

    hdr_cam_obj.location = (0, 0, 0)
    hdr_cam_obj.rotation_euler = (math.radians(90), 0, 0)

    # set camera properties
    hdr_cam_data.type = "PANO"
    hdr_cam_data.panorama_type = "EQUIRECTANGULAR"

    bpy.context.collection.objects.link(hdr_cam_obj)

    # set the camera as the active camera for the scene
    bpy.context.scene.camera = hdr_cam_obj

    # change render settings
    scene = bpy.context.scene
    scene.render.resolution_x = 2048
    scene.render.resolution_y = 1024
    # scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.file_format = "OPEN_EXR"
    scene.render.filepath = os.path.join(path, "envmap.exr")

    depth_file_output.mute = True
    normal_file_output.mute = True
    mask_file_output.mute = True
    # hide the target_object

    bpy.ops.render.render(write_still=True)

    depth_file_output.mute = False
    normal_file_output.mute = False
    mask_file_output.mute = False


def save_images(object_file: str) -> None:
    """Saves rendered images of the object in the scene."""
    os.makedirs(args.output_dir, exist_ok=True)

    # reset_scene()

    # load the object
    load_object(object_file)
    object_uid = os.path.basename(object_file).split(".")[0]
    target_object = bpy.context.scene.objects["Sketchfab_model"]
    normalize_scene(target_object)
    # target_object.location = (0, 0, 0)
    # if object_uid in ["sony", "pigeon"]:
    #     subdivide_object(target_object, 2)
    # elif object_uid in ["pine_cone"]:
    #     subdivide_object(target_object, 1)

    target_object.pass_index = 1
    for obj in target_object.children_recursive:
        obj.pass_index = 1

    # create an empty object to track
    cam_empty = bpy.data.objects.new("Empty", None)
    scene.collection.objects.link(cam_empty)
    cam_constraint.target = cam_empty
    cam.parent = cam_empty
    cam.location = (0, -camera_dist, 0)

    # light2_constraint.target = cam_empty

    obj_empty = bpy.data.objects.new("Empty", None)
    scene.collection.objects.link(obj_empty)
    obj_empty.parent = target_object
    obj_empty.matrix_parent_inverse = target_object.matrix_world.inverted()

    # randomize_lighting(0)

    # print(light2_object.matrix_world.translation)

    # out_data["lighting"] = {"location": list(light2_object.matrix_world.translation), "energy": light2.energy}
    out_data["frames"] = []

    stepsize = 360 / (args.num_images // N_elevs)
    target_object.rotation_mode = "XYZ"

    for n in range(N_elevs):
        camera_elev = math.radians(camera_elevs[n])
        cam_empty.rotation_euler = (camera_elev, 0, math.radians(camera_azimth))
        for i in range(args.num_images // N_elevs):
            # # set the camera position
            # theta = (i / args.num_images) * math.pi * 2
            # phi = math.radians(60)
            # point = (
            #     args.camera_dist * math.sin(phi) * math.cos(theta),
            #     args.camera_dist * math.sin(phi) * math.sin(theta),
            #     args.camera_dist * math.cos(phi),
            # )
            # # reset_lighting()
            # cam.location = point
            print("Rotation {}, {}".format((stepsize * i), math.radians(stepsize * i)))
            # set camera
            # camera = randomize_camera()

            # render the image
            render_path = os.path.join(args.output_dir, object_uid, f"{n:02d}_{i:03d}")
            scene.render.filepath = render_path
            depth_file_output.base_path = os.path.join(args.output_dir, object_uid)
            normal_file_output.base_path = os.path.join(args.output_dir, object_uid)
            mask_file_output.base_path = os.path.join(args.output_dir, object_uid)
            depth_file_output.file_slots[0].path = f"{n:02d}_{i:03d}_depth"
            normal_file_output.file_slots[0].path = f"{n:02d}_{i:03d}_normal"
            mask_file_output.file_slots[0].path = f"{n:02d}_{i:03d}_mask"

            bpy.ops.render.render(write_still=True)

            # save camera RT matrix
            # RT = get_3x4_RT_matrix_from_blender(camera)
            # RT_path = os.path.join(args.output_dir, object_uid, f"{i:03d}.npy")
            # np.save(RT_path, RT)
            # np.savetxt(RT_path.replace(".npy", ".txt"), RT, delimiter=" ")

            frame_data = {
                "file_path": f"{n:02d}_{i:03d}",
                "elevation": math.degrees(camera_elev),
                "rotation": math.radians(stepsize * i),
                "object": listify_matrix(get_matrix_world(obj_empty)),
                "transform_matrix": listify_matrix(get_matrix_world(cam)),
            }
            out_data["frames"].append(frame_data)

            target_object.rotation_euler[2] += math.radians(stepsize)

    with open(os.path.join(args.output_dir, object_uid, "transforms.json"), "w") as f:
        json.dump(out_data, f, indent=4)

    hide_object(target_object)

    render_envmap(os.path.join(args.output_dir, object_uid))

    show_object(target_object)

    # for debugging the workflow
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(os.path.abspath(args.output_dir), object_uid, "scene.blend"))

    target_object.select_set(True)
    for obj in target_object.children_recursive:
        if isinstance(obj.data, bpy.types.Mesh):
            obj.select_set(True)
    # export obj mesh of the target object as ground truth
    export_options = {
        "export_selected_objects": True,  # Only export selected objects
        "forward_axis": "Y",  # Forward is along the Y axis
        "up_axis": "Z",  # Up is along the Z axis
        "export_triangulated_mesh": True,  # Triangulate mesh
    }
    bpy.ops.wm.obj_export(
        filepath=os.path.join(os.path.join(args.output_dir), object_uid, "mesh_gt.obj"), **export_options
    )


def show_object(obj):
    obj.hide_render = False
    obj.hide_viewport = False
    for child in obj.children_recursive:
        child.hide_render = False
        child.hide_viewport = False


def hide_object(obj):
    obj.hide_render = True
    obj.hide_viewport = True
    for child in obj.children_recursive:
        child.hide_render = True
        child.hide_viewport = True


def download_object(object_url: str) -> str:
    """Download the object and return the path."""
    # uid = uuid.uuid4()
    uid = object_url.split("/")[-1].split(".")[0]
    tmp_local_path = os.path.join("tmp-objects", f"{uid}.glb" + ".tmp")
    local_path = os.path.join("tmp-objects", f"{uid}.glb")
    # wget the file and put it in local_path
    os.makedirs(os.path.dirname(tmp_local_path), exist_ok=True)
    urllib.request.urlretrieve(object_url, tmp_local_path)
    os.rename(tmp_local_path, local_path)
    # get the absolute path
    local_path = os.path.abspath(local_path)
    return local_path


if __name__ == "__main__":
    try:
        start_i = time.time()
        if args.object_path.startswith("http"):
            local_path = download_object(args.object_path)
        else:
            local_path = args.object_path
        save_images(local_path)
        end_i = time.time()
        print("Finished", local_path, "in", end_i - start_i, "seconds")
        # delete the object if it was downloaded
        if args.object_path.startswith("http"):
            os.remove(local_path)
    except Exception as e:
        print("Failed to render", args.object_path)
        print(e)
