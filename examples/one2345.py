import blenderproc as bproc
import os
import time
import math
import argparse
import numpy as np
import blenderproc.python.renderer.RendererUtility as RendererUtility
from blenderproc.scripts.saveAsImg import save_array_as_image

"""Blender script to render images of 3D models.

This script is used to render images of 3D models. It takes in a list of paths
to .glb files and renders images of each model. The images are from rotating the
object around the origin. The images are saved to the output directory.
"""

import bpy
from mathutils import Vector, Matrix

parser = argparse.ArgumentParser()
parser.add_argument(
    "--object_path",
    type=str,
    required=True,
    help="Path to the object file",
)

parser.add_argument(
    "--sub_object_path",
    type=str,
    # required=True,
    help="Path to the object file",
)
parser.add_argument("--output_dir", type=str, default="./views")
parser.add_argument(
    "--engine", type=str, default="CYCLES", choices=["CYCLES", "BLENDER_EEVEE"]
)
parser.add_argument("--camera_dist", type=float, default=3.5)
parser.add_argument("--resolution", type=int, default=512)

args = parser.parse_args()

bproc.init()

context = bpy.context
scene = context.scene
render = scene.render

render.engine = args.engine
render.image_settings.file_format = "PNG"
render.image_settings.color_mode = "RGBA"
render.resolution_x = 512
render.resolution_y = 512
render.resolution_percentage = 100

scene.cycles.device = "GPU"
scene.cycles.samples = 32
scene.cycles.diffuse_bounces = 1
scene.cycles.glossy_bounces = 1
scene.cycles.transparent_max_bounces = 3
scene.cycles.transmission_bounces = 3
scene.cycles.filter_width = 0.01
scene.cycles.use_denoising = True
scene.render.film_transparent = True
# scene.view_settings.view_transform = "Standard"
print("display device", scene.display_settings.display_device)
print("view type", scene.view_settings.view_transform)
print("view exposure", scene.view_settings.exposure)
print("view gamma", scene.view_settings.gamma)

def add_lighting() -> None:
    # add a new light
    bpy.ops.object.light_add(type="AREA")
    light2 = bpy.data.lights["Area"]
    light2.energy = 70000
    bpy.data.objects["Area"].location[2] = 0.5
    bpy.data.objects["Area"].scale[0] = 100
    bpy.data.objects["Area"].scale[1] = 100
    bpy.data.objects["Area"].scale[2] = 100


# load the glb model
def load_object(object_path: str) -> None:
    """Loads a glb model into the scene."""
    if object_path.endswith(".glb"):
        bpy.ops.import_scene.gltf(filepath=object_path, merge_vertices=True)
    elif object_path.endswith(".fbx"):
        bpy.ops.import_scene.fbx(filepath=object_path)
    elif object_path.endswith(".obj"):
        bpy.ops.wm.obj_import(filepath=object_path)
        mesh = bpy.context.active_object
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.object.material_slot_add()
        mat_slot = mesh.material_slots[0]
        # Create a new material and set up its shader nodes
        mat = bpy.data.materials.new(name="Vertex")
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        nodes.clear()
        output_node = nodes.new(type='ShaderNodeOutputMaterial')
        bsdf_node = nodes.new(type='ShaderNodeBsdfPrincipled')
        attr_node = nodes.new(type='ShaderNodeAttribute')
        attr_node.attribute_name = "Color"  # Replace with the name of your vertex color layer
        # Connect the nodes
        links.new(bsdf_node.outputs['BSDF'], output_node.inputs['Surface'])
        links.new(attr_node.outputs['Color'], bsdf_node.inputs['Base Color'])
        # Assign the material to the object
        mat_slot.material = mat
        # Switch back to object mode and deselect everything
        bpy.ops.object.mode_set(mode='OBJECT')
    else:
        raise ValueError(f"Unsupported file type: {object_path}")


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


def scene_bbox_meshobj(meshobjs: bproc.types.MeshObject):
    bboxs = np.array([obj.get_bound_box() for obj in meshobjs])
    bboxs = bboxs.reshape(-1, 3)
    min_bbox = bboxs.min(axis=0)
    max_bbox = bboxs.max(axis=0)
    return Vector(min_bbox), Vector(max_bbox)

def scene_root_objects():
    for obj in bpy.context.scene.objects.values():
        if not obj.parent:
            yield obj

def find_root_obj(objs):
    root = []
    for obj in objs:
        root.append(find_root(obj))
    return root

def find_root (obj):
    parent = obj
    while parent.parent:
        parent = parent.parent
    return parent

def scene_meshes():
    for obj in bpy.context.scene.objects.values():
        if isinstance(obj.data, (bpy.types.Mesh)):
            yield obj

def find_root_meshobj(meshobj: bproc.types.MeshObject):
    parent = meshobj
    while parent.get_parent():
        parent = parent.get_parent()
    return parent



def normalize_scene(format="glb"):
    bbox_min, bbox_max = scene_bbox()
    scale = 1 / max(bbox_max - bbox_min)
    for obj in scene_root_objects():
        obj.scale = obj.scale * scale * 0.8
    # Apply scale to matrix_world.
    bpy.context.view_layer.update()
    bbox_min, bbox_max = scene_bbox()
    offset = -(bbox_min + bbox_max) / 2
    for obj in scene_root_objects():
        obj.matrix_world.translation += offset
    bpy.ops.object.select_all(action="DESELECT")
    
def normalize_meshobj(meshobjs, translation = None):
    bbox_min, bbox_max = scene_bbox_meshobj(meshobjs)
    scale = 1 / max(bbox_max.xy - bbox_min.xy)
    root_meshobj = list(set([find_root_meshobj(obj) for obj in meshobjs]))[0]
    scale_vec = Vector((scale, scale, scale)) * 0.6
    root_meshobj.set_scale(scale_vec)
    bbox_min, bbox_max = scene_bbox_meshobj(meshobjs)
    offset = -(bbox_min + bbox_max) / 2
    if not translation:
        root_meshobj.set_location(offset)
    else:
        root_meshobj.set_location(offset + translation)

   

def normalize_scene_offset(format="glb"):
    bbox_min, bbox_max = scene_bbox()
    scale = 1 / max(bbox_max - bbox_min)
    for obj in scene_root_objects():
        obj.scale = obj.scale * scale * 0.8
    # Apply scale to matrix_world.
    bpy.context.view_layer.update()
    bbox_min, bbox_max = scene_bbox()
    offset = -(bbox_min + bbox_max) / 2
    for obj in scene_root_objects():
        obj.matrix_world.translation += (offset  +Vector((0.5, 0. ,0.)))
    bpy.ops.object.select_all(action="DESELECT")

def normalize_scene_obj(obj, format="glb", ):
    bbox_min, bbox_max = scene_bbox(obj)
    scale = 1 / max(bbox_max - bbox_min)
    obj.scale = obj.scale * scale * 0.8
    # Apply scale to matrix_world.
    bpy.context.view_layer.update()
    bbox_min, bbox_max = scene_bbox()
    offset = -(bbox_min + bbox_max) / 2
    obj.matrix_world.translation += offset
    bpy.ops.object.select_all(action="DESELECT")


def compose_RT(R, T):
    return np.hstack((R, T.reshape(-1, 1)))



def setup_camera():
    cam = scene.objects["Camera"]
    cam.location = (0, args.camera_dist, 0)
    cam.data.lens = 35
    cam.data.sensor_width = 32
    cam_constraint = cam.constraints.new(type="TRACK_TO")
    cam_constraint.track_axis = "TRACK_NEGATIVE_Z"
    cam_constraint.up_axis = "UP_Y"
    return cam, cam_constraint

def sample_camera_loc(phi=None, theta=None, r=1.0):
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    return np.array([x, y, z])

def save_images(object_file: str, subobject_file : str = None) -> None:
    """Saves rendered images of the object in the scene."""
    os.makedirs(args.output_dir, exist_ok=True)
    obj_folder = object_file.split("/")[-2:]
    obj_folder = obj_folder + subobject_file.split("/")[-2:] if subobject_file else obj_folder
    obj_folder = [f[:-4] if f.endswith(".glb") else f for f in obj_folder]
    obj_folder = "_".join(obj_folder)
    output_dir = os.path.join(args.output_dir, obj_folder)
    os.makedirs(output_dir, exist_ok=True)
    # load the object
    # load_object(object_file)
    first_obj = bproc.loader.load_obj(object_file)
    first_obj_root = list(set([find_root_meshobj(obj) for obj in first_obj]))
    # bproc.utility.reset_keyframes()
    object_uid = os.path.basename(object_file).split(".")[0]
    object_format = os.path.basename(object_file).split(".")[-1]

    trans = Vector([0.6, 0, 0])
    normalize_meshobj(first_obj, translation = trans)
    
    
    if subobject_file:
        
        second_obj = bproc.loader.load_obj(subobject_file)
        second_obj_root = list(set([find_root_meshobj(obj) for obj in second_obj]))
        
        normalize_meshobj(second_obj)   
        # load_object(subobject_file)
        # subobject_uid = os.path.basename(subobject_file).split(".")[0]
        # subobject_format = os.path.basename(subobject_file).split(".")[-1]
        # roots = get_all_blender_mesh_objects()
        # obj2 = roots[-1]
        # normalize_scene_obj(obj2)
        
        
    
    add_lighting()
    cam, cam_constraint = setup_camera()
    # create an empty object to track
    empty = bpy.data.objects.new("Empty", None)
    scene.collection.objects.link(empty)
    cam_constraint.target = empty
    

    img_ids = [f"{view_num}.png" for view_num in range(12)]
    polar_angles = np.radians([60] * 12 + [90] * 12)
    azimuths = np.radians([*range(0, 360, 30)] * 2)

    cams = []    
    for i in range(len(img_ids)):
        # Sample random camera location around the object
        location = sample_camera_loc(polar_angles[i], azimuths[i], args.camera_dist)
        
        # Compute rotation based on vector going from location towards the location of the object
        rotation_matrix = bproc.camera.rotation_from_forward_vec([0,0,0] - location)
        # Add homog cam pose based on location an rotation
        cam2world_matrix = bproc.math.build_transformation_mat(location, rotation_matrix)
        cams.append(cam2world_matrix)
        print(bproc.camera.add_camera_pose(cam2world_matrix))
    
    RendererUtility.set_cpu_threads(36)
    bproc.renderer.enable_normals_output()
    bproc.renderer.enable_depth_output(activate_antialiasing=False)
    bproc.renderer.set_output_format(enable_transparency=True)
    bproc.renderer.enable_segmentation_output(map_by=[ "instance", "name"])

    for obj in first_obj:
        obj.blender_obj.pass_index = 1

    if subobject_file:
        for obj in second_obj:
            obj.blender_obj.pass_index = 2
    
    

    # from blenderproc.python.utility.BlenderUtility import get_all_blender_mesh_objects
    # all_mesh = get_all_blender_mesh_objects()
    # root_mesh =  find_root_obj(all_mesh)
    # root_mesh_idx = [obj.pass_index for obj in root_mesh]
    
    # for index, obj in enumerate(get_all_blender_mesh_objects()):
    #     obj.pass_index = find_root(obj).pass_index + 1
    
    data = bproc.renderer.render(verbose=True)
    # for index, image in enumerate(data["colors"]):
    #     render_path = os.path.join(args.output_dir, img_ids[index])
    #     save_array_as_image(image, "colors", render_path)
    # for index, image in enumerate(data["normals"]):
    #     breakpoint()
    #     render_path = os.path.join(args.output_dir, f"normal_{img_ids[index]}")
    #     save_array_as_image(image, "normals", render_path)
        
    # # for index, image in enumerate(data["depth"]):
    # #     render_path = os.path.join(args.output_dir, f"depth_{img_ids[index]}")
    # #     save_array_as_image(image, "depth", render_path)


    
    for obj in first_obj:
        obj.hide()
    
    data2 = bproc.renderer.render(verbose=True)
    data2 = {key+"_single" : value for key, value in data2.items()}
    data = {**data, **data2}
    
    pose_data = [cam.flatten()[:12] for cam in cams]
    noramlized_focal = [cam.data.lens / cam.data.sensor_width  ] * len(img_ids)
    intrinsic_data = [ np.array([focal, focal, 0.5, 0.5]) for focal in noramlized_focal]
    camera_data = [np.hstack([ pose, K]) for pose, K in zip(pose_data, intrinsic_data) ]
    data = {**data, "cams" : camera_data}
    
    bproc.writer.write_hdf5(output_dir, data)
    imgs = np.array(data["colors"])
    from einops import rearrange, repeat
    from PIL import Image
    if imgs.shape[0] % 2 == 0:
        grid_w = 2
        grid_h = imgs.shape[0] // 2
        visualize_image = rearrange(imgs, '(n m) h w c -> (n h) (m w) c', n = grid_h, m = grid_w)
    else:
        grid_w = 2
        grid_h = imgs.shape[0] // 2 + 1
        white_img =  np.full_like(imgs[0], 255)
        visualize_image = np.vstack([imgs, white_img[None, ...]])
        visualize_image = rearrange(visualize_image, '(n m) h w c -> (n h) (m w) c', n = grid_h, m = grid_w)
    visualize_image = Image.fromarray(visualize_image)
    visualize_image.save(os.path.join(output_dir, 'colors.png'))


if __name__ == "__main__":
    try:
        start_i = time.time()
        save_images(args.object_path, args.sub_object_path)
        end_i = time.time()
        print("Finished", args.object_path, "in", end_i - start_i, "seconds")
    except Exception as e:
        print("Failed to render", args.object_path)
        print(e)