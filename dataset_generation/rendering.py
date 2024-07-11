import blenderproc as bproc
import os
import time
import math
import argparse
import numpy as np
import blenderproc.python.renderer.RendererUtility as RendererUtility
from blenderproc.scripts.saveAsImg import save_array_as_image
from typing import List
import bpy
from mathutils import Vector, Matrix
import yaml
from einops import rearrange, repeat
from matplotlib import pyplot as plt
import h5py

from  blenderproc.python.writer.WriterUtility import _WriterUtility 


###############args_test###############
def obj_test(obj_list):
    essential_keys = ['type', 'normalize_type', 'normalize_scale', 'seg_id']
    for idx,obj in enumerate(obj_list):
        if not isinstance(obj, dict):
            raise ValueError(f"{idx} obj should be a dictionary")
        for ek in essential_keys:
            if ek not in obj.keys():
                raise ValueError(f"{idx} obj should have {ek}")
        scale = obj['normalize_scale']
        if (not isinstance(scale , (int, float))) or scale <= 0 :
            raise ValueError(f"{idx} obj normalize_scale should be a positive number")
        seg_id = obj['seg_id']
        if not isinstance(seg_id, int) or seg_id <0 :
            raise ValueError(f"{idx} obj seg_id should be a positive integer")
        
def obs_position_test(obs_position):
    essential_keys = ['dist']
    for k in obs_position.keys():
        if k not in essential_keys:
            raise ValueError(f"obs_position should have {essential_keys}")        
    if not isinstance(obs_position['dist'], list) or len(obs_position['dist']) != 2 or obs_position['dist'][0] < 0 or obs_position['dist'][1] < 0 \
        or obs_position['dist'][0] > obs_position['dist'][1]:
        raise ValueError("obs_position dist should be a list of two positive numbers")
    
        
def cam_test(cam_list):
    essential_keys = [ 'dist', 'first_cam_front', 'first_cam_dist', 'azimuth', 'resolution_x', 'resolution_y', 'lens', 'sensor_width','num_views']
    for idx, cam in enumerate(cam_list):
        if not isinstance(cam, dict):
            raise ValueError(f"{idx} camera should be a dictionary")
        for ek in essential_keys:
            if ek not in cam.keys():
                raise ValueError(f"{idx} camera should have {ek}")
        dist = cam['dist']
        if not isinstance(dist, list) or not all (d >0 for d in dist):
            raise ValueError(f"{idx} camera dist should be have positive values")
        first_cam_dist = cam['first_cam_dist']
        if not isinstance(first_cam_dist, list) or not all (d >0 for d in dist):
            raise ValueError(f"{idx} camera first_cam_dist should be a list of two positive numbers")
        if not isinstance(cam['resolution_x'], int) or not isinstance(cam['resolution_y'], int):
            raise ValueError(f"{idx} camera resolution should be an integer")
        if not (('height' in cam.keys()) ^ ('elevation'  in cam.keys())):
            raise ValueError(f"{idx} camera should have height or elevation")
        if cam['azimuth'] == 'uniform' and (not cam.get('azimuth_start', False)):
            raise ValueError("if azimuth is uniform, azimuth_start should be determined")


def render_test(render_config):
    essential_keys = ['normals', 'depth', 'segmentation']
    for ek in essential_keys:
        if ek not in render_config.keys():
            raise ValueError(f"render should have {essential_keys}")
    if render_config['normals']:
        normals_coordinate = render_config.get('normals_coordinate', False)
        if not normals_coordinate:
            raise ValueError("normals coordinates should be determined")
        if not normals_coordinate in ['world', 'camera']:
            raise ValueError("normals coordinates should be world or camera") 

def config_test(args):
    if not args.engine in ["CYCLES", "BLENDER_EEVEE"]:
        raise ValueError("engine should be CYCLES or BLENDER_EEVEE")
    if not args.device in ["CPU", "GPU"]:
        raise ValueError("device should be CPU or GPU")
    if not args.file_format in ["PNG", "JPEG", "BMP", "TIFF", "HDR", "OPEN_EXR"]:
        raise ValueError("file_format should be PNG, JPEG, BMP, TIFF, HDR, OPEN_EXR")
    if not args.color_mode in ["RGB", "RGBA"]:
        raise ValueError("color_mode should be RGB or RGBA")
    if not args.output_type in ["hdf5", "image"]:
        raise ValueError("output_type should be hdf5 or image")
    if not args.naming in ["all", "center_obj", "nth_obj"]:
        raise ValueError("naming should be all, center_obj, nth_obj")
    
    obs_position_test(args.obs_position)
    
    if len(args.obj) == 0:
        raise ValueError("obj should be determined")
    if len(args.obj) != len(args.obj_path):
        raise ValueError("obj and obj_path should have the same length")
    obj_test(args.obj)
    if len(args.camera) == 0:
        raise ValueError("camera should be determined")
    cam_test(args.camera)
    render_test(args.render)
    
    
###############PARSER###############
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, required=True, help="Path to the output directory")
parser.add_argument("--engine", type=str, help="engine type", choices=["CYCLES", "BLENDER_EEVEE"])
parser.add_argument("--device", type=str, help="device type")
parser.add_argument("--file_format", type=str, help="file_format")

parser.add_argument("--color_mode", type=str, help="")
parser.add_argument("--output_type", type=str, help="", choices=["hdf5", "image"])

parser.add_argument("--naming", type=str, help="if you choice 'nth_obj', please check naming_Idx", choices=["all", "center_obj", "nth_obj"])
parser.add_argument("--naming_idx", type=int, help="naming_Idx ; nth obj's name is used for naming")
parser.add_argument("--output_dir", type=str, help="output_dir")
parser.add_argument("--visualize", type=bool, help="if true, grid image is saved")
parser.add_argument("--remove_obs", type=bool, help="if true, there are two type of output ; ().PNG and ()_single.PNG")
parser.add_argument("--obj", type=list, help="object type(center, obs), normalize opt, segment_id")
parser.add_argument("--obs_position", type=dict, help="")

parser.add_argument("--obj_path", action='append', dest='obj_path', help="object_path")
parser.add_argument("--camera", type=list, help="list of camera option")
parser.add_argument("--render", type=list, help="normal, depth, seg, rgb")
parser.add_argument("--gpu_id", type=int, help="gpu_id")
parser.add_argument("--cpu_thread", type=int, help="cpu_thread")
args = parser.parse_args()
if args.config is not None :
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        
for ck, cv in config.items():
    if getattr(args, ck) is None:
        setattr(args, ck, cv)

config_test(args)
###############END PARSER###############


################ BPY SETTING ################
bproc.init()
context = bpy.context
scene = context.scene
render = scene.render

render.engine = args.engine
render.image_settings.file_format = args.file_format
render.image_settings.color_mode = args.color_mode
render.resolution_x = 512 
render.resolution_y = 512 
render.resolution_percentage = 100 

scene.cycles.device = args.device
scene.cycles.samples = 32
scene.cycles.diffuse_bounces = 1
scene.cycles.glossy_bounces = 1
scene.cycles.transparent_max_bounces = 3
scene.cycles.transmission_bounces = 3
scene.cycles.filter_width = 0.01
scene.cycles.use_denoising = True
scene.render.film_transparent = True

bproc.renderer.set_render_devices(use_only_cpu=False, desired_gpu_device_type=None, desired_gpu_ids=args.gpu_id)
RendererUtility.set_cpu_threads(args.cpu_thread)

def add_lighting() -> None:
    # add a new light
    bpy.ops.object.light_add(type="AREA")
    light2 = bpy.data.lights["Area"]
    light2.energy = 70000
    bpy.data.objects["Area"].location[2] = 0.5
    bpy.data.objects["Area"].scale[0] = 100
    bpy.data.objects["Area"].scale[1] = 100
    bpy.data.objects["Area"].scale[2] = 100
    
    
############ load and normalize object ############
def scene_bbox_meshobj(meshobjs: bproc.types.MeshObject):
    bboxs = np.array([obj.get_bound_box() for obj in meshobjs])
    bboxs = bboxs.reshape(-1, 3)
    min_bbox = bboxs.min(axis=0)
    max_bbox = bboxs.max(axis=0)
    return Vector(min_bbox), Vector(max_bbox)

def find_root_meshobj(meshobj: bproc.types.MeshObject):
    parent = meshobj
    while parent.get_parent():
        parent = parent.get_parent()
    return parent

def normalize_meshobj(meshobjs, scale :int = 1 ,translation = None, normalize_type : str = 'xyz'):
    bbox_min, bbox_max = scene_bbox_meshobj(meshobjs)
    scale = 1 / max(getattr(bbox_max, normalize_type) - getattr(bbox_min, normalize_type))
    root_meshobj = list(set([find_root_meshobj(obj) for obj in meshobjs]))[0]
    scale_vec = Vector((scale, scale, scale)) 
    root_meshobj.set_scale(scale * root_meshobj.get_scale( ))
    bbox_min, bbox_max = scene_bbox_meshobj(meshobjs)
    offset = -(bbox_min + bbox_max) / 2
    
    if translation is None:
        root_meshobj.set_location(offset)
    else:
        translation = Vector(translation)
        root_meshobj.set_location(offset + translation)
        
############ END load and normalize object ############

############# camera / obj pose sampling #############
def obj_translation_direction(obs_idx, num_obs):
    return Vector([np.cos(obs_idx* np.pi*2 / num_obs),  np.sin(obs_idx * np.pi*2 / num_obs), 0])

def sample_camera_location_aer(azimuth = 0, elevation = 90, radius = 1.0, deg = True):
    if deg:
        azimuth = np.deg2rad(azimuth)
        elevation = np.deg2rad(elevation)
    x = radius * np.sin(elevation) * np.cos(azimuth)
    y = radius * np.sin(elevation) * np.sin(azimuth)
    z = radius * np.cos(elevation)
    return np.array([x, y, z])
def sample_camera_location_ahr(azimuth = 0, height = 0, radius = 1.0, deg = True):
    if deg:
        azimuth = np.deg2rad(azimuth)
    x = radius * np.cos(azimuth)
    y = radius * np.sin(azimuth)
    z = height
    return np.array([x, y, z])

def sample_camera_pose_aer(phi = 90, theta = 0, radius = 1.0, deg = True):
    location = sample_camera_location_aer(phi, theta, radius, deg)
    rotation_matrix = bproc.camera.rotation_from_forward_vec([0,0,0] - location)
    cam2world_matrix = bproc.math.build_transformation_mat(location, rotation_matrix)
    return cam2world_matrix

def sample_camera_pose_ahr(azimuth = 0, height = 0, radius = 1.0, deg = True):
    location = sample_camera_location_ahr(azimuth, height, radius, deg)
    rotation_matrix = bproc.camera.rotation_from_forward_vec([0,0,0] - location)
    cam2world_matrix = bproc.math.build_transformation_mat(location, rotation_matrix)
    return cam2world_matrix

############# END camera pose sampling #############



############# camera / render configure setting #############
def setup_camera(camera_config):
    render.resolution_x = camera_config['resolution_x']
    render.resolution_y = camera_config['resolution_y']
    cam = scene.objects["Camera"]
    cam.location = (0, 1.0, 0)
    cam.data.lens = camera_config['lens']
    cam.data.sensor_width = camera_config['sensor_width']
    cam_constraint = cam.constraints.new(type="TRACK_TO")
    cam_constraint.track_axis = "TRACK_NEGATIVE_Z"
    cam_constraint.up_axis = "UP_Y"
    return cam, cam_constraint

def setup_render(render_config):
    default_render = ['normals', 'depth', 'segmentation']
    for k, v in render_config.items():
        if v:
            if k == 'segmentation':
                getattr(bproc.renderer, f"enable_{k}_output")(map_by=[ "instance", "name"])
            elif k == 'depth':
                getattr(bproc.renderer, f"enable_{k}_output")(activate_antialiasing=False)
            elif  k in default_render:
                getattr(bproc.renderer, f"enable_{k}_output")()
    bproc.renderer.set_output_format(enable_transparency=True)
    
def setup_normal_render():
    bproc.renderer.enable_normals_output()
    bproc.renderer.set_output_format(enable_transparency=True)
############# END camera configure setting #############



############# output_name #############
def determine_output_dir(args):
    if args.output_dir is None:
        raise ValueError("output_dir should be determined")
    if not (args.naming in ['all', 'center_obj', 'nth_obj']):
        raise ValueError("naming should be one of ['all', 'center_obj', 'nth_obj']")
    if args.naming == 'nth_obj' and args.naming_idx is None:
        raise ValueError("naming_idx should be determined when naming is 'nth_obj'")
    
    obj_paths = args.obj_path
    obj_path_files = [os.path.basename(obj_path).split('.')[0] for obj_path in obj_paths]
    path = None
    if args.naming == 'all':
        path = '+'.join(obj_path_files)
    elif args.naming == 'center_obj':
        obj_config = args.obj 
        center_obj_idx= next((i for i, item in enumerate(obj_config) if item["type"] == "center"), None)
        if center_obj_idx is None:
            raise ValueError("center_obj should be determined when naming is 'center_obj'")
        path = obj_path_files[center_obj_idx]
    elif args.naming == 'nth_obj':
        if args.naming_idx >= len(obj_path_files):
            raise ValueError("naming_idx should be less than the number of objects")
        path = obj_path_files[args.naming_idx]
    return os.path.join(args.output_dir, path)


############# END output_name #############

############# visualize ##############

def visualize(rendered_data, output_dir):
    default_rgb_keys = ["colors","colors_single"]
    default_normal_keys = ["normals", "normals_single"]
    default_seg_keys = ["instance_segmaps", "instance_segmaps_single"]
    default_depth_keys = ['depth', 'depth_single']
    default_depth_max = 5
    default_depth_min = 2.0
    # color, normal (V, H, W, C)
    # seg, depth(V, H, W)
    
    num_view = len(rendered_data['colors'])
    grid_w = 2  
    grid_h = (num_view + grid_w -1) // grid_w 
    num_additional = grid_w * grid_h - num_view
    
    for k, v in rendered_data.items():
        file_path = os.path.join(output_dir, f'{k}.png')
        if k in default_rgb_keys:
            data = np.array(v + [np.zeros_like(v[0])] * num_additional).astype(np.uint8)
            visualize_image = rearrange(data, '(n m) h w c -> (n h) (m w) c', n = grid_h, m = grid_w)
            plt.imsave(file_path, visualize_image)
            
        elif k in default_normal_keys:
            data = np.array(v + [np.zeros_like(v[0])] * num_additional).astype(np.float32)
            data = np.clip(data, 0., 1.)
            visualize_image = rearrange(data, '(n m) h w c -> (n h) (m w) c', n = grid_h, m = grid_w)
            plt.imsave(file_path, visualize_image)
            
        elif k in default_seg_keys:
            data = np.array(v + [np.zeros_like(v[0])] * num_additional)
            visualize_image = rearrange(data, '(n m) h w -> (n h) (m w)', n = grid_h, m = grid_w)
            plt.imsave(file_path, visualize_image, cmap='jet')

        elif k in default_depth_keys:
            data = np.array(v + [np.zeros_like(v[0])] * num_additional)
            visualize_image = rearrange(data, '(n m) h w-> (n h) (m w)', n = grid_h, m = grid_w)
            vmax = np.max(data)
            vmin = np.min(data)
            plt.imsave(file_path, visualize_image, cmap='viridis', vmax = vmax, vmin = vmin)
        
        


############# END visualize ##############

############# hdf IO #####################
def write_hdf5(output_dir, rendered_data, file_name = "data"):
    def to_ndarray(data):
        if isinstance(data, list):
            if isinstance(data[0], (np.ndarray)):
                return np.stack(data) 
            elif isinstance(data[0], (int, float)):
                return np.array(data)
            elif isinstance(data[0], tuple):
                return np.stack([np.array(d) for d in data])
        
        return None
    
    
    
    with h5py.File(os.path.join(output_dir, f"{file_name}.h5"), 'w') as f:
        for k, v in rendered_data.items():
            array = to_ndarray(v)
            if array is not None:
                if array.dtype in [np.int16, np.int32, np.int64]:  # possible integer value is rgb or segmentation 
                    array = array.astype(np.int8)
                _WriterUtility.write_to_hdf_file(f, k, array)
    return

############# END hdf IO #####################









def rendering(args) :
    output_dir = determine_output_dir(args)
    os.makedirs(output_dir)
    
    empty = bpy.data.objects.new("Empty", None)
    scene.collection.objects.link(empty)
    
    setup_render(args.render)
    
    num_obj = len(args.obj)
    center_obj_idx = next((i for i, item in enumerate(args.obj) if item["type"] == "center"), None)
    num_obstacle = num_obj - 1 if center_obj_idx is not None else num_obj
    obstacle_idx = [i for i in range(num_obj) if i < center_obj_idx] \
        + [None] +[i-1 for i in range(num_obj) if i > center_obj_idx]
    
    obstacles = []
    
    
    
    for obj_config , obj_path, obs_idx in zip(args.obj, args.obj_path, obstacle_idx):
        try :
            obj_mesh = bproc.loader.load_obj(obj_path)
        except Exception as e:
            raise ValueError(f"Error in loading {obj_path}")
        
        translation = None if obj_config['type'] == 'center' \
            else  np.random.uniform(*args.obs_position['dist'] , 1)  * obj_translation_direction(obs_idx, num_obstacle)
        
        normalize_meshobj(obj_mesh, 
                          scale = obj_config['normalize_scale'],
                          translation = translation,
                          normalize_type=obj_config['normalize_type'])
        for obj_ in obj_mesh:
            obj_.blender_obj.pass_index = obj_config['seg_id']
        
        if obj_config['type'] == 'obs': obstacles.append(obj_mesh)
            
    add_lighting()
    rendered_data = {}
    for cam_config in args.camera :
        cam, cam_constraint = setup_camera(cam_config)
        cam_constraint.target = empty
        
        cam_poses = []
        azimuths = None
        dists = None
        
        if cam_config['azimuth'] == "random":
            azimuths = np.random.uniform(0, 360, cam_config['num_views'])
        else:
            azimuths = np.linspace(0, 360, cam_config['num_views'], endpoint=False)
        
        dist_range = cam_config['dist']
        if len(dist_range) == 2 :
            dists = np.random.uniform(*dist_range, cam_config['num_views'])
        else:
            random_idx = np.random.randint(0, len(dist_range), cam_config['num_views'])
            dists = np.array([ dist_range[idx] for idx in random_idx])
        
        if 'height' in cam_config.keys():
            heights = None
            if len(cam_config['height']) == 2:
                heights = np.random.uniform(*cam_config['height'], cam_config['num_views'])
            else:
                random_idx = np.random.randint(0, len(cam_config['height']), cam_config['num_views'])
                heights = np.array([ cam_config['height'][idx] for idx in random_idx])
            
            for azimuth, dist, height in zip(azimuths, dists, heights):
                cam2world_matrix = sample_camera_pose_ahr(azimuth, height, dist, deg = True)
                cam_poses.append(cam2world_matrix)
        elif 'elevation' in cam_config.keys():
            elevations = None
            if len(cam_config['elevation']) == 2:
                elevations = np.random.uniform(*cam_config['elevation'], cam_config['num_views'])
            else:
                random_idx = np.random.randint(0, len(cam_config['elevation']), cam_config['num_views'])
                elevations = np.array([ cam_config['elevation'][idx] for idx in random_idx])
            for azimuth, dist, elevation in zip(azimuths, dists, elevations):
                cam2world_matrix = sample_camera_pose_aer(azimuth, elevation, dist, deg = True)
                cam_poses.append(cam2world_matrix)
        else :
            raise ValueError("camera should have height or elevation")
        
        if cam_config.get('first_cam_front', False):
            first_cam_dist = cam_config['first_cam_dist']
            dist = None
            if len(first_cam_dist) != 2 :
                raise ValueError("first_cam_dist should be range (two values)")
            dist = np.random.uniform(*first_cam_dist, 1)
            cam2world_matrix = sample_camera_pose_ahr(0, 0, dist, deg = True)
            cam_poses[0] = cam2world_matrix
            
        for pose in cam_poses: 
            print(bproc.camera.add_camera_pose(pose))
        
        setup_normal_render()
        
        time2 = time.time()
        
        
        data = bproc.renderer.render(verbose=True)
        
        time3 = time.time()
        
        if args.remove_obs :
            for obs in obstacles:
                for obj in obs:
                    obj.hide() 
            single_data = bproc.renderer.render(verbose=True)
            single_data = {key+"_single" : value for key, value in single_data.items()}
            data = {**data, **single_data}
            
        time4 = time.time()
        ####### add camera paramter ##############
        K = bproc.camera.get_intrinsics_as_K_matrix()
        K = [K] * cam_config['num_views']
        fov = bproc.camera.get_fov()
        fov = [fov] * cam_config['num_views']
        resolution = [(cam_config['resolution_x'], cam_config['resolution_y'])] * cam_config['num_views']
        cam_data = {"cam_poses" : cam_poses,
                    "K" : K,
                    "fov" : fov,
                    "resolution" : resolution}

        data = {**data, **cam_data}
        if args.render['normals'] and args.render['normals_coordinate'] == "world":
            data['normals'] = [(np.einsum('ij, ...j -> ...i', pose[:3, :3], (normal-0.5) *2)) /2 +0.5 for normal, pose in zip(data['normals'], cam_poses)]
            if args.remove_obs:
                data['normals_single'] = [(np.einsum('ij, ...j -> ...i', pose[:3, :3], (normal-0.5) *2)) /2 +0.5 for normal, pose in zip(data['normals_single'], cam_poses)]
            
        union_key = set(rendered_data.keys()) | set(data.keys())
        
        rendered_data = {k :rendered_data.get(k, []) + data.get(k, [])  for k in union_key}
        bproc.utility.reset_keyframes()
        

    # visualize(rendered_data, output_dir)
    write_hdf5(output_dir, rendered_data)
    
    with open(os.path.join(output_dir, "description.txt"), "w") as f:
        f.write("Obstacle\n")
        for idx, obsi  in enumerate(obstacle_idx):
            if obsi is None:
                continue
            obs_file = args.obj_path[idx]
            f.write(f"{obs_file}\n")
        f.write("Center\n")
        center_file = args.obj_path[center_obj_idx]
        f.write(f"{center_file}\n")
    
if __name__ == "__main__":
    rendering(args)