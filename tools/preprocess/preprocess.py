import cv2
import numpy as np
import os
import json
import random
from scipy.io import loadmat
from tqdm import tqdm 

def quat2dcm(q):
    q_norm = np.linalg.norm(q)
    if q_norm == 0: return np.identity(3, dtype=np.float32)
    q = q / q_norm
    q0, q1, q2, q3 = q[0], q[1], q[2], q[3] # w, x, y, z
    dcm = np.zeros((3, 3), dtype=np.float32)
    dcm[0, 0] = 2 * q0 ** 2 - 1 + 2 * q1 ** 2
    dcm[1, 1] = 2 * q0 ** 2 - 1 + 2 * q2 ** 2
    dcm[2, 2] = 2 * q0 ** 2 - 1 + 2 * q3 ** 2
    dcm[0, 1] = 2 * q1 * q2 + 2 * q0 * q3
    dcm[0, 2] = 2 * q1 * q3 - 2 * q0 * q2
    dcm[1, 0] = 2 * q1 * q2 - 2 * q0 * q3
    dcm[1, 2] = 2 * q2 * q3 + 2 * q0 * q1
    dcm[2, 0] = 2 * q1 * q3 + 2 * q0 * q2
    dcm[2, 1] = 2 * q2 * q3 - 2 * q0 * q1
    return dcm

def project_keypoints(q, r, K, dist, keypoints):
    if keypoints.shape[0] != 3: keypoints = np.transpose(keypoints)
    keypoints_homo = np.vstack((keypoints, np.ones((1, keypoints.shape[1]))))
    pose_mat = np.hstack((np.transpose(quat2dcm(q)), np.expand_dims(r, 1)))
    xyz = np.dot(pose_mat, keypoints_homo)
    z_eps = 1e-8
    xyz[2,:] = np.maximum(xyz[2,:], z_eps)
    x0, y0 = xyz[0,:] / xyz[2,:], xyz[1,:] / xyz[2,:]
    r2 = x0*x0 + y0*y0
    cdist = 1 + dist[0]*r2 + dist[1]*r2*r2 + dist[4]*r2*r2*r2
    x = x0*cdist + dist[2]*2*x0*y0 + dist[3]*(r2 + 2*x0*x0)
    y = y0*cdist + dist[2]*(r2 + 2*y0*y0) + dist[3]*2*x0*y0
    points2D = np.vstack((K[0,0]*x + K[0,2], K[1,1]*y + K[1,2]))
    return points2D, xyz

def ray_triangle_intersect(Ray_O, Ray_D, v0, v1, v2, epsilon=1e-6):
    edge1 = v1 - v0
    edge2 = v2 - v0
    h = np.cross(Ray_D, edge2)
    a = np.dot(edge1, h)
    if -epsilon < a < epsilon: return False
    f = 1.0 / a
    s = Ray_O - v0
    u = f * np.dot(s, h)
    if u < 0.0 or u > 1.0: return False
    q = np.cross(s, edge1)
    v = f * np.dot(Ray_D, q)
    if v < 0.0 or u + v > 1.0: return False
    t = f * np.dot(edge2, q)
    if t > epsilon and t < (1.0 - epsilon):
        return True 
    else:
        return False

def get_bbox_from_mask(mask_path):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None: return None
    coords = np.where(mask > 0)
    if len(coords[0]) == 0 or len(coords[1]) == 0: return None
    ymin, ymax = np.min(coords[0]), np.max(coords[0])
    xmin, xmax = np.min(coords[1]), np.max(coords[1])
    return xmin, ymin, xmax, ymax

def get_safe_crop_coords(img_shape, target_center_x, target_center_y, target_size):
    img_h, img_w = img_shape
    xmin = target_center_x - target_size // 2
    ymin = target_center_y - target_size // 2
    xmax = xmin + target_size
    ymax = ymin + target_size
    
    dx = 0
    if xmin < 0: dx = -xmin
    elif xmax > img_w: dx = img_w - xmax
        
    dy = 0
    if ymin < 0: dy = -ymin
    elif ymax > img_h: dy = img_h - ymax

    final_xmin = int(xmin + dx)
    final_ymin = int(ymin + dy)
    final_xmax = int(xmax + dx)
    final_ymax = int(ymax + dy)
    
    return final_xmin, final_ymin, final_xmax, final_ymax

def main():
    root_dir = '/workspace'
    speedplus_dir = os.path.join(root_dir, 'speedplusv2')
    synthetic_dir = os.path.join(speedplus_dir, 'synthetic')
    
    annot_out_dir = os.path.join(speedplus_dir, 'annotations')
    train_img_out_dir = os.path.join(speedplus_dir, 'train')
    val_img_out_dir = os.path.join(speedplus_dir, 'val')
    
    os.makedirs(annot_out_dir, exist_ok=True)
    os.makedirs(train_img_out_dir, exist_ok=True)
    os.makedirs(val_img_out_dir, exist_ok=True)

    # Random setting for training set
    RANDOM_SIZE_RANGE = (1.1, 1.3)
    RANDOM_SHIFT_RATIO = 0.1
    TEST_SIZE_SCALE = 1.2
    FINAL_RESIZE_SIZE = 224

    print("Loading shared data (camera, keypoints)...")
    camera_config_path = os.path.join(speedplus_dir, 'camera.json')
    try:
        with open(camera_config_path, 'r') as f: cam_data = json.load(f)
        IMG_WIDTH = cam_data.get('Nu', 1920)
        IMG_HEIGHT = cam_data.get('Nv', 1200)
        K = np.array(cam_data['cameraMatrix'], dtype=np.float32)
        D = np.array(cam_data['distCoeffs'], dtype=np.float32)
        camera = {'cameraMatrix': K, 'distCoeffs': D}
    except FileNotFoundError:
        print(f"Error: {camera_config_path} not found."); return
    IMG_SHAPE = (IMG_HEIGHT, IMG_WIDTH)

    keypoints_path = os.path.join(speedplus_dir, 'tangoPoints.mat')
    mat_data = loadmat(keypoints_path)
    data_key = [k for k in mat_data.keys() if not k.startswith('__')][0]
    keypts3d = mat_data[data_key].astype(np.float32)
    if keypts3d.shape == (3, 11): keypts3d = keypts3d.T
    keypts3d_transposed = keypts3d.T
    origin = np.zeros((3, 1), dtype=np.float32)
    keypts3d_origin = np.concatenate((origin, keypts3d_transposed), axis=1) # (3, 12)
    
    def process_split(json_filename, is_training, output_img_dir):
        
        print(f"\n--- Processing {json_filename} (Mode: {'Train' if is_training else 'Val'}) ---")
        
        json_path = os.path.join(synthetic_dir, json_filename)
        try:
            with open(json_path, 'r') as f: labels = json.load(f)
        except FileNotFoundError:
            print(f"Error: {json_path} not found."); return

        # COCO annotation dictionary
        coco_output = {
            'images': [],
            'annotations': [],
            'categories': [{'id': 1, 'name': 'satellite'}],
            'original_images': {} # original recovery
        }
        
        annotation_id_counter = 1
        
        for image_id_counter, image_data in enumerate(tqdm(labels), 1):
            image_name = image_data['filename']
            q_vbs2tango = np.array(image_data['q_vbs2tango_true'], dtype=np.float32)
            r_Vo2To_vbs = np.array(image_data['r_Vo2To_vbs_true'], dtype=np.float32)
            
            mask_path = os.path.join(synthetic_dir, 'masks', image_name)
            image_path = os.path.join(synthetic_dir, 'images', image_name)
            
            new_file_name = image_name.replace('img', '')
            output_image_path = os.path.join(output_img_dir, new_file_name)

            # Occulation
            keypts2d_orig_raw, keypts3d_cam = project_keypoints(
                q_vbs2tango, r_Vo2To_vbs,
                camera['cameraMatrix'], camera['distCoeffs'],
                keypts3d_origin
            )
            keypts2d_orig = keypts2d_orig_raw.T       # (12, 2) [x, y]
            keypts3d_cam_vis = keypts3d_cam.T # (12, 3) [X, Y, Z]

            faces = [
                [1, 2, 3, 4], [5, 6, 7, 8], [1, 2, 6, 5],
                [2, 3, 7, 6], [3, 4, 8, 7], [4, 1, 5, 8]
            ]
            triangles = []
            for face in faces:
                v0, v1, v2, v3 = [keypts3d_cam_vis[idx] for idx in face]
                triangles.append((v0, v1, v2)); triangles.append((v0, v2, v3))

            object_center_cam = np.mean(keypts3d_cam_vis[1:9], axis=0)
            face_is_visible = []
            for face in faces:
                v1, v2, v3 = [keypts3d_cam_vis[idx] for idx in face[:3]]
                normal = np.cross(v2 - v1, v3 - v1)
                face_centroid = (v1 + v2 + v3) / 3.0
                center_to_face_vec = face_centroid - object_center_cam
                if np.dot(normal, center_to_face_vec) < 0: normal = -normal
                view_vec = -face_centroid 
                face_is_visible.append(np.dot(normal, view_vec) > 0)

            vertex_faces_map = {i: [] for i in range(1, 9)}
            for f_idx, face in enumerate(faces):
                for v_idx in face: vertex_faces_map[v_idx].append(f_idx)
                    
            vertex_is_visible = {i: False for i in range(12)}
            for v_idx, f_indices in vertex_faces_map.items():
                if any(face_is_visible[f_idx] for f_idx in f_indices):
                    vertex_is_visible[v_idx] = True
            vertex_is_visible[0] = True 

            for kp_idx in [9, 10, 11]:
                is_occluded = False
                Ray_O = keypts3d_cam_vis[kp_idx]; Ray_D = -Ray_O
                for tri in triangles:
                    if ray_triangle_intersect(Ray_O, Ray_D, tri[0], tri[1], tri[2]):
                        is_occluded = True; break
                vertex_is_visible[kp_idx] = not is_occluded
            
            # Calculate crop region
            bbox = get_bbox_from_mask(mask_path)
            if bbox is None:
                continue
                
            xmin, ymin, xmax, ymax = bbox
            bbox_w = xmax - xmin; bbox_h = ymax - ymin
            bbox_center_x = xmin + bbox_w / 2
            bbox_center_y = ymin + bbox_h / 2
            long_side = max(bbox_w, bbox_h)

            if is_training:
                size_jitter = random.uniform(RANDOM_SIZE_RANGE[0], RANDOM_SIZE_RANGE[1])
                target_size = int(long_side * size_jitter)
                max_shift = int(long_side * RANDOM_SHIFT_RATIO / 2)
                shift_x = random.randint(-max_shift, max_shift)
                shift_y = random.randint(-max_shift, max_shift)
                target_center_x = bbox_center_x + shift_x
                target_center_y = bbox_center_y + shift_y
            else: # (Test/Val)
                target_size = int(long_side * TEST_SIZE_SCALE)
                target_center_x = bbox_center_x
                target_center_y = bbox_center_y

            img_h, img_w = IMG_SHAPE
            target_size = min(target_size, img_h, img_w)
            
            cx_min, cy_min, cx_max, cy_max = get_safe_crop_coords(
                IMG_SHAPE, target_center_x, target_center_y, target_size
            )
            crop_size = cx_max - cx_min # square
            if crop_size == 0: continue 

            # Crop and resize
            image = cv2.imread(image_path)
            if image is None:
                continue
                
            cropped_image = image[cy_min:cy_max, cx_min:cx_max]
            
            cropped_resized_image = cv2.resize(
                cropped_image, 
                (FINAL_RESIZE_SIZE, FINAL_RESIZE_SIZE), 
                interpolation=cv2.INTER_LINEAR
            )
            
            cv2.imwrite(output_image_path, cropped_resized_image)

            # Generate COCO annotation
            image_entry = {
                'file_name': new_file_name,
                'height': FINAL_RESIZE_SIZE,
                'width': FINAL_RESIZE_SIZE,
                'id': image_id_counter
            }
            coco_output['images'].append(image_entry)
            
            # 'original_images' for recovery
            scale_factor = FINAL_RESIZE_SIZE / crop_size
            orig_info = {
                'original_size': [IMG_WIDTH, IMG_HEIGHT],
                'crop_box': [cx_min, cy_min, cx_max, cy_max],
                'scale_factor': scale_factor
            }
            coco_output['original_images'][new_file_name] = orig_info

            # 'annotations'
            keypoints_coco_list = [] # [x,y,v] 11개 (총 33개)
            num_keypoints_count = 0
            visible_x_coords = []
            visible_y_coords = []
            
            for i in range(12): 
                x_orig, y_orig = keypts2d_orig[i]
                is_3d_visible = vertex_is_visible[i]
                
                x_crop = x_orig - cx_min
                y_crop = y_orig - cy_min
                x_final_raw = x_crop * scale_factor
                y_final_raw = y_crop * scale_factor
                x_final_int = int(round(x_final_raw))
                y_final_int = int(round(y_final_raw))

                is_in_new_frame = (0 <= x_final_int < FINAL_RESIZE_SIZE) and \
                                  (0 <= y_final_int < FINAL_RESIZE_SIZE)

                if not is_in_new_frame:
                    flag, x_out, y_out = 0, 0, 0
                else:
                    x_out, y_out = x_final_int, y_final_int
                    flag = 2 if is_3d_visible else 1
                
                # exclude origin(i=0) and save 11 'keypoints'
                if i > 0:
                    keypoints_coco_list.extend([x_out, y_out, flag])
                    if flag > 0: # 1 or 2
                        num_keypoints_count += 1
                        visible_x_coords.append(x_out)
                        visible_y_coords.append(y_out)
            
            # Bbox
            if num_keypoints_count > 0:
                b_xmin = int(np.min(visible_x_coords))
                b_ymin = int(np.min(visible_y_coords))
                b_xmax = int(np.max(visible_x_coords))
                b_ymax = int(np.max(visible_y_coords))
                b_w = b_xmax - b_xmin
                b_h = b_ymax - b_ymin
                bbox_coco = [b_xmin, b_ymin, b_w, b_h]
            else:
                bbox_coco = [0, 0, 0, 0] # Not visible
            
            # 'annotations'
            annotation_entry = {
                'keypoints': keypoints_coco_list,
                'num_keypoints': num_keypoints_count,
                'iscrowd': 0,
                'image_id': image_id_counter,
                'bbox': bbox_coco,
                'category_id': 1,
                'id': annotation_id_counter
            }
            coco_output['annotations'].append(annotation_entry)
            
            annotation_id_counter += 1
        
        output_json_path = os.path.join(annot_out_dir, json_filename)
        with open(output_json_path, 'w') as f:
            json.dump(coco_output, f, indent=4)
        print(f"\nSaved annotation file to {output_json_path}")
        print(f"Total {len(coco_output['images'])} images processed.")

    process_split('train.json', True, train_img_out_dir)
    process_split('validation.json', False, val_img_out_dir)

    print("\n--- All processing complete. ---")

if __name__ == '__main__':
    main()