import os
import json
import cv2
import numpy as np
import torch
from scipy.io import loadmat
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

from mmpose.apis import init_model, inference_topdown
from mmpose.utils import register_all_modules

CONFIG_FILE = 'satellite/rtmpose-t_satellite_f.py'
CHECKPOINT_FILE = '/workspace/rtmpose-t_final/epoch_310.pth'

# 1. GT Pose가 들어있는 파일 (synthetic 폴더)
POSE_GT_FILE = '/workspace/speedplusv2/sunlamp/test.json' 
# 2. Crop Box 등 이미지 메타정보가 들어있는 파일 (annotations 폴더, COCO 포맷)
IMAGE_META_FILE = '/workspace/speedplusv2/annotations/test_sunlamp.json'

IMG_ROOT = '/workspace/speedplusv2/sunlamp_preprocessed/'
MODEL_3D_POINTS_FILE = '/workspace/speedplusv2/tangoPoints.mat'
CAMERA_FILE = '/workspace/speedplusv2/camera.json'

# ------------------------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------------------------

def load_camera_intrinsics(camera_json_path):
    if os.path.exists(camera_json_path):
        with open(camera_json_path) as f:
            cam = json.load(f)
        K = np.array(cam['cameraMatrix'], dtype=np.float32)
        dist = np.array(cam['distCoeffs'], dtype=np.float32).flatten()
        return K, dist
    else:
        raise FileNotFoundError(f"Camera parameters file not found: {camera_json_path}")

def load_tango_3d_keypoints(mat_path):
    if not os.path.exists(mat_path):
        raise FileNotFoundError(f"3D points file not found: {mat_path}")
    try:
        vertices = loadmat(mat_path)['tango3Dpoints']
        corners3D = np.transpose(np.array(vertices, dtype=np.float32))
        return corners3D
    except Exception as e:
        raise RuntimeError(f"Failed to load 3D points from {mat_path}: {e}")

def solve_pnp_epnp(points_3D, points_2D, cameraMatrix, distCoeffs):
    if distCoeffs is None:
        distCoeffs = np.zeros((5, 1), dtype=np.float32)

    points_3D = np.ascontiguousarray(points_3D).reshape((-1, 1, 3))
    points_2D = np.ascontiguousarray(points_2D).reshape((-1, 1, 2))

    success, rvec, tvec = cv2.solvePnP(
        points_3D, points_2D, cameraMatrix, distCoeffs, flags=cv2.SOLVEPNP_EPNP
    )

    if not success:
        return None, None

    R_mat, _ = cv2.Rodrigues(rvec)
    q = R.from_matrix(R_mat).as_quat() # [x, y, z, w]
    q_scalar_first = q[[3, 0, 1, 2]]   # [w, x, y, z]
    
    return q_scalar_first, np.squeeze(tvec)

def compute_errors(q_pred, t_pred, q_gt, t_gt):
    t_err = np.linalg.norm(t_pred - t_gt)

    q_pred = q_pred / np.linalg.norm(q_pred)
    q_gt = q_gt / np.linalg.norm(q_gt)
    dot = np.clip(np.abs(np.dot(q_pred, q_gt)), -1.0, 1.0)
    rad_err = 2 * np.arccos(dot)
    deg_err = np.rad2deg(rad_err)
    
    return t_err, deg_err

# ------------------------------------------------------------------------------
# Main Evaluation Logic
# ------------------------------------------------------------------------------

def main():
    register_all_modules()
    
    print(f"Loading model from {CONFIG_FILE}...")
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = init_model(CONFIG_FILE, CHECKPOINT_FILE, device=device)

    # 데이터 로드
    points_3d = load_tango_3d_keypoints(MODEL_3D_POINTS_FILE)
    K, dist = load_camera_intrinsics(CAMERA_FILE)
    
    print(f"Loading GT Poses from {POSE_GT_FILE}...")
    with open(POSE_GT_FILE, 'r') as f:
        pose_data = json.load(f)
        pose_list = pose_data['images'] if (isinstance(pose_data, dict) and 'images' in pose_data) else pose_data

    print(f"Loading Image Meta from {IMAGE_META_FILE}...")
    with open(IMAGE_META_FILE, 'r') as f:
        meta_data = json.load(f)
        original_images_map = meta_data.get('original_images')
        if original_images_map is None:
             raise KeyError(f"'original_images' key missing in {IMAGE_META_FILE}")

    t_errors = []
    q_errors = []
    
    print(f"Starting evaluation on {len(pose_list)} images...")
    
    for item in tqdm(pose_list):
        # 1. 파일명 매칭
        raw_filename = item['filename'] # 예: "img000014.jpg"
        if raw_filename.startswith('img'):
            clean_filename = raw_filename[3:] # "000014.jpg"
        else:
            clean_filename = raw_filename
            
        # 메타데이터 확인
        if clean_filename not in original_images_map:
            # print(f"Meta missing for {clean_filename}")
            continue
            
        img_meta = original_images_map[clean_filename]
        
        # 2. 이미지 로드 (224x224 Preprocessed Image)
        # 실제 파일명이 raw_filename인지 clean_filename인지 확인하여 경로 설정
        img_path = os.path.join(IMG_ROOT, clean_filename)
        if not os.path.exists(img_path):
            img_path_alt = os.path.join(IMG_ROOT, raw_filename)
            if os.path.exists(img_path_alt):
                img_path = img_path_alt
            else:
                # print(f"Image not found: {img_path}")
                continue

        # 이미지를 읽어옵니다 (이미 224x224라고 가정)
        img_input = cv2.imread(img_path)
        if img_input is None:
            continue
        
        # GT Pose 로드
        if 'r_Vo2To_vbs_true' not in item: continue
        t_gt = np.array(item['r_Vo2To_vbs_true'], dtype=np.float32)
        q_gt = np.array(item['q_vbs2tango_true'], dtype=np.float32)

        # 3. Inference (224x224 입력)
        # bboxes=None으로 주면 이미지 전체를 사용하며, 
        # 입력 이미지가 모델 입력 사이즈와 같으면 리사이즈 없이 처리됩니다.
        results = inference_topdown(model, img_input)
        pred_instances = results[0].pred_instances
        
        # 224x224 이미지 상의 좌표 추출
        kpts_224 = pred_instances.keypoints[0] # shape: (11, 2)
        
        # 4. 좌표 복원 (Restore Coordinates)
        # 공식: kpts_orig = kpts_224 / scale_factor + [x1, y1]
        crop_box = img_meta['crop_box']   # [x1, y1, x2, y2]
        scale_factor = img_meta['scale_factor']
        
        x1, y1 = crop_box[0], crop_box[1]
        
        # (N, 2) 좌표에 대해 브로드캐스팅 연산
        kpts_orig = (kpts_224 / scale_factor) + np.array([x1, y1])
        
        # 5. PnP (EPnP)
        # 원본 좌표계의 2D 좌표(kpts_orig)와 3D 모델 좌표(points_3d) 매칭
        q_pred, t_pred = solve_pnp_epnp(points_3d, kpts_orig, K, dist)

        if q_pred is not None:
            t_err, q_err = compute_errors(q_pred, t_pred, q_gt, t_gt)
            t_errors.append(t_err)
            q_errors.append(q_err)
        else:
            print(f"PnP Failed for {clean_filename}")

    # 결과 출력
    if len(t_errors) > 0:
        print("\n=== Evaluation Results ===")
        print(f"Total Images Evaluated: {len(t_errors)}")
        print(f"Mean Translation Error: {np.mean(t_errors):.6f} m")
        print(f"Mean Rotation Error:    {np.mean(q_errors):.6f} deg")
    else:
        print("No valid results found.")

if __name__ == '__main__':
    main()