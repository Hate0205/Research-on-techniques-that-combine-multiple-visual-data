import streamlit as st
import numpy as np
import cv2
import tempfile
from pathlib import Path
from PIL import Image
from matplotlib.colors import LinearSegmentedColormap
from ultralytics import YOLO

# --- Cấu hình giao diện trang ---
st.set_page_config(layout="wide")
st.title("Demo YOLOv8n + LIDAR")
st.sidebar.header("Cài đặt Point Cloud")

# Slider lọc điểm theo trục y trong camera frame
ground_thresh = st.sidebar.slider(
    "Giữ điểm thấp hơn mức y (m)",
    0.0, 3.0, 0.9, 0.1   # mặc định 0.9 m
)

# Slider kích thước voxel
voxel_size = st.sidebar.slider(
    "Kích thước voxel (m)", 0.05, 1.0, 0.1, 0.05   # mặc định 0.1 m
)

# --- Hàm đọc ma trận hiệu chỉnh KITTI ---
def load_calib(calib_file: Path):
    mats = {}
    for line in calib_file.read_text().splitlines():
        if not line:
            continue
        name, raw = line.split(":", 1)
        vals = np.fromstring(raw, sep=" ", dtype=np.float32)
        if name.startswith("P2"):
            mats["P2"] = vals.reshape(3, 4)
        elif name.startswith("R0_rect"):
            mats["R0"] = vals.reshape(3, 3)
        elif name.startswith("Tr_velo_to_cam"):
            mats["Tr"] = vals.reshape(3, 4)
    return mats

# --- Giảm mẫu theo Centroid ---
def voxel_downsample_centroid(points, voxel_size):
    """
    Chia không gian thành các voxel và lấy trung bình (centroid) mỗi voxel làm điểm đại diện.
    """
    voxel_indices = np.floor(points / voxel_size).astype(np.int32)
    voxel_dict = {}
    for idx, coord in zip(map(tuple, voxel_indices), points):
        if idx not in voxel_dict:
            voxel_dict[idx] = []
        voxel_dict[idx].append(coord)
    down_points = [np.mean(v, axis=0) for v in voxel_dict.values()]
    return np.array(down_points, dtype=np.float32)

# --- Chiếu LiDAR sang ảnh (tính pixel u,v và toạ độ camera) ---
def project_lidar(xyz, mats):
    P2, R0, Tr = mats["P2"], mats["R0"], mats["Tr"]

    # Tạo homogeneous
    ones = np.ones((xyz.shape[0], 1), np.float32)
    X_hom = np.hstack([xyz, ones]).T

    # Chuyển LiDAR → camera frame
    cam = R0 @ (Tr @ X_hom)

    # Chuyển sang homogeneous lần 2 để áp dụng P2
    cam_h = np.vstack([cam, ones.T])            # kích thước 4×N
    Y = P2 @ cam_h                              # (3×4) × (4×N) → (3×N)

    # Chuyển về dạng (N×3)
    Yt = Y.T                                    # mỗi hàng là [x', y', z']
    valid = Yt[:, 2] > 0                        # chỉ giữ điểm phía trước camera

    # Tính toạ độ pixel
    uv = (Yt[valid, :2] / Yt[valid, 2:3]).astype(int)
    return uv[:, 0], uv[:, 1], cam.T[valid]

# --- Phát hiện bbox bằng YOLOv8 ---
def detect_bboxes(img_bgr, model, conf_thresh=0.0):
    res = model.predict(img_bgr, verbose=False)[0]
    xyxy = res.boxes.xyxy.cpu().numpy()
    confs = res.boxes.conf.cpu().numpy().tolist()
    cls = res.boxes.cls.cpu().numpy().astype(int).tolist()
    return [
        {'class_id': c, 'confidence': f, 'bbox': list(map(float, box))}
        for box, f, c in zip(xyxy, confs, cls) if f >= conf_thresh
    ]

# --- Colormap cho trực quan hoá ---
CUSTOM_CMAP = LinearSegmentedColormap.from_list(
    "custom", [
        (102/255, 204/255, 51/255),
        (255/255, 204/255, 51/255),
        (255/255, 102/255, 51/255)
    ]
)

# --- Tính giá trị khoảng cách theo mode ---
def compute_plane_val(pts_cam, mode):
    depths = np.linalg.norm(pts_cam, axis=1)
    if depths.size == 0:
        return 0.0
    if mode == "min":
        return float(depths.min())
    elif mode == "max":
        return float(depths.max())
    elif mode == "mean":
        return float(depths.mean())
    elif mode == "median":
        return float(np.median(depths))
    else:
        return 0.0

# --- Hàm hiển thị ---
def visualize_plane(img_orig, u, v, cam_xyz, bboxes, model, mode):
    dists_all = np.linalg.norm(cam_xyz, axis=1)
    dmin, dmax = float(dists_all.min()), float(dists_all.max())
    img = img_orig.copy()

    for obj in bboxes:
        x1, y1, x2, y2 = map(int, obj['bbox'])
        cid = obj['class_id']

        # Vẽ nhãn và bbox
        cv2.putText(img, model.names[cid], (x1, y1-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 2, cv2.LINE_AA)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Chọn điểm nằm trong bbox
        mask = (u >= x1) & (u <= x2) & (v >= y1) & (v <= y2)
        pts_u = u[mask].astype(int)
        pts_v = v[mask].astype(int)
        pts_cam_box = cam_xyz[mask]
        pts_d = dists_all[mask]

        # Tính giá trị khoảng cách
        plane_val = compute_plane_val(pts_cam_box, mode)

        # Vẽ tất cả điểm
        for ui, vi, di in zip(pts_u, pts_v, pts_d):
            norm = (di - dmin) / (dmax - dmin + 1e-6)
            r, g, b, _ = CUSTOM_CMAP(norm)
            cv2.circle(img, (ui, vi), 1,
                       (int(255*b), int(255*g), int(255*r)), -1)

        # Highlight điểm gần nhất với giá trị
        if pts_d.size > 0:
            idx_min = int(np.argmin(np.abs(pts_d - plane_val)))
            u_min, v_min = int(pts_u[idx_min]), int(pts_v[idx_min])
            cv2.circle(img, (u_min, v_min), 6, (0, 0, 255), -1)

        # Hiển thị text giá trị
        txt = f"{plane_val:.2f} m"
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        pos = (x2 - tw, y1 - 6)
        cv2.putText(img, txt, pos,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 255), 2, cv2.LINE_AA)

    return img

# --- Luồng Streamlit ---
files = st.file_uploader(
    "Tải lên: ảnh (.png/.jpg), LiDAR (.bin), Calib (.txt)",
    type=['png', 'jpg', 'jpeg', 'bin', 'txt'],
    accept_multiple_files=True
)

if st.button("Chạy pipeline") and files:
    img_f = next(f for f in files if f.name.lower().endswith(('png', 'jpg', 'jpeg')))
    lidar_f = next(f for f in files if f.name.lower().endswith('bin'))
    calib_f = next(f for f in files if f.name.lower().endswith('txt'))

    with tempfile.TemporaryDirectory() as tmp:
        p_img = Path(tmp) / "img.png"
        p_bin = Path(tmp) / "scan.bin"
        p_calib = Path(tmp) / "calib.txt"
        Image.open(img_f).convert('RGB').save(p_img)
        p_bin.write_bytes(lidar_f.read())
        p_calib.write_bytes(calib_f.read())

        # Load LiDAR gốc
        pc = np.fromfile(str(p_bin), np.float32).reshape(-1, 4)[:, :3]

        # Downsample bằng Centroid
        pc_centroid = voxel_downsample_centroid(pc, voxel_size)

        # Load ma trận hiệu chỉnh
        mats = load_calib(p_calib)

        # Project
        u, v, cam_xyz = project_lidar(pc_centroid, mats)

        # Lọc theo ground_thresh
        mask = cam_xyz[:, 1] < ground_thresh
        u = u[mask]
        v = v[mask]
        cam_xyz = cam_xyz[mask]

        # Phát hiện bbox bằng YOLO
        MODEL_PATH = r"D:\study\Thesis\Thesis\results_experiments\Yolov8\Using_Adapter_to_Convert_3_chanels\Log_Train\yolov8n-RGBD\seed_42\weights\best.pt"
        model = YOLO(MODEL_PATH)
        img_bgr = cv2.imread(str(p_img))
        bboxes = detect_bboxes(img_bgr, model, conf_thresh=0)

        # 4 tab: Min, Max, Mean, Median
        tabs = st.tabs(["Min", "Max", "Mean", "Median"])
        modes = ["min", "max", "mean", "median"]

        for tab, mode in zip(tabs, modes):
            with tab:
                vis = visualize_plane(
                    img_bgr, u, v, cam_xyz, bboxes,
                    model, mode
                )
                st.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), use_container_width=True)
