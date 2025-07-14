# app.py

import streamlit as st
import numpy as np
import cv2
import tempfile
from pathlib import Path
from PIL import Image
from matplotlib.colors import LinearSegmentedColormap
from ultralytics import YOLO
import open3d as o3d  # pip install open3d

# === Page & sidebar setup ===
st.set_page_config(layout="wide")
st.title("Demo YOLOv8 + LIDAR — Radius Outlier + (Min & Median)")
st.sidebar.header("Point-Cloud Band")
eps = st.sidebar.slider("Depth band ±ε (m)", 0.0, 2.0, 0.2, 0.05)

# === 1) Load KITTI calib ===
def load_calib(calib_file: Path):
    mats = {}
    for line in calib_file.read_text().splitlines():
        if not line: continue
        name, raw = line.split(":", 1)
        vals = np.fromstring(raw, sep=" ", dtype=np.float32)
        if name.startswith("P2"):
            mats["P2"] = vals.reshape(3,4)
        elif name.startswith("R0_rect"):
            mats["R0"] = vals.reshape(3,3)
        elif name.startswith("Tr_velo_to_cam"):
            mats["Tr"] = vals.reshape(3,4)
    return mats

# === 2) Project LiDAR → pixels & cam-frame coords ===
def project_lidar(xyz, mats):
    P2, R0, Tr = mats["P2"], mats["R0"], mats["Tr"]
    ones = np.ones((xyz.shape[0],1), np.float32)
    X = np.hstack([xyz, ones]).T
    cam = R0 @ (Tr @ X)
    cam_h = np.vstack([cam, ones.T])
    Y = P2 @ cam_h
    Yt = Y.T
    valid = Yt[:,2] > 0
    uv = (Yt[valid,:2] / Yt[valid,2:3]).astype(int)
    return uv[:,0], uv[:,1], cam.T[valid]

# === 3) Detect boxes with YOLOv8 ===
def detect_bboxes(img_bgr, model, conf_thresh=0.0):
    res   = model.predict(img_bgr, verbose=False)[0]
    xyxy  = res.boxes.xyxy.cpu().numpy()
    confs = res.boxes.conf.cpu().numpy().tolist()
    cls   = res.boxes.cls.cpu().numpy().astype(int).tolist()
    return [
        {'class_id':c, 'confidence':f, 'bbox':list(map(float,box))}
        for box,f,c in zip(xyxy, confs, cls) if f >= conf_thresh
    ]

# === 4) Custom colormap for point-cloud ===
CUSTOM_CMAP = LinearSegmentedColormap.from_list(
    "custom", [
        (102/255, 204/255, 51/255),
        (255/255, 204/255, 51/255),
        (255/255, 102/255, 51/255)
    ]
)

# === 5) Compute plane_val using Radius Outlier Removal + statistic ===
def compute_plane_val(pts_cam, plane_type, eps):
    depths = np.linalg.norm(pts_cam, axis=1)
    if depths.size == 0:
        return 0.0
    # remove radius outliers
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_cam)
    clean, _ = pcd.remove_radius_outlier(nb_points=5, radius=eps)
    pts2 = np.asarray(clean.points)
    d2 = np.linalg.norm(pts2, axis=1) if pts2.size else depths
    # return min or median
    return float(d2.min() if plane_type == "min" else np.median(d2))

# === 6) Visualize one plane_type ===
def visualize_plane(img_orig, u, v, cam_xyz, bboxes, model, plane_type, eps):
    dists_all = np.linalg.norm(cam_xyz, axis=1)
    dmin, dmax = float(dists_all.min()), float(dists_all.max())
    img = img_orig.copy()

    for obj in bboxes:
        x1,y1,x2,y2 = map(int, obj['bbox'])
        cid = obj['class_id']

        # draw class label + box
        cv2.putText(img, model.names[cid], (x1, y1-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255,255,255), 2, cv2.LINE_AA)
        cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)

        # select points in bbox
        mask = (u>=x1)&(u<=x2)&(v>=y1)&(v<=y2)
        pts_u = u[mask].astype(int)
        pts_v = v[mask].astype(int)
        pts_cam_box = cam_xyz[mask]

        # compute plane_val
        plane_val = compute_plane_val(pts_cam_box, plane_type, eps)

        # band mask ±eps
        pts_d = dists_all[mask]
        band_mask = np.abs(pts_d - plane_val) < eps

        # scatter band points
        for ui,vi,di in zip(pts_u[band_mask], pts_v[band_mask], pts_d[band_mask]):
            norm = (di - dmin)/(dmax - dmin + 1e-6)
            r,g,b,_ = CUSTOM_CMAP(norm)
            color = (int(255*b), int(255*g), int(255*r))
            cv2.circle(img, (ui,vi), 1, color, -1)

        # draw distance text
        txt = f"{plane_val:.2f} m"
        (tw,th),_ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        pos = (x2 - tw, y1 - 6)
        cv2.putText(img, txt, pos,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0,0,255), 2, cv2.LINE_AA)

    return img

# === 7) Streamlit UI ===
files = st.file_uploader(
    "Upload: image (.png/.jpg), LiDAR (.bin), Calib (.txt)",
    type=['png','jpg','jpeg','bin','txt'],
    accept_multiple_files=True
)

if st.button("Run pipeline") and files:
    img_f   = next(f for f in files if f.name.lower().endswith(('png','jpg','jpeg')))
    lidar_f = next(f for f in files if f.name.lower().endswith('bin'))
    calib_f = next(f for f in files if f.name.lower().endswith('txt'))

    with tempfile.TemporaryDirectory() as tmp:
        p_img   = Path(tmp)/"img.png"
        p_bin   = Path(tmp)/"scan.bin"
        p_calib = Path(tmp)/"calib.txt"
        Image.open(img_f).convert('RGB').save(p_img)
        p_bin.write_bytes(lidar_f.read())
        p_calib.write_bytes(calib_f.read())

        # project LiDAR
        pc       = np.fromfile(str(p_bin), np.float32).reshape(-1,4)[:,:3]
        mats     = load_calib(p_calib)
        u, v, cam_xyz = project_lidar(pc, mats)

        # detect
        MODEL_PATH = "best.pt"  # adjust path
        model      = YOLO(MODEL_PATH)
        img_bgr    = cv2.imread(str(p_img))
        bboxes     = detect_bboxes(img_bgr, model, conf_thresh=0)

        # two tabs for Min & Median
        tab_min, tab_med = st.tabs(["Min (RadiusOutlier)", "Median (RadiusOutlier)"])
        with tab_min:
            vis = visualize_plane(img_bgr, u, v, cam_xyz, bboxes, model, "min", eps)
            st.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), use_container_width =True)
        with tab_med:
            vis = visualize_plane(img_bgr, u, v, cam_xyz, bboxes, model, "median", eps)
            st.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), use_container_width =True)
