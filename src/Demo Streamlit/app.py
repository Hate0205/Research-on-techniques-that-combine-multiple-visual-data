import streamlit as st
import numpy as np
import cv2
from pathlib import Path
import tempfile
import datetime
import re
from ultralytics import YOLO
from matplotlib.colors import LinearSegmentedColormap

# === 0) Đường dẫn tới 2 model ===
MODEL_DET_PATH = (
    r"D:\study\Thesis\Thesis\results_experiments\Yolov8"
    r"\Using_Adapter_to_Convert_3_chanels"
    r"\Log_Train\yolov8n-RGBD\seed_42\weights\best.pt"
)
MODEL_SEG_PATH = "yolov8n-seg.pt"

# === 1) Sidebar: cấu hình dữ liệu & tham số ===
st.sidebar.header("Cấu hình KITTI → Video")
raw_root   = Path(st.sidebar.text_input("Thư mục KITTI raw", ""))
calib_dir  = Path(st.sidebar.text_input("Thư mục calibration", ""))
fps        = st.sidebar.number_input("FPS cho video xuất", 1, 30, 10)
voxel_size = st.sidebar.slider("Voxel size (m)", 0.05, 1.0, 0.1, 0.05)
# radius slider: max = 5, default = 1
radius     = st.sidebar.slider(
    "Radius (m) for neighbor search", 0.1, 5.0, 0.5, 0.1
)

st.title("KITTI → Det+Seg YOLOv8 + LIDAR + RadiusNN (Min/Median/Max/Mean)")

# === 2) Parse timestamps.txt ===
def parse_timestamps(fpath: Path):
    lines = fpath.read_text().splitlines()
    ts = []
    for L in lines:
        if not L:
            continue
        date_str, frac = L.split('.')
        dt = datetime.datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
        ts.append(dt.timestamp() + float("0." + frac))
    return np.array(ts, dtype=np.float64)

# === 3) Load KITTI calibration ===
@st.cache_data
def load_calibration(cam2cam: Path, velo2cam: Path):
    numpat = r'[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?'
    mats = {}
    for L in cam2cam.read_text().splitlines():
        if ':' not in L:
            continue
        name, raw = L.split(':', 1)
        vals = np.array(re.findall(numpat, raw), dtype=np.float32)
        if name.strip() == "R_rect_02":
            mats['R0'] = vals.reshape(3, 3)
        elif name.strip() == "P_rect_02":
            mats['P2'] = vals.reshape(3, 4)
    for L in velo2cam.read_text().splitlines():
        if ':' not in L:
            continue
        name, raw = L.split(':', 1)
        vals = np.array(re.findall(numpat, raw), dtype=np.float32)
        if name.strip() == "R":
            R = vals.reshape(3, 3)
        elif name.strip() == "T":
            T = vals.reshape(3, 1)
    mats['Tr'] = np.hstack([R, T])
    assert all(k in mats for k in ('P2', 'R0', 'Tr')), "Thiếu calibration"
    return mats

# === 4) Voxel centroid downsample ===
def voxel_downsample_centroid(pts: np.ndarray, vsz: float):
    idxs = np.floor(pts / vsz).astype(int)
    vox = {}
    for I, p in zip(map(tuple, idxs), pts):
        vox.setdefault(I, []).append(p)
    return np.array([np.mean(v, axis=0) for v in vox.values()], dtype=np.float32)

# === 5) Project LiDAR → pixel + cam_xyz ===
def project_lidar(xyz: np.ndarray, mats: dict):
    P2, R0, Tr = mats['P2'], mats['R0'], mats['Tr']
    ones = np.ones((xyz.shape[0], 1), np.float32)
    Xh = np.hstack([xyz, ones]).T
    cam = R0 @ (Tr @ Xh)
    Ch  = np.vstack([cam, ones.T])
    Y   = P2 @ Ch
    Yt  = Y.T
    valid = Yt[:, 2] > 0
    uv = (Yt[valid, :2] / Yt[valid, 2:3]).astype(int)
    return uv[:, 0], uv[:, 1], cam.T[valid]

# === 6) Load YOLO models ===
@st.cache_resource
def load_det_model():
    return YOLO(MODEL_DET_PATH)

@st.cache_resource
def load_seg_model():
    return YOLO(MODEL_SEG_PATH)

# === 7) Detect bounding boxes ===
def detect_bboxes(img: np.ndarray, model, conf_thresh=0.0):
    res  = model.predict(img, task='detect', verbose=False)[0]
    xyxy = res.boxes.xyxy.cpu().numpy()
    conf = res.boxes.conf.cpu().numpy().tolist()
    cls  = res.boxes.cls.cpu().numpy().astype(int).tolist()
    bbs = []
    for b, f, c in zip(xyxy, conf, cls):
        if f >= conf_thresh:
            bbs.append({
                "bbox": list(map(float, b)),
                "class_id": c,
                "confidence": float(f)
            })
    return bbs

# === 8) Detect segments for Car/Person/Bicycle ===
def detect_segments(img: np.ndarray, model, conf_thresh=0.0):
    H, W = img.shape[:2]
    res = model.predict(img, task='segment', verbose=False)[0]
    if res.masks is None:
        st.error("Seg-model không hỗ trợ masks")
        return []
    raw_masks = res.masks.data.cpu().numpy()
    xyxy      = res.boxes.xyxy.cpu().numpy()
    conf      = res.boxes.conf.cpu().numpy().tolist()
    cls       = res.boxes.cls.cpu().numpy().astype(int).tolist()
    allowed   = {0, 1, 2}  # COCO idx: 0=person,1=bicycle,2=car

    segs = []
    for m_raw, b, f, c in zip(raw_masks, xyxy, conf, cls):
        if f < conf_thresh or c not in allowed:
            continue
        m = cv2.resize(
            m_raw.astype(np.uint8),
            (W, H),
            interpolation=cv2.INTER_NEAREST
        ).astype(bool)
        segs.append({
            "mask": m,
            "bbox": list(map(float, b))
        })
    return segs

# === 9) IoU helper ===
def bbox_iou(A, B):
    x1, y1, x2, y2   = A
    x1b, y1b, x2b, y2b = B
    xi1, yi1 = max(x1, x1b), max(y1, y1b)
    xi2, yi2 = min(x2, x2b), min(y2, y2b)
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    areaA = (x2 - x1) * (y2 - y1)
    areaB = (x2b - x1b) * (y2b - y1b)
    return inter / (areaA + areaB - inter + 1e-9)

# === 10) VideoWriters ===
def make_writers(out_dir: Path, W: int, H: int, fps: int):
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    modes  = ["min", "median", "max", "mean"]
    return {
        m: cv2.VideoWriter(
            str(out_dir / f"out_{m}.mp4"), fourcc, fps, (W, H)
        )
        for m in modes
    }

# === 11) Colormap & stats ===
CMAP = LinearSegmentedColormap.from_list("cmap", [
    (102/255, 204/255, 51/255),
    (255/255, 204/255, 51/255),
    (255/255, 102/255, 51/255)
])

def compute_val(ds: np.ndarray, mode: str) -> float:
    if ds.size == 0:
        return 0.0
    return {
        "min":    float(ds.min()),
        "median": float(np.median(ds)),
        "max":    float(ds.max()),
        "mean":   float(ds.mean())
    }[mode]

# === 12) Visualize per-frame with Radius NN ===
def visualize(img, u, v, cam_xyz, bbs, segs, mode, radius, det_model):
    d_all = np.linalg.norm(cam_xyz, axis=1)
    dmin, dmax = float(d_all.min()), float(d_all.max())
    H, W = img.shape[:2]
    out = img.copy()

    # only points inside image
    valid = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u0, v0, d0   = u[valid], v[valid], d_all[valid]
    xyz0         = cam_xyz[valid]

    # draw all detection bboxes
    for o in bbs:
        x1, y1, x2, y2 = map(int, o["bbox"])
        cid = o["class_id"]
        cls_name = det_model.names[cid]   # tên class
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(out, cls_name, (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)

    # for each detection, match best seg-mask by IoU
    for o in bbs:
        best_mask, best_iou = None, 0.0
        for s in segs:
            iou = bbox_iou(o["bbox"], s["bbox"])
            if iou > best_iou:
                best_iou, best_mask = iou, s["mask"]
        if best_mask is None or best_iou < 0.1:
            continue

        # segment points
        seg_mask = best_mask[v0, u0]
        pu, pv, pd = u0[seg_mask], v0[seg_mask], d0[seg_mask]
        xyz_seg    = xyz0[seg_mask]
        if pd.size == 0:
            continue

        # compute 3D centroid of segment
        center3D = np.mean(xyz_seg, axis=0)

        # Radius NN: pick points within radius
        d2c    = np.linalg.norm(xyz_seg - center3D, axis=1)
        r_mask = d2c <= radius
        pu_rn, pv_rn, pd_rn = pu[r_mask], pv[r_mask], pd[r_mask]

        # compute statistic on pd_rn
        val = compute_val(pd_rn, mode)

        # draw entire segment point cloud
        for ui, vi, di in zip(pu, pv, pd):
            norm = (di - dmin) / (dmax - dmin + 1e-6)
            r, g, b, _ = CMAP(norm)
            cv2.circle(out, (int(ui),int(vi)), 1,
                       (int(255*b), int(255*g), int(255*r)), -1)

        # highlight on subset
        if pd_rn.size > 0:
            idx0 = int(np.argmin(np.abs(pd_rn - val)))
            cx, cy = int(pu_rn[idx0]), int(pv_rn[idx0])
            cv2.circle(out, (cx, cy), 6, (0,0,255), -1)

            txt2 = f"{val:.2f} m"
            (tw, th), _ = cv2.getTextSize(
                txt2, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            org = (int(o["bbox"][2] - tw), int(o["bbox"][1] - 6))
            cv2.putText(out, txt2, org,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

    return out

# === 13) Main pipeline ===
if st.button("Run on KITTI sequence"):
    img_dir   = raw_root/"image_02"/"data"
    ts_img    = raw_root/"image_02"/"timestamps.txt"
    velo_dir  = raw_root/"velodyne_points"/"data"
    ts_velo   = raw_root/"velodyne_points"/"timestamps.txt"
    cam2cam   = calib_dir/"calib_cam_to_cam.txt"
    velo2cam  = calib_dir/"calib_velo_to_cam.txt"

    imgs   = sorted(img_dir.glob("*.png"))
    bins   = sorted(velo_dir.glob("*.txt"))
    t_img  = parse_timestamps(ts_img)
    t_vel  = parse_timestamps(ts_velo)
    n      = len(imgs)

    mats      = load_calibration(cam2cam, velo2cam)
    det_model = load_det_model()
    seg_model = load_seg_model()

    H, W    = cv2.imread(str(imgs[0])).shape[:2]
    outdir  = Path(tempfile.mkdtemp())
    writers = make_writers(outdir, W, H, fps)
    prog    = st.progress(0.0)
    videos  = {}

    for i in range(n):
        j    = int(np.argmin(np.abs(t_vel - t_img[i])))
        img  = cv2.imread(str(imgs[i]))
        pc   = np.loadtxt(str(bins[j]), dtype=np.float32).reshape(-1,4)[:,:3]

        pc2       = voxel_downsample_centroid(pc, voxel_size)
        u, v, cam = project_lidar(pc2, mats)

        bbs  = detect_bboxes(img, det_model, 0.0)
        segs = detect_segments(img, seg_model, 0.0)

        for mode, wr in writers.items():
            frame = visualize(img, u, v, cam, bbs, segs, mode, radius, det_model)
            wr.write(frame)

        prog.progress((i + 1) / n)

    for wr in writers.values():
        wr.release()
    for m in writers.keys():
        videos[m] = (outdir/f"out_{m}.mp4").read_bytes()
    st.session_state['videos'] = videos

# === 14) Hiển thị & download ===
if 'videos' in st.session_state:
    for m, vid in st.session_state['videos'].items():
        st.video(vid)
        st.download_button(
            label=f"Tải video {m.upper()}",
            data=vid,
            file_name=f"out_{m}.mp4",
            mime="video/mp4",
            key=m
        )
