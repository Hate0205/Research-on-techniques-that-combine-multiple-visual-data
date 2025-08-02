# Thesis: Nghiên cứu các kỹ thuật kết hợp nhiều dữ liệu hình ảnh

## Tổng quan
Dự án này nghiên cứu các phương pháp kết hợp ảnh 2D RGB và dữ liệu đám mây điểm 3D (từ LiDAR) để tính được khoảng cách từ camera đến các đối tượng và nâng cao hiệu quả nhận dạng đối tượng. Chúng tôi sử dụng YOLOv8 của Ultralytics và Open3D để triển khai và so sánh các chiến lược kết hợp khác nhau, đồng thời cung cấp một bản demo trên Streamlit để trực quan hóa kết quả.

## Dữ liệu gốc (raw_data)

Do dung lượng rất lớn, bộ dữ liệu gốc không được lưu trong repo. Bạn cần tải thủ công từ KITTI Object Detection benchmark:

> https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d

Cần tải 4 gói:

- **Left color images of object data set** (12 GB)  
  → Ảnh RGB (thư mục `image_2/`)  
- **Velodyne point clouds** (29 GB)  
  → Dữ liệu point-cloud (thư mục `data_object_velodyne/`)  
- **Camera calibration matrices of object data set** (16 MB)  
  → Ma trận hiệu chỉnh camera (thư mục `data_object_calib/`)  
- **Training labels of object data set** (5 MB)  
  → Nhãn training (thư mục `data_object_label_2/`)


## Cài đặt

1. **Clone repo**  
   ```bash
   git clone https://github.com/Hate0205/Research-on-techniques-that-combine-multiple-visual-data.git
   cd Research-on-techniques-that-combine-multiple-visual-data

2. **Tạo và kích hoạt môi trường ảo**
  ### Python ≥3.8
  ```bash
  python -m venv venv
  ```
  ### Windows
  ```bash
  venv\Scripts\activate
  ```
  #### macOS/Linux
  ```bash
  source venv/bin/activate
  ```
  
  3. Cài đặt phụ thuộc
  ```bash
  pip install --upgrade pip
  pip install -r requirements.txt
  ```

