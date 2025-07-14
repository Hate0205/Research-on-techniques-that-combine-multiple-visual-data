# Thesis: Nghiên cứu các kỹ thuật kết hợp nhiều dữ liệu hình ảnh

## Tổng quan
Dự án này nghiên cứu các phương pháp kết hợp ảnh 2D RGB và dữ liệu đám mây điểm 3D (từ LiDAR) để nâng cao hiệu quả nhận dạng đối tượng. Chúng tôi sử dụng YOLOv8 của Ultralytics và Open3D để triển khai và so sánh các chiến lược kết hợp khác nhau, đồng thời cung cấp một bản demo trên Streamlit để trực quan hóa kết quả một cách tương tác.

## Cấu trúc thư mục


## Cài đặt

1. **Clone repo**  
   ```bash
   git clone https://github.com/Hate0205/Research-on-techniques-that-combine-multiple-visual-data.git
   cd Research-on-techniques-that-combine-multiple-visual-data

2. Tạo và kích hoạt môi trường ảo
# Python ≥3.8
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate

3. Cài đặt phụ thuộc
pip install --upgrade pip
pip install -r requirements.txt

## Sử dụng 
1. Chạy ứng dụng Streamlit
streamlit run src/app.py
bạn có thể chỉnh MODEL_PATH trong app.py nếu cần
Ứng dụng cho phép chọn 3 file :
- ảnh gốc (RGB)
- file chứa dữ liệu Point Cloud .bin  
- file calib .txt 

2. Chạy các notebook & thí nghiệm
Mở các notebook trong src/ để thực thi từng bước:

Chuẩn bị dữ liệu:

  create_image_RGBD.ipynb
  
  read_img_rgbd.ipynb
  
  Convert_to_RGB_D.ipynb
  
  Split_Data.ipynb

Huấn luyện & đánh giá:

  Yolov8_4chan.ipynb (early fusion)
  
  Yolov8_on_RGBD_KITTI.ipynb
  
  Apdapter_train.ipynb

Inference & báo cáo kết quả:

  Yolov8_RGB_on_KITTI.ipynb
  
  Yolov8_test_with_kitti_dataset.ipynb
