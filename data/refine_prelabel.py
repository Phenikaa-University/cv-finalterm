import os
import pandas as pd
from re import search
from PIL import Image
from ultralytics import YOLO



input_folder = "custom_dataset/images"
label_folder = 'custom_dataset/labels'
no_detection_folder = "custom_dataset/no_detection-yolov8x"
no_detection_images_folder = "custom_dataset/raw/no_detection-yolov8x/images"
no_detection_labels_folder = "custom_dataset/no_detection-yolov8x/labels"

if not os.path.exists(label_folder):
    os.makedirs(label_folder)
if not os.path.exists(no_detection_folder):
    os.makedirs(no_detection_folder)
if not os.path.exists(no_detection_images_folder):
    os.makedirs(no_detection_images_folder)
if not os.path.exists(no_detection_labels_folder):
    os.makedirs(no_detection_labels_folder)

# Đường dẫn đến tệp Excel chứa thông tin về ảnh và nhãn
excel_file = 'dataset/data_refine.xlsx'  # Thay đổi thành tên tệp Excel của bạn

model = YOLO('yolov8n.pt')  # pretrained YOLOv8n 

# Đọc dữ liệu từ tệp Excel
df = pd.read_excel(excel_file)

# Lặp qua từng dòng trong DataFrame và thực hiện prelabeling
for index, row in df.iterrows():
    img_url = row['img_url']
    label_detail = row['label_detail']

    # Tạo đường dẫn đầy đủ đến ảnh
    image_path = os.path.join(input_folder, os.path.basename(img_url))

    # Đảm bảo rằng tệp ảnh tồn tại
    if os.path.exists(image_path):
        # Thực hiện prelabeling bằng YOLO cho ảnh
        results = model(image_path, classes=53, save_txt=None)

        if len(results[0].boxes.xywhn) == 0:
            # Kiểm tra xem có phát hiện nào không
            txt_file = os.path.splitext(os.path.basename(img_url))[0] + '.txt'
            txt_path = os.path.join(no_detection_labels_folder, txt_file)
            img_dest_path = os.path.join(no_detection_images_folder, os.path.basename(img_url))

            # Di chuyển ảnh không có phát hiện vào thư mục no_detection/images
            os.rename(image_path, img_dest_path)

            with open(txt_path, 'w') as file:
                file.write(f"{label_detail}\n")  # Ghi label vào file .txt cho ảnh không có phát hiện
        else:
            # Tìm bounding box lớn nhất
            largest_box = max(results[0].boxes.xywhn, key=lambda x: x[3])

            txt_file = os.path.splitext(os.path.basename(img_url))[0] + '.txt'
            txt_path = os.path.join(label_folder, txt_file)

            x, y, w, h = largest_box[:4].tolist()

            with open(txt_path, 'w') as file:
                # for idx, prediction in enumerate(results[0].boxes.xywhn):
                    # x, y, w, h = prediction[:4].tolist()
                file.write(f"{label_detail} {x} {y} {w} {h}\n")

print("Convert Successfull!")


        