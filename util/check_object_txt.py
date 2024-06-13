import os
import shutil

labels_folder = 'custom_dataset/labels'  
images_folder = 'dataset_yolov8x/images' 
output_folder = 'more_label_yolov8x'  

# Tạo thư mục đầu ra nếu nó chưa tồn tại
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

output_images_folder = os.path.join(output_folder, 'images')
output_labels_folder = os.path.join(output_folder, 'labels')

# Tạo thư mục con images và labels trong thư mục đầu ra nếu chúng chưa tồn tại
if not os.path.exists(output_images_folder):
    os.makedirs(output_images_folder)

if not os.path.exists(output_labels_folder):
    os.makedirs(output_labels_folder)

# Lặp qua tất cả các tệp tin .txt trong thư mục labels
for txt_file in os.listdir(labels_folder):
    if txt_file.endswith('.txt'):
        txt_path = os.path.join(labels_folder, txt_file)
        with open(txt_path, 'r') as file:
            lines = file.readlines()
            # Kiểm tra nếu tệp tin .txt chứa nhiều hơn 1 dòng (nhiều đối tượng)
            if len(lines) > 1:
                # Di chuyển tệp tin .txt và tệp tin ảnh có cùng tên vào thư mục labels và images trong thư mục output
                image_file = os.path.splitext(txt_file)[0] + '.jpg'  # Đặt tên tệp tin ảnh
                image_path = os.path.join(images_folder, image_file)
                new_txt_path = os.path.join(output_labels_folder, txt_file)
                new_image_path = os.path.join(output_images_folder, image_file)
                shutil.move(txt_path, new_txt_path)
                shutil.move(image_path, new_image_path)

print("Done")
