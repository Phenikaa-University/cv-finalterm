import os
import random
import shutil

# Đường dẫn đến thư mục gốc
root_folder = "custom_dataset2"

# Đường dẫn đến thư mục images và labels ban đầu
images_folder = os.path.join(root_folder, "images")
labels_folder = os.path.join(root_folder, "labels")

# Đường dẫn đến thư mục train, val, và test cho images và labels
train_images_folder = os.path.join(images_folder, "train")
val_images_folder = os.path.join(images_folder, "val")
test_images_folder = os.path.join(images_folder, "test")
train_labels_folder = os.path.join(labels_folder, "train")
val_labels_folder = os.path.join(labels_folder, "val")
test_labels_folder = os.path.join(labels_folder, "test")

# Tạo các thư mục nếu chưa tồn tại
os.makedirs(train_images_folder, exist_ok=True)
os.makedirs(val_images_folder, exist_ok=True)
os.makedirs(test_images_folder, exist_ok=True)
os.makedirs(train_labels_folder, exist_ok=True)
os.makedirs(val_labels_folder, exist_ok=True)
os.makedirs(test_labels_folder, exist_ok=True)

# Lấy danh sách tệp tin ảnh và nhãn
image_files = [f for f in os.listdir(images_folder) if f.endswith(".jpg")]
label_files = [f for f in os.listdir(labels_folder) if f.endswith(".txt")]

# Xáo trộn danh sách các tệp tin
random.shuffle(image_files)

# Tính số lượng ảnh cho tập train, validation và test (70/15/15)
total_images = len(image_files)
num_train = int(0.8 * total_images)
num_val = int(0.1 * total_images)
num_test = total_images - num_train - num_val

# Chia danh sách tệp tin ảnh và nhãn thành tập train, validation và test
train_image_files = image_files[:num_train]
val_image_files = image_files[num_train:num_train + num_val]
test_image_files = image_files[num_train + num_val:]

# Lọc danh sách tệp tin nhãn để chỉ giữ lại những tệp tin có cùng tên với tệp tin ảnh
train_label_files = [f.replace(".jpg", ".txt") for f in train_image_files if f.replace(".jpg", ".txt") in label_files]
val_label_files = [f.replace(".jpg", ".txt") for f in val_image_files if f.replace(".jpg", ".txt") in label_files]
test_label_files = [f.replace(".jpg", ".txt") for f in test_image_files if f.replace(".jpg", ".txt") in label_files]

# Di chuyển các tệp tin ảnh và nhãn vào các thư mục tập train, validation và test
for img_file, lbl_file in zip(train_image_files, train_label_files):
    shutil.move(os.path.join(images_folder, img_file), os.path.join(train_images_folder, img_file))
    shutil.move(os.path.join(labels_folder, lbl_file), os.path.join(train_labels_folder, lbl_file))

for img_file, lbl_file in zip(val_image_files, val_label_files):
    shutil.move(os.path.join(images_folder, img_file), os.path.join(val_images_folder, img_file))
    shutil.move(os.path.join(labels_folder, lbl_file), os.path.join(val_labels_folder, lbl_file))

for img_file, lbl_file in zip(test_image_files, test_label_files):
    shutil.move(os.path.join(images_folder, img_file), os.path.join(test_images_folder, img_file))
    shutil.move(os.path.join(labels_folder, lbl_file), os.path.join(test_labels_folder, lbl_file))

print("Chia dữ liệu thành công và tạo cấu trúc thư mục custom_dataset!")
